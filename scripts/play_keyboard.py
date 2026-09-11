"""Play a trained policy and command one Go2 from the keyboard."""

import argparse

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Go2-Rough-Teleop-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--terrain_rows", type=int, default=6)
parser.add_argument("--terrain_cols", type=int, default=6)
timing_group = parser.add_mutually_exclusive_group()
timing_group.add_argument("--real_time", dest="real_time", action="store_true")
timing_group.add_argument("--no_real_time", dest="real_time", action="store_false")
parser.set_defaults(real_time=True)
parser.add_argument("--linear_step", type=float, default=0.2)
parser.add_argument("--yaw_step", type=float, default=0.2)
parser.add_argument(
    "--visualize_height_scan",
    action="store_true",
    help="Show the height-scanner ray hit points in the Isaac Sim viewport.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.terrain_rows < 1 or args_cli.terrain_cols < 1:
    parser.error("--terrain_rows and --terrain_cols must be positive integers")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time

import carb
import gymnasium as gym
import omni.appwindow
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import mglf_lab  # noqa: F401


class KeyboardVelocityController:
    """Persistent vx/vy/wz command increments for a single robot."""

    def __init__(self, command_buffer: torch.Tensor, linear_step: float, yaw_step: float):
        self.command_buffer = command_buffer
        self.command = torch.zeros_like(command_buffer)
        self.linear_step = linear_step
        self.yaw_step = yaw_step
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)
        self._print_help()

    def _print_help(self):
        print(
            "\nKeyboard velocity control (click the viewport first):\n"
            "  W/S: +vx/-vx    A/D: +vy/-vy    Q/E: +wz/-wz\n"
            "  SPACE: stop     R: reset command\n"
            "Commands persist after key release.\n"
        )

    def _on_event(self, event):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        key = event.input.name
        if key == "W":
            self.command[:, 0] += self.linear_step
        elif key == "S":
            self.command[:, 0] -= self.linear_step
        elif key == "A":
            self.command[:, 1] += self.linear_step
        elif key == "D":
            self.command[:, 1] -= self.linear_step
        elif key == "Q":
            self.command[:, 2] += self.yaw_step
        elif key == "E":
            self.command[:, 2] -= self.yaw_step
        elif key in {"SPACE", "R"}:
            self.command.zero_()
        else:
            return True
        self.command[:, 0].clamp_(-1.5, 1.5)
        self.command[:, 1].clamp_(-0.8, 0.8)
        self.command[:, 2].clamp_(-1.5, 1.5)
        print(f"command [vx, vy, wz] = {self.command[0].tolist()}")
        return True

    def close(self):
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            self._subscription = None

    def apply(self):
        """Override commands that the environment's sampler may have changed."""
        self.command_buffer.copy_(self.command)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    # Keyboard teleoperation should run continuously. Keep safety terminations
    # (for example, base contact) but do not reset solely because 20 s elapsed.
    env_cfg.terminations.time_out = None
    cli_args.configure_play_terrain(env_cfg, args_cli.terrain_rows, args_cli.terrain_cols)
    if args_cli.visualize_height_scan:
        env_cfg.scene.height_scanner.debug_vis = True
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        checkpoint = cli_args.resolve_checkpoint(args_cli.checkpoint, retrieve_file_path)
    else:
        checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading model checkpoint from: {checkpoint}")
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy

    command_buffer = env.unwrapped.command_manager.get_command("base_velocity")
    keyboard = KeyboardVelocityController(command_buffer, args_cli.linear_step, args_cli.yaw_step)
    dt = env.unwrapped.step_dt
    try:
        while simulation_app.is_running():
            start = time.time()
            # The command manager may resample internally. Override it and
            # recompute observations so the policy sees the teleop command.
            keyboard.apply()
            obs = env.get_observations()
            with torch.inference_mode():
                actions = policy(obs)
                _, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)
            if args_cli.real_time:
                delay = dt - (time.time() - start)
                if delay > 0:
                    time.sleep(delay)
    finally:
        keyboard.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
