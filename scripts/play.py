"""Play a trained policy in multiple environments with random velocity commands."""

import argparse

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Go2-Rough-Teleop-Play-v0")
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--seed", type=int, default=1)
timing_group = parser.add_mutually_exclusive_group()
timing_group.add_argument("--real_time", dest="real_time", action="store_true")
timing_group.add_argument("--no_real_time", dest="real_time", action="store_false")
parser.set_defaults(real_time=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import mglf_lab  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        checkpoint = retrieve_file_path(args_cli.checkpoint)
    else:
        checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading model checkpoint from: {checkpoint}")
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy

    # Do not overwrite the command buffer here. The environment command manager
    # independently samples vx, vy and yaw rate for every environment.
    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    try:
        while simulation_app.is_running():
            start = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)
            if args_cli.real_time:
                delay = dt - (time.time() - start)
                if delay > 0:
                    time.sleep(delay)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
