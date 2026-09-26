"""Play a trained policy in multiple environments with random velocity commands."""

import argparse

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Go2-Rough-Teleop-v0")
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--terrain_rows", type=int, default=6)
parser.add_argument("--terrain_cols", type=int, default=6)
parser.add_argument("--no_export", action="store_true", help="Disable exporting the loaded policy to JIT and ONNX.")
parser.add_argument("--sim2sim_log", default=None, help="Path to save an Isaac Sim rollout for sim2sim comparison.")
parser.add_argument("--sim2sim_log_steps", type=int, default=1000, help="Number of policy steps to record when --sim2sim_log is set.")
parser.add_argument("--sim2sim_log_env", type=int, default=0, help="Environment index to record when --sim2sim_log is set.")
timing_group = parser.add_mutually_exclusive_group()
timing_group.add_argument("--real_time", dest="real_time", action="store_true")
timing_group.add_argument("--no_real_time", dest="real_time", action="store_false")
parser.set_defaults(real_time=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.terrain_rows < 1 or args_cli.terrain_cols < 1:
    parser.error("--terrain_rows and --terrain_cols must be positive integers")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import mglf_lab  # noqa: F401


def get_policy_normalizer(policy_nn):
    if hasattr(policy_nn, "actor_obs_normalizer"):
        return policy_nn.actor_obs_normalizer
    if hasattr(policy_nn, "student_obs_normalizer"):
        return policy_nn.student_obs_normalizer
    return None


def _tensor_row(value, env_id):
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if value.ndim == 0:
        return value.detach().cpu()
    return value[env_id].detach().cpu()


def _append_if_present(log, key, value, env_id):
    row = _tensor_row(value, env_id)
    if row is not None:
        log[key].append(row)


def _get_policy_obs(obs):
    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"]
        if len(obs) == 1:
            return next(iter(obs.values()))
    if isinstance(obs, (tuple, list)) and len(obs) > 0:
        return _get_policy_obs(obs[0])
    return obs


def _init_sim2sim_log(env, checkpoint):
    if args_cli.sim2sim_log is None:
        return None
    if args_cli.sim2sim_log_env < 0 or args_cli.sim2sim_log_env >= env.num_envs:
        raise ValueError(f"--sim2sim_log_env must be in [0, {env.num_envs - 1}]")

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    return {
        "meta": {
            "source": "isaacsim",
            "task": args_cli.task,
            "checkpoint": checkpoint,
            "dt": float(unwrapped.step_dt),
            "env_id": int(args_cli.sim2sim_log_env),
            "joint_names": list(getattr(robot.data, "joint_names", [])),
        },
        "step": [],
        "obs": [],
        "action": [],
        "joint_pos": [],
        "joint_vel": [],
        "default_joint_pos": [],
        "root_quat_w": [],
        "root_ang_vel_b": [],
        "root_ang_vel_w": [],
        "root_lin_vel_w": [],
        "command": [],
    }


def _record_sim2sim_step(log, env, obs, actions, step_id):
    if log is None:
        return
    env_id = log["meta"]["env_id"]
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    data = robot.data

    log["step"].append(torch.tensor(step_id, dtype=torch.int64))
    _append_if_present(log, "obs", _get_policy_obs(obs), env_id)
    _append_if_present(log, "action", actions, env_id)
    _append_if_present(log, "joint_pos", getattr(data, "joint_pos", None), env_id)
    _append_if_present(log, "joint_vel", getattr(data, "joint_vel", None), env_id)
    _append_if_present(log, "default_joint_pos", getattr(data, "default_joint_pos", None), env_id)
    _append_if_present(log, "root_quat_w", getattr(data, "root_quat_w", None), env_id)
    _append_if_present(log, "root_ang_vel_b", getattr(data, "root_ang_vel_b", None), env_id)
    _append_if_present(log, "root_ang_vel_w", getattr(data, "root_ang_vel_w", None), env_id)
    _append_if_present(log, "root_lin_vel_w", getattr(data, "root_lin_vel_w", None), env_id)

    command = None
    if hasattr(unwrapped, "command_manager"):
        try:
            command = unwrapped.command_manager.get_command("base_velocity")
        except Exception:
            command = None
    _append_if_present(log, "command", command, env_id)


def _save_sim2sim_log(log):
    if log is None:
        return
    output = {"meta": log["meta"]}
    for key, values in log.items():
        if key == "meta":
            continue
        output[key] = torch.stack(values) if values else torch.empty(0)

    path = os.path.abspath(args_cli.sim2sim_log)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(output, path)
    print(f"[INFO] Saved Isaac Sim sim2sim log to: {path}")


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    cli_args.configure_play_terrain(env_cfg, args_cli.terrain_rows, args_cli.terrain_cols)
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

    if not args_cli.no_export:
        export_model_dir = os.path.join(os.path.dirname(checkpoint), "exported")
        normalizer = get_policy_normalizer(policy_nn)
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
        print(f"[INFO] Exported policy to: {export_model_dir}")

    # Do not overwrite the command buffer here. The environment command manager
    # independently samples vx, vy and yaw rate for every environment.
    obs = env.get_observations()
    sim2sim_log = _init_sim2sim_log(env, checkpoint)
    dt = env.unwrapped.step_dt
    step_id = 0
    try:
        while simulation_app.is_running():
            start = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                _record_sim2sim_step(sim2sim_log, env, obs, actions, step_id)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)
                step_id += 1
                if sim2sim_log is not None and step_id >= args_cli.sim2sim_log_steps:
                    break
            if args_cli.real_time:
                delay = dt - (time.time() - start)
                if delay > 0:
                    time.sleep(delay)
    finally:
        _save_sim2sim_log(sim2sim_log)
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
