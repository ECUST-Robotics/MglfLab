"""Replay MuJoCo contact-test joint targets in Isaac Sim."""

import argparse
import csv
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Go2W-Flat-Handstand-Back-v0")
parser.add_argument("--mujoco_log", default="/home/mglf/rc/Mglf_sar/logs/sim2sim/go2w_contact_mujoco.csv")
parser.add_argument("--output", default="/home/mglf/rc/MglfLab/logs/sim2sim/go2w_contact_isaac.csv")
parser.add_argument("--steps", type=int, default=0)
parser.add_argument("--render_every", type=int, default=4, help="Render every N physics steps when not headless.")
parser.add_argument("--progress_every", type=int, default=100, help="Print progress every N steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import mglf_lab  # noqa: F401


MUJOCO_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
]


def read_targets(path):
    with open(path, "r", newline="") as f:
        first = f.readline()
        lines = f.readlines() if first.startswith("#") else [first] + f.readlines()
    reader = csv.DictReader(lines)
    rows = list(reader)

    def collect(prefix):
        names = sorted([n for n in reader.fieldnames if n.startswith(prefix + "_")], key=lambda n: int(n.rsplit("_", 1)[1]))
        return torch.tensor([[float(row[n]) for n in names] for row in rows], dtype=torch.float32)

    return collect("target_pos"), collect("target_vel")


def contact_summary(env):
    sensor = env.unwrapped.scene.sensors.get("contact_forces")
    if sensor is None:
        return 0, 0.0
    forces = sensor.data.net_forces_w[0]
    norms = torch.linalg.norm(forces, dim=-1)
    return int(torch.sum(norms > 1.0).item()), float(torch.sum(norms).item())


def main():
    target_pos, target_vel = read_targets(args_cli.mujoco_log)
    steps = target_pos.shape[0] if args_cli.steps <= 0 else min(args_cli.steps, target_pos.shape[0])

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.episode_length_s = max(float(getattr(env_cfg, "episode_length_s", 0.0)), 1.0e6)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    isaac_joint_names = list(robot.data.joint_names)
    num_dofs = min(target_pos.shape[1], robot.data.joint_pos.shape[1])
    isaac_joint_ids = [isaac_joint_names.index(name) for name in MUJOCO_JOINT_NAMES[:num_dofs]]

    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        f.write(f"# source=isaacsim,task={args_cli.task},mujoco_log={args_cli.mujoco_log},dt={env.unwrapped.step_dt},num_of_dofs={num_dofs}\n")
        writer = csv.writer(f)
        header = ["step", "time"]
        header += [f"target_pos_{i}" for i in range(num_dofs)]
        header += [f"target_vel_{i}" for i in range(num_dofs)]
        header += [f"joint_pos_{i}" for i in range(num_dofs)]
        header += [f"joint_vel_{i}" for i in range(num_dofs)]
        header += [f"root_quat_w_{i}" for i in range(4)]
        header += [f"root_ang_vel_b_{i}" for i in range(3)]
        header += ["contact_count", "total_contact_force_norm"]
        writer.writerow(header)

        for step in range(steps):
            if not simulation_app.is_running():
                print(f"[INFO] Simulation app stopped at step {step}")
                break

            q_target = target_pos[step : step + 1, :num_dofs].to(env.unwrapped.device)
            dq_target = target_vel[step : step + 1, :num_dofs].to(env.unwrapped.device)
            robot.set_joint_position_target(q_target, joint_ids=isaac_joint_ids)
            robot.set_joint_velocity_target(dq_target, joint_ids=isaac_joint_ids)
            env.unwrapped.scene.write_data_to_sim()
            render = (not args_cli.headless) and (args_cli.render_every > 0) and (step % args_cli.render_every == 0)
            env.unwrapped.sim.step(render=render)
            if render:
                env.unwrapped.sim.render()
            env.unwrapped.scene.update(env.unwrapped.physics_dt)
            contact_count, contact_force = contact_summary(env)

            row = [step, step * env.unwrapped.physics_dt]
            row += q_target[0].detach().cpu().tolist()
            row += dq_target[0].detach().cpu().tolist()
            row += robot.data.joint_pos[0, isaac_joint_ids].detach().cpu().tolist()
            row += robot.data.joint_vel[0, isaac_joint_ids].detach().cpu().tolist()
            row += robot.data.root_quat_w[0].detach().cpu().tolist()
            row += robot.data.root_ang_vel_b[0].detach().cpu().tolist()
            row += [contact_count, contact_force]
            writer.writerow(row)

            if args_cli.progress_every > 0 and (step + 1) % args_cli.progress_every == 0:
                print(f"[INFO] replay_contact_isaac step {step + 1}/{steps}", flush=True)

    env.close()
    print(f"[INFO] Saved Isaac contact replay to: {output}")


if __name__ == "__main__":
    main()
    simulation_app.close()
