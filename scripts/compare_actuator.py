"""Compare MuJoCo actuator test data against Isaac replay data."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


JOINT_NAMES = [
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "FR_wheel",
    "FL_wheel",
    "RR_wheel",
    "RL_wheel",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mujoco", default="/home/mglf/rc/Mglf_sar/logs/sim2sim/go2w_actuator_mujoco.csv")
    parser.add_argument("--isaac", default="/home/mglf/rc/MglfLab/logs/sim2sim/go2w_actuator_isaac.csv")
    parser.add_argument("--out_dir", default="/home/mglf/rc/MglfLab/logs/sim2sim/actuator_compare")
    parser.add_argument("--max_steps", type=int, default=0)
    return parser.parse_args()


def load_csv(path):
    with open(path, "r", newline="") as f:
        first = f.readline()
        lines = f.readlines() if first.startswith("#") else [first] + f.readlines()
    reader = csv.DictReader(lines)
    rows = list(reader)

    def collect(prefix):
        names = sorted(
            [name for name in reader.fieldnames if name.startswith(prefix + "_")],
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        return np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)

    return {
        "target_pos": collect("target_pos"),
        "target_vel": collect("target_vel"),
        "joint_pos": collect("joint_pos"),
        "joint_vel": collect("joint_vel"),
        "tau_est": collect("tau_est"),
        "root_quat_w": collect("root_quat_w"),
        "root_ang_vel_b": collect("root_ang_vel_b"),
    }


def common(a, b, max_steps):
    steps = min(a.shape[0], b.shape[0])
    dims = min(a.shape[1], b.shape[1])
    if max_steps > 0:
        steps = min(steps, max_steps)
    return a[:steps, :dims], b[:steps, :dims]


def stats(a, b):
    diff = b - a
    return np.mean(np.abs(diff)), np.sqrt(np.mean(diff * diff)), np.max(np.abs(diff))


def plot(out_dir, key, mujoco, isaac, max_dims=16):
    dims = min(mujoco.shape[1], max_dims)
    fig, axes = plt.subplots(dims, 1, figsize=(12, max(4, 2.0 * dims)), sharex=True)
    if dims == 1:
        axes = [axes]
    t = np.arange(mujoco.shape[0])
    for i in range(dims):
        axes[i].plot(t, mujoco[:, i], label="MuJoCo target", linewidth=1.1)
        axes[i].plot(t, isaac[:, i], label="Isaac replay", linewidth=1.0, alpha=0.85)
        label = JOINT_NAMES[i] if i < len(JOINT_NAMES) else str(i)
        axes[i].set_ylabel(label)
        axes[i].grid(True, alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("step")
    fig.suptitle(key)
    fig.tight_layout()
    fig.savefig(out_dir / f"{key}.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mujoco = load_csv(args.mujoco)
    isaac = load_csv(args.isaac)

    lines = [f"MuJoCo: {args.mujoco}", f"Isaac:  {args.isaac}", "", "Isaac - MuJoCo errors:"]
    lines.append("Joint index order:")
    for i, name in enumerate(JOINT_NAMES):
        lines.append(f"  {i:02d}: {name}")
    lines.append("")
    for key in ["joint_pos", "joint_vel", "tau_est", "root_quat_w", "root_ang_vel_b"]:
        a, b = common(mujoco[key], isaac[key], args.max_steps)
        mae, rmse, max_abs = stats(a, b)
        lines.append(f"{key}: shape={a.shape}, mae={mae:.6g}, rmse={rmse:.6g}, max_abs={max_abs:.6g}")
        plot(out_dir, key, a, b)

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved actuator comparison to: {out_dir}")


if __name__ == "__main__":
    main()
