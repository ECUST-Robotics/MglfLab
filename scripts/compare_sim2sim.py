"""Compare Isaac Sim and MuJoCo sim2sim rollout logs."""

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_JOINT_NAMES = [
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
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaac",
        default="/home/mglf/rc/MglfLab/logs/sim2sim/go2w_handstand_isaac.pt",
        help="Isaac Sim rollout saved by scripts/play.py --sim2sim_log.",
    )
    parser.add_argument(
        "--mujoco",
        default="/home/mglf/rc/Mglf_sar/logs/sim2sim/go2w_handstand_mujoco.csv",
        help="MuJoCo rollout CSV saved by RL_SAR_SIM2SIM_LOG.",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/mglf/rc/MglfLab/logs/sim2sim/compare",
        help="Directory for figures and summary.txt.",
    )
    parser.add_argument("--max_steps", type=int, default=0, help="Limit comparison to this many steps. 0 means all common steps.")
    return parser.parse_args()


def as_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_isaac(path):
    data = torch.load(path, map_location="cpu")
    loaded = {"meta": data.get("meta", {})}
    for key, value in data.items():
        if key == "meta":
            continue
        loaded[key] = as_numpy(value).astype(np.float64)
    return loaded


def load_mujoco_csv(path):
    with open(path, "r", newline="") as f:
        first_line = f.readline()
        f.seek(0)
        lines = f.readlines()

    meta = {}
    data_lines = lines
    if first_line.startswith("#"):
        data_lines = lines[1:]
        for item in first_line[1:].strip().split(","):
            if "=" in item:
                key, value = item.split("=", 1)
                meta[key] = value

    reader = csv.DictReader(data_lines)
    rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {path}")

    columns = reader.fieldnames or []
    raw = {column: np.asarray([float(row[column]) for row in rows], dtype=np.float64) for column in columns}

    def collect(prefix):
        names = [name for name in columns if name.startswith(prefix + "_")]
        names = sorted(names, key=lambda name: int(name.rsplit("_", 1)[1]))
        if not names:
            return None
        return np.stack([raw[name] for name in names], axis=1)

    return {
        "meta": meta,
        "step": raw.get("step", np.arange(len(rows), dtype=np.float64)),
        "obs": collect("obs"),
        "action": collect("action"),
        "joint_pos": collect("joint_pos"),
        "joint_vel": collect("joint_vel"),
        "target_joint_pos": collect("target_joint_pos"),
        "target_joint_vel": collect("target_joint_vel"),
        "root_quat_w": collect("root_quat_w"),
        "root_ang_vel_b": collect("root_ang_vel_b"),
        "command": collect("command"),
    }


def common_slice(isaac, mujoco, key, max_steps):
    a = isaac.get(key)
    b = mujoco.get(key)
    if a is None or b is None or a.size == 0 or b.size == 0:
        return None, None
    steps = min(a.shape[0], b.shape[0])
    dims = min(a.shape[1] if a.ndim > 1 else 1, b.shape[1] if b.ndim > 1 else 1)
    if max_steps > 0:
        steps = min(steps, max_steps)
    a = a[:steps]
    b = b[:steps]
    if a.ndim == 1:
        return a, b
    return a[:, :dims], b[:, :dims]


def error_stats(a, b):
    diff = b - a
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def plot_series(out_dir, name, isaac, mujoco, labels=None, max_cols=14):
    if isaac is None or mujoco is None:
        return
    if isaac.ndim == 1:
        isaac = isaac[:, None]
        mujoco = mujoco[:, None]

    cols = min(isaac.shape[1], max_cols)
    fig, axes = plt.subplots(cols, 1, figsize=(12, max(2.2 * cols, 4)), sharex=True)
    if cols == 1:
        axes = [axes]

    t = np.arange(isaac.shape[0])
    for i in range(cols):
        label = labels[i] if labels and i < len(labels) else str(i)
        axes[i].plot(t, isaac[:, i], label="Isaac", linewidth=1.2)
        axes[i].plot(t, mujoco[:, i], label="MuJoCo", linewidth=1.0, alpha=0.85)
        axes[i].set_ylabel(label)
        axes[i].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("policy step")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=160)
    plt.close(fig)


def plot_error_heatmap(out_dir, name, isaac, mujoco):
    if isaac is None or mujoco is None or isaac.ndim == 1:
        return
    diff = np.abs(mujoco - isaac)
    fig, ax = plt.subplots(figsize=(12, 4))
    image = ax.imshow(diff.T, aspect="auto", interpolation="nearest")
    ax.set_title(f"{name} abs error")
    ax.set_xlabel("policy step")
    ax.set_ylabel("dim")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_abs_error.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    isaac = load_isaac(args.isaac)
    mujoco = load_mujoco_csv(args.mujoco)

    keys = ["obs", "action", "joint_pos", "joint_vel", "root_quat_w", "root_ang_vel_b", "command"]
    lines = []
    lines.append(f"Isaac:  {args.isaac}")
    lines.append(f"MuJoCo: {args.mujoco}")
    lines.append(f"Isaac meta:  {isaac.get('meta', {})}")
    lines.append(f"MuJoCo meta: {mujoco.get('meta', {})}")
    lines.append("")
    lines.append("Error statistics use MuJoCo - Isaac over common steps/dims.")

    for key in keys:
        a, b = common_slice(isaac, mujoco, key, args.max_steps)
        if a is None or b is None:
            lines.append(f"{key}: missing")
            continue
        stats = error_stats(a, b)
        lines.append(
            f"{key}: shape={a.shape}, mae={stats['mae']:.6g}, rmse={stats['rmse']:.6g}, max_abs={stats['max_abs']:.6g}"
        )

        labels = DEFAULT_JOINT_NAMES if key in {"action", "joint_pos", "joint_vel"} else None
        plot_series(out_dir, key, a, b, labels=labels)
        plot_error_heatmap(out_dir, key, a, b)

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved figures and summary to: {out_dir}")


if __name__ == "__main__":
    main()
