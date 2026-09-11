"""Shared RSL-RL command-line helpers."""

import argparse
import os
import re


def _latest_model_in_directory(directory: str) -> str:
    """Return the highest-numbered model_*.pt checkpoint in a run directory."""
    candidates = []
    for name in os.listdir(directory):
        match = re.fullmatch(r"model_(\d+)\.pt", name)
        if match:
            candidates.append((int(match.group(1)), os.path.join(directory, name)))
    if not candidates:
        raise FileNotFoundError(
            f"No model_*.pt checkpoint was found in run directory: {directory}"
        )
    return max(candidates, key=lambda item: item[0])[1]


def resolve_checkpoint(checkpoint: str, retrieve_file_path) -> str:
    """Resolve an explicit checkpoint file or select the latest file in a run directory."""
    local_path = os.path.abspath(os.path.expanduser(checkpoint))
    if os.path.isdir(local_path):
        return _latest_model_in_directory(local_path)
    if os.path.isfile(local_path):
        return local_path

    resolved_path = retrieve_file_path(checkpoint)
    if os.path.isdir(resolved_path):
        return _latest_model_in_directory(resolved_path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Checkpoint is not a file: {resolved_path}")
    return resolved_path


def configure_play_terrain(env_cfg, terrain_rows: int, terrain_cols: int):
    """Configure compact play terrain with one terrain type per column when possible."""
    terrain_generator = env_cfg.scene.terrain.terrain_generator
    if terrain_generator is None:
        return

    terrain_generator.num_rows = terrain_rows
    terrain_generator.num_cols = terrain_cols
    terrain_generator.curriculum = True

    sub_terrains = getattr(terrain_generator, "sub_terrains", None)
    if sub_terrains and terrain_cols == len(sub_terrains):
        for sub_terrain in sub_terrains.values():
            sub_terrain.proportion = 1.0

    if env_cfg.scene.terrain.max_init_terrain_level is not None:
        env_cfg.scene.terrain.max_init_terrain_level = min(
            env_cfg.scene.terrain.max_init_terrain_level,
            terrain_rows - 1,
        )


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("rsl_rl")
    group.add_argument("--experiment_name", type=str, default=None)
    group.add_argument("--run_name", type=str, default=None)
    group.add_argument("--resume", action="store_true", default=False)
    group.add_argument(
        "--reset_optimizer",
        action="store_true",
        default=False,
        help="Resume policy weights and iteration count without loading optimizer state.",
    )
    group.add_argument("--load_run", type=str, default=None)
    group.add_argument("--checkpoint", type=str, default=None)
    group.add_argument("--logger", choices={"wandb", "tensorboard", "neptune"}, default=None)
    group.add_argument("--log_project_name", type=str, default=None)


def update_rsl_rl_cfg(cfg, args):
    for arg_name, cfg_name in (
        ("seed", "seed"),
        ("run_name", "run_name"),
        ("load_run", "load_run"),
        ("checkpoint", "load_checkpoint"),
        ("logger", "logger"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(cfg, cfg_name, value)
    cfg.resume = getattr(args, "resume", False)
    if getattr(args, "experiment_name", None):
        cfg.experiment_name = args.experiment_name
    if cfg.logger in {"wandb", "neptune"} and getattr(args, "log_project_name", None):
        cfg.wandb_project = args.log_project_name
        cfg.neptune_project = args.log_project_name
    return cfg
