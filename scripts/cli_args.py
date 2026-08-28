"""Shared RSL-RL command-line helpers."""

import argparse


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
