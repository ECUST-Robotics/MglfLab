"""Print all environments registered by MglfLab."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="List MglfLab Gym environments.")
parser.add_argument(
    "--prefix",
    nargs="*",
    default=["Go2-", "Go2W-", "Mglf-", "UIKA-"],
    help="Task ID prefixes to include. Use an empty value to list all registered tasks.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.headless is None:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from prettytable import PrettyTable

import mglf_lab  # noqa: F401


def _matches_prefix(task_id: str) -> bool:
    """Return whether a task ID should be listed."""
    if not args_cli.prefix:
        return True
    return any(task_id.startswith(prefix) for prefix in args_cli.prefix)


def main():
    """Print all environments registered in the MglfLab extension."""
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Env Config", "RSL-RL Config"])
    table.title = "Available Environments in MglfLab"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Env Config"] = "l"
    table.align["RSL-RL Config"] = "l"

    index = 0
    for task_spec in sorted(gym.registry.values(), key=lambda spec: spec.id):
        if not _matches_prefix(task_spec.id):
            continue

        kwargs = task_spec.kwargs or {}
        table.add_row(
            [
                index + 1,
                task_spec.id,
                task_spec.entry_point,
                kwargs.get("env_cfg_entry_point", ""),
                kwargs.get("rsl_rl_cfg_entry_point", ""),
            ]
        )
        index += 1

    print(table)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
