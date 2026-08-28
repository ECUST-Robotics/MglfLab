"""Gym task registrations."""

import gymnasium as gym


gym.register(
    id="Go2-Rough-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "mglf_lab.tasks.go2_rough_env_cfg:Go2RoughTeleopEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "mglf_lab.tasks.rsl_rl_ppo_cfg:Go2RoughTeleopPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Go2-Rough-Teleop-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "mglf_lab.tasks.go2_rough_env_cfg:Go2RoughTeleopEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "mglf_lab.tasks.rsl_rl_ppo_cfg:Go2RoughTeleopPPORunnerCfg"
        ),
    },
)
