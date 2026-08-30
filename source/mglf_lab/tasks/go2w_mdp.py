"""Numerically robust MDP terms used by the height-aware Go2W task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def finite_height_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
) -> torch.Tensor:
    """Return the local height scan without allowing missed rays to emit NaN/Inf."""

    sensor = env.scene.sensors[sensor_cfg.name]
    scan = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    # The observation term subsequently clips to [-1, 1]. Convert invalid ray
    # results first because clamp keeps NaN unchanged.
    return torch.nan_to_num(scan, nan=0.0, posinf=1.0, neginf=-1.0)


def command_conditioned_joint_deviation_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float = 5.0,
    velocity_threshold: float = 0.5,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """RobotLab-style leg posture cost, stronger while commanded to stand."""

    asset = env.scene[asset_cfg.name]
    command = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_velocity = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    deviation = torch.linalg.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids],
        dim=1,
    )
    penalty = torch.where(
        (command > command_threshold) | (body_velocity > velocity_threshold),
        deviation,
        stand_still_scale * deviation,
    )
    return penalty * upright_factor(env)


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """RobotLab-style default-pose cost for near-zero commands."""

    asset = env.scene[asset_cfg.name]
    deviation = torch.sum(
        torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    standing = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    return deviation * standing * upright_factor(env)


def joint_power_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize absolute mechanical power on the selected leg joints."""

    asset = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(
            asset.data.joint_vel[:, asset_cfg.joint_ids]
            * asset.data.applied_torque[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )


def diagonal_leg_mirror_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Encourage diagonal leg pairs to use symmetric joint magnitudes."""

    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    pairs = (
        ("FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"),
        ("FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"),
    )
    if not hasattr(env, "_go2w_mirror_joint_ids"):
        env._go2w_mirror_joint_ids = tuple(
            (
                asset.find_joints(left_pattern, preserve_order=True)[0],
                asset.find_joints(right_pattern, preserve_order=True)[0],
            )
            for left_pattern, right_pattern in pairs
        )
    penalty = torch.zeros(env.num_envs, device=joint_pos.device)
    for left_ids, right_ids in env._go2w_mirror_joint_ids:
        penalty += torch.sum(
            torch.square(torch.abs(joint_pos[:, left_ids]) - torch.abs(joint_pos[:, right_ids])),
            dim=1,
        )
    return penalty / len(pairs) * upright_factor(env)


def wheel_contact_without_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward wheel contact events while a zero velocity command is active."""

    sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(first_contact, dim=1).float()
    stopped = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    return reward * stopped * upright_factor(env)


def upright_factor(env: ManagerBasedRLEnv) -> torch.Tensor:
    """RobotLab's orientation gate for posture and gait-related terms."""

    return torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7


def upward_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RobotLab-style positive reward for keeping the base's up-axis upright."""

    asset = env.scene[asset_cfg.name]
    return torch.square(1.0 - asset.data.projected_gravity_b[:, 2])
