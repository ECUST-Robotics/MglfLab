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


def degraded_height_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
    fill_value: float = 0.0,
    noise_std: float = 0.025,
    point_dropout_prob: float = 0.08,
    block_dropout_prob: float = 0.20,
    block_size_range: tuple[int, int] = (2, 5),
    early_step_count: int = 75,
    early_point_dropout_prob: float = 0.25,
    early_block_dropout_prob: float = 0.55,
    grid_shape: tuple[int, int] = (11, 17),
) -> torch.Tensor:
    """Return a height scan with missing-map and noise randomization.

    The deployment-side MID360 elevation map can be sparse while it is warming
    up or when noisy cells are rejected. This keeps the policy from depending
    on a perfectly dense RayCaster observation during training.
    """

    scan = finite_height_scan(env, sensor_cfg=sensor_cfg, offset=offset)
    if scan.numel() == 0:
        return scan

    if noise_std > 0.0:
        scan = scan + torch.randn_like(scan) * noise_std

    point_prob = point_dropout_prob
    block_prob = block_dropout_prob
    episode_length = getattr(env, "episode_length_buf", None)
    if episode_length is not None:
        early = episode_length < early_step_count
        if torch.any(early):
            point_prob = torch.full((env.num_envs, 1), point_dropout_prob, device=scan.device)
            block_prob = torch.full((env.num_envs,), block_dropout_prob, device=scan.device)
            point_prob[early] = early_point_dropout_prob
            block_prob[early] = early_block_dropout_prob

    if isinstance(point_prob, torch.Tensor):
        point_missing = torch.rand_like(scan) < point_prob
    else:
        point_missing = torch.rand_like(scan) < point_prob

    missing = point_missing
    rows, cols = grid_shape
    if rows * cols == scan.shape[1] and block_dropout_prob > 0.0:
        if not hasattr(env, "_mglf_height_scan_grid"):
            row_ids = torch.arange(rows, device=scan.device).repeat_interleave(cols)
            col_ids = torch.arange(cols, device=scan.device).repeat(rows)
            env._mglf_height_scan_grid = (row_ids, col_ids)
        row_ids, col_ids = env._mglf_height_scan_grid

        if isinstance(block_prob, torch.Tensor):
            has_block = torch.rand(env.num_envs, device=scan.device) < block_prob
        else:
            has_block = torch.rand(env.num_envs, device=scan.device) < block_prob
        block_envs = torch.nonzero(has_block, as_tuple=False).flatten()
        if block_envs.numel() > 0:
            min_size, max_size = block_size_range
            max_size = max(min(max_size, rows, cols), min_size)
            heights = torch.randint(min_size, max_size + 1, (block_envs.numel(),), device=scan.device)
            widths = torch.randint(min_size, max_size + 1, (block_envs.numel(),), device=scan.device)
            centers_r = torch.randint(0, rows, (block_envs.numel(),), device=scan.device)
            centers_c = torch.randint(0, cols, (block_envs.numel(),), device=scan.device)

            for i, env_id in enumerate(block_envs):
                half_h = heights[i] // 2
                half_w = widths[i] // 2
                block = (
                    (row_ids >= centers_r[i] - half_h)
                    & (row_ids <= centers_r[i] + half_h)
                    & (col_ids >= centers_c[i] - half_w)
                    & (col_ids <= centers_c[i] + half_w)
                )
                missing[env_id, block] = True

    env._mglf_height_scan_missing = missing
    return torch.where(missing, torch.as_tensor(fill_value, device=scan.device), scan)


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
