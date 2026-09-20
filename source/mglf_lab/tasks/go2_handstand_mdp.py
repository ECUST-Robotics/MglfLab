"""Reward terms for the Go2 handstand task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def handstand_feet_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    feet_height_error = torch.sum(torch.square(feet_height - target_height), dim=1)
    return torch.exp(-feet_height_error / std**2)


def handstand_feet_on_air(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    return torch.all(first_air, dim=1).float()


def handstand_feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    return torch.sum((last_air_time - threshold) * first_contact, dim=1)


def handstand_feet_no_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.all(~is_contact, dim=1).float()


def handstand_support_feet_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.all(is_contact, dim=1).float()


def base_lin_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=1)


def base_ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_w), dim=1)


def fore_hind_joint_alignment_l2(
    env: ManagerBasedRLEnv,
    front_asset_cfg: SceneEntityCfg,
    hind_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[front_asset_cfg.name]
    front_joint_pos = asset.data.joint_pos[:, front_asset_cfg.joint_ids]
    hind_joint_pos = asset.data.joint_pos[:, hind_asset_cfg.joint_ids]
    return torch.sum(torch.square(front_joint_pos - hind_joint_pos), dim=1)


def paired_joint_alignment_l2(
    env: ManagerBasedRLEnv,
    left_asset_cfg: SceneEntityCfg,
    right_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[left_asset_cfg.name]
    left_joint_pos = asset.data.joint_pos[:, left_asset_cfg.joint_ids]
    right_joint_pos = asset.data.joint_pos[:, right_asset_cfg.joint_ids]
    return torch.sum(torch.square(left_joint_pos - right_joint_pos), dim=1)


def handstand_orientation_l2(
    env: ManagerBasedRLEnv,
    target_gravity: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    target_gravity_tensor = torch.tensor(target_gravity, device=env.device)
    return torch.sum(torch.square(asset.data.projected_gravity_b - target_gravity_tensor), dim=1)
