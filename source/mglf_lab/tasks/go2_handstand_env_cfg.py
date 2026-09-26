"""Go2 flat-ground handstand task.

This ports RobotLab's A1 handstand reward structure onto MglfLab's local Go2
URDF asset without changing any existing MglfLab task configuration.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import mglf_lab.tasks.robotlab_velocity.mdp as mdp
from mglf_lab.tasks.robotlab_velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from mglf_lab.assets.go2 import UNITREE_GO2_URDF_CFG
from mglf_lab.tasks import go2_handstand_mdp


@configclass
class Go2HandstandRewardsCfg(RewardsCfg):
    """Reward terms for Go2 handstand training."""

    handstand_feet_height_exp = RewTerm(
        func=go2_handstand_mdp.handstand_feet_height_exp,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.0, "std": math.sqrt(0.25)},
    )

    handstand_feet_on_air = RewTerm(
        func=go2_handstand_mdp.handstand_feet_on_air,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="")},
    )

    handstand_feet_air_time = RewTerm(
        func=go2_handstand_mdp.handstand_feet_air_time,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 5.0},
    )

    handstand_feet_no_contact = RewTerm(
        func=go2_handstand_mdp.handstand_feet_no_contact,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )

    handstand_feet_contact_count = RewTerm(
        func=go2_handstand_mdp.handstand_feet_contact_count,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )

    handstand_support_feet_contact = RewTerm(
        func=go2_handstand_mdp.handstand_support_feet_contact,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )

    base_lin_vel_xy_l2 = RewTerm(func=go2_handstand_mdp.base_lin_vel_xy_l2, weight=0.0)

    base_ang_vel_l2 = RewTerm(func=go2_handstand_mdp.base_ang_vel_l2, weight=0.0)

    fore_hind_joint_alignment_l2 = RewTerm(
        func=go2_handstand_mdp.fore_hind_joint_alignment_l2,
        weight=0.0,
        params={
            "front_asset_cfg": SceneEntityCfg("robot", joint_names=[], preserve_order=True),
            "hind_asset_cfg": SceneEntityCfg("robot", joint_names=[], preserve_order=True),
        },
    )

    support_pair_alignment_l2 = RewTerm(
        func=go2_handstand_mdp.paired_joint_alignment_l2,
        weight=0.0,
        params={
            "left_asset_cfg": SceneEntityCfg("robot", joint_names=[], preserve_order=True),
            "right_asset_cfg": SceneEntityCfg("robot", joint_names=[], preserve_order=True),
        },
    )

    air_pair_alignment_l2 = RewTerm(
        func=go2_handstand_mdp.paired_joint_alignment_l2,
        weight=0.0,
        params={
            "left_asset_cfg": SceneEntityCfg("robot", joint_names=[], preserve_order=True),
            "right_asset_cfg": SceneEntityCfg("robot", joint_names=[], preserve_order=True),
        },
    )

    air_thigh_forward_limit_l2 = RewTerm(
        func=go2_handstand_mdp.joint_upper_limit_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[], preserve_order=True),
            "upper_limit": 1.15,
        },
    )

    handstand_orientation_l2 = RewTerm(
        func=go2_handstand_mdp.handstand_orientation_l2,
        weight=0.0,
        params={"target_gravity": []},
    )


@configclass
class Go2HandstandFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Train Go2 to perform a rear-leg handstand on flat ground."""

    rewards: Go2HandstandRewardsCfg = Go2HandstandRewardsCfg()

    handstand_type = "back"
    support_hip_deviation_weight = 0.0
    base_lin_vel_xy_weight = 0.0
    base_ang_vel_weight = 0.0
    joint_vel_weight = 0.0
    action_rate_weight = -0.05
    rear_leg_deviation_weight = 0.0
    fore_hind_alignment_weight = 0.0
    support_pair_alignment_weight = 0.0
    air_pair_alignment_weight = 0.0
    air_hip_deviation_weight = 0.0
    air_feet_contact_penalty_weight = 0.0
    air_thigh_forward_limit_weight = 0.0
    air_thigh_forward_limit = 1.15
    base_link_name = "base"
    foot_link_name = ".*_foot"
    joint_names = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0
        self.scene.robot = UNITREE_GO2_URDF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.38)
        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        }
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None

        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.base_lin_vel.scale = 2.0
        self.observations.critic.base_ang_vel.scale = 0.25
        self.observations.critic.joint_pos.scale = 1.0
        self.observations.critic.joint_vel.scale = 0.05
        self.observations.critic.base_lin_vel = None
        self.observations.critic.height_scan = None
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None

        self.rewards.is_terminated.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.base_height_l2.params["target_height"] = 0.35
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0.0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.joint_torques_l2.weight = -1.0e-3
        self.rewards.joint_vel_l2.weight = self.joint_vel_weight
        self.rewards.joint_acc_l2.weight = -2.5e-6
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.joint_power.weight = -2.0e-4
        self.rewards.stand_still.weight = 0.0
        self.rewards.action_rate_l2.weight = self.action_rate_weight
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_thigh"]
        self.rewards.contact_forces.weight = 0.0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.feet_contact.weight = 0.0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0.0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp

        air_foot_name, support_foot_name, target_gravity, target_height, orientation_weight = self._handstand_target()
        self.rewards.handstand_orientation_l2.weight = orientation_weight
        self.rewards.handstand_orientation_l2.params["target_gravity"] = target_gravity
        self.rewards.handstand_feet_height_exp.weight = 10.0
        self.rewards.handstand_feet_height_exp.params["target_height"] = target_height
        self.rewards.handstand_feet_height_exp.params["asset_cfg"].body_names = [air_foot_name]
        self.rewards.handstand_feet_on_air.weight = 5.0
        self.rewards.handstand_feet_on_air.params["sensor_cfg"].body_names = [air_foot_name]
        self.rewards.handstand_feet_air_time.weight = 5.0
        self.rewards.handstand_feet_air_time.params["sensor_cfg"].body_names = [air_foot_name]
        self.rewards.handstand_feet_no_contact.weight = 2.0
        self.rewards.handstand_feet_no_contact.params["sensor_cfg"].body_names = [air_foot_name]
        self.rewards.handstand_feet_contact_count.weight = self.air_feet_contact_penalty_weight
        self.rewards.handstand_feet_contact_count.params["sensor_cfg"].body_names = [air_foot_name]
        self.rewards.handstand_support_feet_contact.weight = 2.0
        self.rewards.handstand_support_feet_contact.params["sensor_cfg"].body_names = [support_foot_name]
        self.rewards.base_lin_vel_xy_l2.weight = self.base_lin_vel_xy_weight
        self.rewards.base_ang_vel_l2.weight = self.base_ang_vel_weight
        self.rewards.create_joint_deviation_l1_rewterm(
            "support_hip_deviation_l1", self.support_hip_deviation_weight, [self._support_hip_pattern()]
        )
        self.rewards.create_joint_deviation_l1_rewterm(
            "rear_leg_deviation_l1",
            self.rear_leg_deviation_weight,
            ["R.*_thigh_joint", "R.*_calf_joint"],
        )
        self.rewards.create_joint_deviation_l1_rewterm(
            "air_hip_deviation_l1", self.air_hip_deviation_weight, [self._air_hip_pattern()]
        )
        front_joint_names, hind_joint_names = self._fore_hind_alignment_joint_names()
        self.rewards.fore_hind_joint_alignment_l2.weight = self.fore_hind_alignment_weight
        self.rewards.fore_hind_joint_alignment_l2.params["front_asset_cfg"].joint_names = front_joint_names
        self.rewards.fore_hind_joint_alignment_l2.params["hind_asset_cfg"].joint_names = hind_joint_names
        support_left_joints, support_right_joints = self._support_pair_alignment_joint_names()
        self.rewards.support_pair_alignment_l2.weight = self.support_pair_alignment_weight
        self.rewards.support_pair_alignment_l2.params["left_asset_cfg"].joint_names = support_left_joints
        self.rewards.support_pair_alignment_l2.params["right_asset_cfg"].joint_names = support_right_joints
        air_left_joints, air_right_joints = self._air_pair_alignment_joint_names()
        self.rewards.air_pair_alignment_l2.weight = self.air_pair_alignment_weight
        self.rewards.air_pair_alignment_l2.params["left_asset_cfg"].joint_names = air_left_joints
        self.rewards.air_pair_alignment_l2.params["right_asset_cfg"].joint_names = air_right_joints
        self.rewards.air_thigh_forward_limit_l2.weight = self.air_thigh_forward_limit_weight
        self.rewards.air_thigh_forward_limit_l2.params["asset_cfg"].joint_names = [self._air_thigh_pattern()]
        self.rewards.air_thigh_forward_limit_l2.params["upper_limit"] = self.air_thigh_forward_limit

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.disable_zero_weight_rewards()

    def _handstand_target(self):
        """Return RobotLab-style target settings for the selected handstand type."""
        if self.handstand_type == "front":
            return "F.*_foot", "R.*_foot", [-1.0, 0.0, 0.0], 0.45, -1.5
        if self.handstand_type == "back":
            return "R.*_foot", "F.*_foot", [1.0, 0.0, 0.0], 0.5, -1.0
        if self.handstand_type == "left":
            return ".*L_foot", ".*R_foot", [0.0, -1.0, 0.0], 0.3, -1.0
        if self.handstand_type == "right":
            return ".*R_foot", ".*L_foot", [0.0, 1.0, 0.0], 0.3, -1.0
        raise ValueError(f"Unknown handstand_type: {self.handstand_type}")

    def _support_hip_pattern(self):
        """Return the hip joints on the feet that should stay on the ground."""
        if self.handstand_type == "front":
            return "R.*_hip_joint"
        if self.handstand_type == "back":
            return "F.*_hip_joint"
        if self.handstand_type == "left":
            return ".*R_hip_joint"
        if self.handstand_type == "right":
            return ".*L_hip_joint"
        raise ValueError(f"Unknown handstand_type: {self.handstand_type}")

    def _air_hip_pattern(self):
        """Return hip joints on the feet that should stay lifted."""
        if self.handstand_type == "front":
            return "F.*_hip_joint"
        if self.handstand_type == "back":
            return "R.*_hip_joint"
        if self.handstand_type == "left":
            return ".*L_hip_joint"
        if self.handstand_type == "right":
            return ".*R_hip_joint"
        raise ValueError(f"Unknown handstand_type: {self.handstand_type}")

    def _air_thigh_pattern(self):
        """Return thigh joints on the feet that should stay lifted."""
        if self.handstand_type == "front":
            return "F.*_thigh_joint"
        if self.handstand_type == "back":
            return "R.*_thigh_joint"
        if self.handstand_type == "left":
            return ".*L_thigh_joint"
        if self.handstand_type == "right":
            return ".*R_thigh_joint"
        raise ValueError(f"Unknown handstand_type: {self.handstand_type}")

    def _fore_hind_alignment_joint_names(self):
        """Return ordered front/hind joint names that should keep similar angles."""
        if self.handstand_type == "left":
            return (
                ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
                ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
            )
        if self.handstand_type == "right":
            return (
                ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
                ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
            )
        return ([], [])

    def _support_pair_alignment_joint_names(self):
        """Return left/right support leg joints that should keep similar angles."""
        if self.handstand_type == "front":
            return (
                ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
                ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
            )
        if self.handstand_type == "back":
            return (
                ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
                ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
            )
        return ([], [])

    def _air_pair_alignment_joint_names(self):
        """Return left/right lifted leg joints that should keep similar angles."""
        if self.handstand_type == "front":
            return (
                ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
                ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
            )
        if self.handstand_type == "back":
            return (
                ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
                ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
            )
        return ([], [])


@configclass
class Go2HandstandFrontFlatEnvCfg(Go2HandstandFlatEnvCfg):
    """Train Go2 with the front feet lifted and rear feet supporting."""

    handstand_type = "front"
    support_hip_deviation_weight = -0.8
    support_pair_alignment_weight = -1.3
    air_pair_alignment_weight = -0.8
    air_hip_deviation_weight = -0.8
    base_lin_vel_xy_weight = -1.0
    base_ang_vel_weight = -0.25
    joint_vel_weight = -2.0e-4
    action_rate_weight = -0.08


@configclass
class Go2HandstandBackFlatEnvCfg(Go2HandstandFlatEnvCfg):
    """Train Go2 with the rear feet lifted and front feet supporting."""

    handstand_type = "back"
    support_hip_deviation_weight = -0.8
    support_pair_alignment_weight = -1.3
    air_pair_alignment_weight = -0.8
    air_hip_deviation_weight = -0.8
    base_lin_vel_xy_weight = -0.8
    base_ang_vel_weight = -0.2
    joint_vel_weight = -1.0e-4
    action_rate_weight = -0.08


@configclass
class Go2HandstandLeftFlatEnvCfg(Go2HandstandFlatEnvCfg):
    """Train Go2 with the left feet lifted and right feet supporting."""

    handstand_type = "left"
    support_hip_deviation_weight = -0.4
    rear_leg_deviation_weight = -0.6
    fore_hind_alignment_weight = -0.8
    base_lin_vel_xy_weight = -0.5
    base_ang_vel_weight = -0.15
    joint_vel_weight = -1.0e-4
    action_rate_weight = -0.06


@configclass
class Go2HandstandRightFlatEnvCfg(Go2HandstandFlatEnvCfg):
    """Train Go2 with the right feet lifted and left feet supporting."""

    handstand_type = "right"
    support_hip_deviation_weight = -0.4
    rear_leg_deviation_weight = -0.6
    fore_hind_alignment_weight = -0.8
    base_lin_vel_xy_weight = -0.5
    base_ang_vel_weight = -0.15
    joint_vel_weight = -1.0e-4
    action_rate_weight = -0.06
