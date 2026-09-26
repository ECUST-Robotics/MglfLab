"""Go2W flat-ground handstand tasks."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from mglf_lab.assets.go2w import UNITREE_GO2W_CFG
from mglf_lab.tasks import go2_handstand_mdp
import mglf_lab.tasks.robotlab_velocity.mdp as mdp
from mglf_lab.tasks.go2_handstand_env_cfg import (
    Go2HandstandBackFlatEnvCfg,
    Go2HandstandFrontFlatEnvCfg,
    Go2HandstandLeftFlatEnvCfg,
    Go2HandstandRightFlatEnvCfg,
)
from mglf_lab.tasks.robotlab_velocity.velocity_env_cfg import ActionsCfg


LEG_JOINT_NAMES = [
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
FRONT_WHEEL_JOINT_NAMES = ["FR_foot_joint", "FL_foot_joint"]
BACK_HANDSTAND_JOINT_VEL_NAMES = LEG_JOINT_NAMES + FRONT_WHEEL_JOINT_NAMES


@configclass
class Go2WBackHandstandActionsCfg(ActionsCfg):
    """Leg position actions plus front-wheel velocity actions for safer balance."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale={".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25},
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )
    front_wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=FRONT_WHEEL_JOINT_NAMES,
        scale=2.7,
        use_default_offset=True,
        preserve_order=True,
    )


_GO2W_INIT_JOINT_POS = {
    ".*L_hip_joint": 0.0,
    ".*R_hip_joint": -0.0,
    "F.*_thigh_joint": 0.8,
    "R.*_thigh_joint": 0.8,
    ".*_calf_joint": -1.5,
    ".*_foot_joint": 0.0,
}


class _Go2WHandstandMixin:
    """Swap the Go2 handstand task onto the Go2W articulation."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.45)
        self.scene.robot.init_state.joint_pos = _GO2W_INIT_JOINT_POS.copy()
        self.scene.robot.init_state.joint_vel = {".*": 0.0}

        if self.handstand_type == "front":
            self.rewards.air_pair_alignment_l2.weight = -1.2
            self.rewards.air_hip_deviation_l1.weight = -1.0
            self.rewards.base_ang_vel_l2.weight = -0.35
            self.rewards.joint_vel_l2.weight = -3.0e-4
            self.rewards.action_rate_l2.weight = -0.1

        if self.handstand_type == "back":
            self.rewards.handstand_feet_height_exp.weight = 20.0
            self.rewards.handstand_feet_height_exp.params["target_height"] = 0.35
            self.rewards.handstand_feet_on_air.weight = 8.0
            self.rewards.handstand_feet_air_time.weight = 3.0
            self.rewards.handstand_feet_no_contact.weight = 10.0
            self.rewards.handstand_feet_contact_count.weight = -10.0
            self.rewards.handstand_support_feet_contact.weight = 4.0
            self.rewards.handstand_orientation_l2.weight = -1.5
            self.rewards.support_pair_alignment_l2.weight = -1.0
            self.rewards.air_pair_alignment_l2.weight = -1.0
            self.rewards.air_hip_deviation_l1.weight = -1.0
            self.rewards.air_thigh_forward_limit_l2 = RewTerm(
                func=go2_handstand_mdp.joint_upper_limit_l2,
                weight=-16.0,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=["R.*_thigh_joint"], preserve_order=True
                    ),
                    "upper_limit": 0.7,
                },
            )
            self.rewards.rear_thigh_back_target_l2 = RewTerm(
                func=go2_handstand_mdp.joint_target_l2,
                weight=-3.0,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=["R.*_thigh_joint"], preserve_order=True
                    ),
                    "target": 0.25,
                },
            )
            self.rewards.rear_calf_extend_target_l2 = RewTerm(
                func=go2_handstand_mdp.joint_target_l2,
                weight=-1.0,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=["R.*_calf_joint"], preserve_order=True
                    ),
                    "target": -1.2,
                },
            )
            self.rewards.base_lin_vel_xy_l2.weight = -0.8
            self.rewards.base_ang_vel_l2.weight = -0.4
            self.rewards.joint_vel_l2.weight = -2.0e-4
            self.rewards.joint_acc_l2.weight = -5.0e-6
            self.rewards.action_rate_l2.weight = -0.16
            self.rewards.front_wheel_vel_l2 = RewTerm(
                func=mdp.joint_vel_l2,
                weight=-5.0e-4,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=FRONT_WHEEL_JOINT_NAMES, preserve_order=True
                    )
                },
            )
            self.observations.policy.joint_vel.params["asset_cfg"].joint_names = BACK_HANDSTAND_JOINT_VEL_NAMES
            self.observations.critic.joint_vel.params["asset_cfg"].joint_names = BACK_HANDSTAND_JOINT_VEL_NAMES


@configclass
class Go2WHandstandFrontFlatEnvCfg(_Go2WHandstandMixin, Go2HandstandFrontFlatEnvCfg):
    """Train Go2W with the front wheels lifted and rear wheels supporting."""


@configclass
class Go2WHandstandBackFlatEnvCfg(_Go2WHandstandMixin, Go2HandstandBackFlatEnvCfg):
    """Train Go2W with the rear wheels lifted and front wheels supporting."""

    actions: Go2WBackHandstandActionsCfg = Go2WBackHandstandActionsCfg()
    air_feet_contact_penalty_weight = -10.0


@configclass
class Go2WHandstandLeftFlatEnvCfg(_Go2WHandstandMixin, Go2HandstandLeftFlatEnvCfg):
    """Train Go2W with the left wheels lifted and right wheels supporting."""


@configclass
class Go2WHandstandRightFlatEnvCfg(_Go2WHandstandMixin, Go2HandstandRightFlatEnvCfg):
    """Train Go2W with the right wheels lifted and left wheels supporting."""
