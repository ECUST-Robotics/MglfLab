"""Mglf rough-terrain velocity task with the Piper arm locked."""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from mglf_lab.assets.mglf import GO2_LEG_JOINT_NAMES, MGLF_CFG
from mglf_lab.tasks.go2_rough_env_cfg import Go2RoughTeleopEnvCfg


@configclass
class MglfRoughTeleopEnvCfg(Go2RoughTeleopEnvCfg):
    """Train the Go2 base while carrying a locked Piper manipulator."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = MGLF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        leg_joint_cfg = SceneEntityCfg("robot", joint_names=GO2_LEG_JOINT_NAMES, preserve_order=True)
        self.actions.joint_pos.joint_names = GO2_LEG_JOINT_NAMES
        self.observations.policy.joint_pos.params["asset_cfg"] = leg_joint_cfg
        self.observations.policy.joint_vel.params["asset_cfg"] = leg_joint_cfg

        self.events.reset_robot_joints.params["asset_cfg"] = leg_joint_cfg
        self.rewards.dof_torques_l2.params["asset_cfg"] = leg_joint_cfg
        self.rewards.dof_acc_l2.params["asset_cfg"] = leg_joint_cfg
        self.rewards.dof_pos_limits.params["asset_cfg"] = leg_joint_cfg
