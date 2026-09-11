"""UIKA rough-terrain velocity task following the MglfLab Go2 training setup."""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from mglf_lab.assets.uika import UIKA_CFG
from mglf_lab.tasks.go2_rough_env_cfg import Go2RoughTeleopEnvCfg


@configclass
class UIKARoughTeleopEnvCfg(Go2RoughTeleopEnvCfg):
    """Train UIKA to track direct planar velocity commands on rough terrain."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UIKA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        self.actions.joint_pos.joint_names = [
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ]
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}

        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.hip_joint_deviation_l1.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")
        }
