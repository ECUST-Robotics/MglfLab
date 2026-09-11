"""UIKA articulation imported from the local ROS workspace URDF."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_UIKA_URDF = _DATA_DIR / "Robots/uika_description/urdf/uika_simple_collision.urdf"


UIKA_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(_UIKA_URDF),
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.33),
        joint_pos={
            "FL_hip_joint": -0.75,
            "FL_thigh_joint": 0.05,
            "FL_calf_joint": 0.70,
            "FR_hip_joint": 0.75,
            "FR_thigh_joint": 0.05,
            "FR_calf_joint": 0.70,
            "RL_hip_joint": -0.75,
            "RL_thigh_joint": 0.05,
            "RL_calf_joint": 0.70,
            "RR_hip_joint": 0.75,
            "RR_thigh_joint": 0.05,
            "RR_calf_joint": 0.70,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit_sim=17.0,
            velocity_limit_sim=28.8,
            stiffness=30.0,
            damping=1.5,
            friction=0.0,
            armature=0.0133752835,
        ),
        "thigh": ImplicitActuatorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit_sim=17.0,
            velocity_limit_sim=28.8,
            stiffness=30.0,
            damping=1.5,
            friction=0.0,
            armature=0.013330029,
        ),
        "calf": ImplicitActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit_sim=31.7,
            velocity_limit_sim=15.43,
            stiffness=30.0,
            damping=1.5,
            friction=0.0,
            armature=0.0231929655,
        ),
    },
)
"""UIKA quadruped with position-controlled hip, thigh, and calf joints."""
