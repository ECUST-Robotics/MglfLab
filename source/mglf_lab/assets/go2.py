"""Go2 imported from local URDF, retaining Isaac Lab's Go2 training settings."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


_URDF = Path(__file__).resolve().parents[1] / "data/Robots/unitree/go2_description/urdf/go2_description.urdf"

# Copy the complete upstream configuration: explicit DC motors, initial pose,
# soft limits, contact sensors and rigid-body/solver overrides stay identical.
UNITREE_GO2_URDF_CFG = UNITREE_GO2_CFG.copy()
UNITREE_GO2_URDF_CFG.spawn = sim_utils.UrdfFileCfg(
    asset_path=str(_URDF),
    fix_base=False,
    merge_fixed_joints=True,
    replace_cylinders_with_capsules=False,
    activate_contact_sensors=UNITREE_GO2_CFG.spawn.activate_contact_sensors,
    rigid_props=UNITREE_GO2_CFG.spawn.rigid_props.copy(),
    articulation_props=UNITREE_GO2_CFG.spawn.articulation_props.copy(),
    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
    ),
)
