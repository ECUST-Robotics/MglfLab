"""Go2 rough-terrain velocity task with exteroceptive height scans.

The terrain source and scanner layout match RobotLab v2.3.2's rough velocity
base task: Isaac Lab ``ROUGH_TERRAINS_CFG`` and a yaw-aligned 1.6 x 1.0 m grid
with 0.1 m resolution. There is deliberately no waypoint or navigation manager.
"""

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)
from mglf_lab.assets.go2 import UNITREE_GO2_URDF_CFG


@configclass
class Go2RoughTeleopEnvCfg(UnitreeGo2RoughEnvCfg):
    """Train a Go2 to track direct planar velocity commands on rough terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_URDF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Direct joystick-style commands: forward/back, lateral, and yaw rate.
        # No target heading and no waypoint is involved.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.10
        self.commands.base_velocity.resampling_time_range = (5.0, 8.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = None

        # RobotLab v2.3.2 uses this exact scanner geometry. The inherited policy
        # observation already includes ``height_scan`` with noise and clipping.
        self.scene.height_scanner.pattern_cfg.resolution = 0.1
        self.scene.height_scanner.pattern_cfg.size = (1.6, 1.0)
        self.scene.height_scanner.ray_alignment = "yaw"
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation

        # Preserve rough-terrain curriculum and add moderate pushes for robust
        # recovery. Go2's upstream task disables this event by default.
        self.scene.terrain.terrain_generator.curriculum = True
        # Match the terrain used to train model_5400.pt.
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.05, 0.23)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.05, 0.23)

        # Discourage the hip ab/adduction joints from folding the feet inward.
        # This is deliberately moderate so lateral motion and foothold
        # adaptation on rough terrain remain available to the policy.
        self.rewards.hip_joint_deviation_l1 = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
        )

        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )
