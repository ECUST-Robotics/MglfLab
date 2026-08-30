"""Height-map rough-terrain velocity task for Unitree Go2W."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ActionsCfg

from mglf_lab.assets import UNITREE_GO2W_CFG
from mglf_lab.tasks import go2w_mdp
from mglf_lab.tasks.go2_rough_env_cfg import Go2RoughTeleopEnvCfg


LEG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
WHEEL_JOINT_NAMES = ["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"]
ALL_JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES


@configclass
class Go2WActionsCfg(ActionsCfg):
    """Twelve leg-position actions followed by four wheel-velocity actions."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale={".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25},
        use_default_offset=True,
        preserve_order=True,
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=5.0,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class Go2WRoughTeleopEnvCfg(Go2RoughTeleopEnvCfg):
    """Train Go2W to combine stepping and rolling on scanned rough terrain."""

    actions: Go2WActionsCfg = Go2WActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # Keep the stair-focused 60% sampling distribution while restoring
        # the original full stair-height range. The remaining 40% retains the
        # already learned rough-ground skill.
        sub_terrains = self.scene.terrain.terrain_generator.sub_terrains
        sub_terrains["pyramid_stairs"].step_height_range = (0.08, 0.28)
        sub_terrains["pyramid_stairs_inv"].step_height_range = (0.08, 0.28)
        sub_terrains["pyramid_stairs"].proportion = 0.30
        sub_terrains["pyramid_stairs_inv"].proportion = 0.30
        sub_terrains["boxes"].proportion = 0.15
        sub_terrains["random_rough"].proportion = 0.15
        sub_terrains["hf_pyramid_slope"].proportion = 0.05
        sub_terrains["hf_pyramid_slope_inv"].proportion = 0.05

        # The Go2 parent applies one common scale; restore Go2W's separate leg
        # and wheel semantics after the parent configuration is complete.
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_vel.scale = 5.0

        # Wheel angles are continuous and unbounded, so they must not enter the
        # relative-position observation. Wheel velocities remain observable.
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=ALL_JOINT_NAMES, preserve_order=True
        )
        # Keep the proven MglfLab Go2 observation scaling. Only append wheel
        # velocities and sanitize the extra height-map input.
        self.observations.policy.height_scan.func = go2w_mdp.finite_height_scan

        # Do not penalize high wheel speed/acceleration with leg-scale weights.
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
        )
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
        )
        self.rewards.wheel_acc_l2 = RewTerm(
            func=mdp.joint_acc_l2,
            weight=-2.5e-9,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True)},
        )

        # RobotLab uses the same task rewards for Go2 and Go2W, then removes the
        # foot-gait terms for wheels. MglfLab samples commands up to 1.5 rather
        # than RobotLab's typical 1.0, so widen the exponential kernels by the
        # same factor. Hard high-speed samples then retain a useful gradient
        # instead of immediately falling into the near-zero reward region.
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.75
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.75
        self.rewards.dof_pos_limits.weight = -5.0
        self.rewards.dof_pos_limits.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
        )
        self.rewards.stand_still = RewTerm(
            func=go2w_mdp.stand_still_joint_deviation_l1,
            weight=-2.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
            },
        )
        self.rewards.joint_pos_penalty = RewTerm(
            func=go2w_mdp.command_conditioned_joint_deviation_l2,
            weight=-0.5,
            params={
                "command_name": "base_velocity",
                # Keep the same effective zero-command posture strength
                # (-0.5 * 10 == -1.0 * 5) while freeing the legs to lift and
                # extend during commanded stair traversal.
                "stand_still_scale": 10.0,
                "velocity_threshold": 0.5,
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
            },
        )
        self.rewards.joint_mirror = RewTerm(
            func=go2w_mdp.diagonal_leg_mirror_l2,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.joint_power = RewTerm(
            func=go2w_mdp.joint_power_l1,
            weight=-2.0e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
        )
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                # Wheel links are the only intended terrain contacts. Calves,
                # thighs, motor housings and the body may touch briefly while
                # crossing obstacles, but sustained crawling is penalized.
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*_foot$).*"),
                "threshold": 1.0,
            },
        )
        self.rewards.wheel_contact_forces = RewTerm(
            func=mdp.contact_forces,
            weight=-1.5e-4,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "threshold": 100.0,
            },
        )
        self.rewards.wheel_contact_without_cmd = RewTerm(
            func=go2w_mdp.wheel_contact_without_command,
            weight=0.1,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )
        self.rewards.upward = RewTerm(func=go2w_mdp.upward_orientation, weight=1.0)

        # A feet-air-time gait reward is correct for Go2 feet but not for Go2W:
        # during efficient rolling all four wheels are expected to remain in
        # contact. Rough terrain and the height scan still teach leg lifting.
        self.rewards.feet_air_time = None

        # RobotLab disables illegal-contact termination for both Go2 and Go2W;
        # contact penalties plus the upward reward teach recovery instead.
        self.terminations.base_contact = None
