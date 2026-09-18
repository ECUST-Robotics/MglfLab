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
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
        )
        self.rewards.wheel_acc_l2 = RewTerm(
            func=mdp.joint_acc_l2,
            weight=-2.5e-9,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True)},
        )

        # Keep Go2's gait-oriented reward structure for Go2W: the wheels are an
        # assistive end effector, not a reason to train the robot as a car.
        # MglfLab samples commands up to 1.5 rather than RobotLab's typical 1.0,
        # so widen the exponential kernels by the same factor. Hard high-speed
        # samples then retain a useful gradient instead of immediately falling
        # into the near-zero reward region.
        # 奖励机身 xy 速度跟踪遥控指令。权重越大，机器人越优先追 vx/vy。
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        # 指数奖励宽度。保持 Go2 的 0.5，让速度跟踪要求更明确。
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        # 奖励偏航角速度 wz 跟踪遥控指令。
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        # 偏航速度跟踪的指数奖励宽度，同样保持 Go2 的 0.5。
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5
        # 惩罚腿部关节接近或超过限位，防止为了过障把腿打到极限姿态。
        self.rewards.dof_pos_limits.weight = 0.0
        self.rewards.dof_pos_limits.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
        )
        # Disabled while aligning Go2W with Go2's dog-like gait rewards:
        # - stand_still: penalizes leg deviation while commanded to stop.
        # - joint_pos_penalty: penalizes leg deviation throughout motion; its
        #   stand_still_scale made the penalty much stronger near zero command.
        # - joint_mirror: encourages diagonal leg pose symmetry.
        # - joint_power: penalizes leg mechanical power.
        # These are useful posture-shaping terms, but they can make Go2W more
        # conservative and less Go2-like. Re-enable one at a time if play shows
        # crouching, asymmetric gaits, or excessive leg thrashing.
        # self.rewards.stand_still = RewTerm(
        #     func=go2w_mdp.stand_still_joint_deviation_l1,
        #     weight=-2.0,
        #     params={
        #         "command_name": "base_velocity",
        #         "command_threshold": 0.1,
        #         "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
        #     },
        # )
        # self.rewards.joint_pos_penalty = RewTerm(
        #     func=go2w_mdp.command_conditioned_joint_deviation_l2,
        #     weight=-0.5,
        #     params={
        #         "command_name": "base_velocity",
        #         "stand_still_scale": 10.0,
        #         "velocity_threshold": 0.5,
        #         "command_threshold": 0.1,
        #         "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
        #     },
        # )
        # self.rewards.joint_mirror = RewTerm(
        #     func=go2w_mdp.diagonal_leg_mirror_l2,
        #     weight=-0.05,
        #     params={"asset_cfg": SceneEntityCfg("robot")},
        # )
        # self.rewards.joint_power = RewTerm(
        #     func=go2w_mdp.joint_power_l1,
        #     weight=-2.0e-5,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
        # )
        # 惩罚非轮子部件碰地。轮子是预期接触点，机身、大腿、小腿等持续擦地会被扣分。
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
        # 惩罚过大的轮地接触力，减少猛烈砸地、卡轮或用轮子硬怼障碍的行为。
        self.rewards.wheel_contact_forces = RewTerm(
            func=mdp.contact_forces,
            weight=-1.5e-4,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "threshold": 100.0,
            },
        )
        # 零速度指令时奖励轮子保持接触，帮助站立时四轮稳稳支撑，而不是频繁抬轮。
        self.rewards.wheel_contact_without_cmd = RewTerm(
            func=go2w_mdp.wheel_contact_without_command,
            weight=0.1,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )
        # Disabled while aligning with Go2. Go2 already keeps the base upright
        # through orientation penalties and base-contact termination.
        # self.rewards.upward = RewTerm(func=go2w_mdp.upward_orientation, weight=1.0)

        # Keep the inherited feet_air_time reward. Even though the feet are
        # wheels, this biases Go2W toward dog-like stepping with wheel assist
        # instead of a pure four-wheel vehicle gait.

        # RobotLab disables illegal-contact termination for both Go2 and Go2W;
        # contact penalties plus the upward reward teach recovery instead.
        self.terminations.base_contact = None
