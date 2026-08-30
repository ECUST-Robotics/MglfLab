"""RSL-RL PPO configuration for Go2W rough-terrain velocity control."""

from isaaclab.utils import configclass
from mglf_lab.tasks.rsl_rl_ppo_cfg import Go2RoughTeleopPPORunnerCfg


@configclass
class Go2WRoughTeleopPPORunnerCfg(Go2RoughTeleopPPORunnerCfg):
    max_iterations = 10000
    save_interval = 50
    experiment_name = "go2w_rough_teleop"

    # Go2W's 16-D action entropy kept raising the exploration std after the
    # locomotion policy had formed. Preserve the proven Go2 optimizer settings
    # and only reduce continued exploration for finer velocity tracking.
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.005
