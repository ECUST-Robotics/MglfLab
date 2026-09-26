"""RSL-RL configuration for Go2W flat-ground handstand training."""

from isaaclab.utils import configclass

from mglf_lab.tasks.go2_handstand_rsl_rl_ppo_cfg import (
    Go2HandstandBackFlatPPORunnerCfg,
    Go2HandstandFrontFlatPPORunnerCfg,
    Go2HandstandLeftFlatPPORunnerCfg,
    Go2HandstandRightFlatPPORunnerCfg,
)


@configclass
class Go2WHandstandFrontFlatPPORunnerCfg(Go2HandstandFrontFlatPPORunnerCfg):
    experiment_name = "go2w_handstand_front_flat"


@configclass
class Go2WHandstandBackFlatPPORunnerCfg(Go2HandstandBackFlatPPORunnerCfg):
    experiment_name = "go2w_handstand_back_flat"


@configclass
class Go2WHandstandLeftFlatPPORunnerCfg(Go2HandstandLeftFlatPPORunnerCfg):
    experiment_name = "go2w_handstand_left_flat"


@configclass
class Go2WHandstandRightFlatPPORunnerCfg(Go2HandstandRightFlatPPORunnerCfg):
    experiment_name = "go2w_handstand_right_flat"
