"""RSL-RL configuration for Go2 flat-ground handstand training."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2HandstandFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "go2_handstand_flat"
    empirical_normalization = False
    clip_actions = 10.0
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2HandstandFrontFlatPPORunnerCfg(Go2HandstandFlatPPORunnerCfg):
    experiment_name = "go2_handstand_front_flat"


@configclass
class Go2HandstandBackFlatPPORunnerCfg(Go2HandstandFlatPPORunnerCfg):
    experiment_name = "go2_handstand_back_flat"


@configclass
class Go2HandstandLeftFlatPPORunnerCfg(Go2HandstandFlatPPORunnerCfg):
    experiment_name = "go2_handstand_left_flat"


@configclass
class Go2HandstandRightFlatPPORunnerCfg(Go2HandstandFlatPPORunnerCfg):
    experiment_name = "go2_handstand_right_flat"
