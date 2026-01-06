# Copyright (c) 2026 Alex Jauregui & Erik Eguskiza.
# PPO Configuration for Quadcopter Obstacles v5 - Directional Observations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QuadcopterObstaclesPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for obstacles with directional observations."""
    
    num_steps_per_env = 32
    max_iterations = 1000  # Más iteraciones para que converja bien
    save_interval = 300
    experiment_name = "quadcopter_obstacles_v5"
    empirical_normalization = True
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.001,  # Moderado
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )