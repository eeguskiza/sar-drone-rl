# Copyright (c) 2026 Alex Jauregui & Erik Eguskiza.
# PPO Configuration for Patrol - Deterministic Forest
# Configuración conservadora para evitar divergencia

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QuadcopterPatrolPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config - Conservative settings for stable training."""
    
    # === Training ===
    num_steps_per_env = 32  # Más steps por rollout para mejor estimación de ventaja
    max_iterations = 3000   # Suficiente para bosque determinista
    save_interval = 300
    experiment_name = "quadcopter_patrol_deterministic"
    empirical_normalization = True
    
    # === Policy Network ===
    # Red más pequeña - bosque fijo no necesita tanta capacidad
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,  # Más bajo que antes (era 0.8)
        actor_hidden_dims=[256, 128, 64],  # Más pequeña
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    
    # === PPO Algorithm ===
    algorithm = RslRlPpoAlgorithmCfg(
        # Value function
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        
        # Policy clipping
        clip_param=0.2,
        
        # Entropy - un poco más alto para exploración inicial
        entropy_coef=0.01,
        
        # SGD epochs y mini-batches
        num_learning_epochs=5,
        num_mini_batches=4,
        
        # Learning rate - conservador
        learning_rate=2.0e-4,  # Más bajo que antes (era 3e-4)
        schedule="adaptive",
        
        # GAE
        gamma=0.99,
        lam=0.95,
        
        # KL target - más estricto para updates más pequeños
        desired_kl=0.008,  # Más bajo (era 0.01)
        
        # Gradient clipping
        max_grad_norm=0.5,  # Más agresivo (era 1.0)
    )


@configclass
class QuadcopterPatrolPPORunnerCfgAggressive(RslRlOnPolicyRunnerCfg):
    """PPO config - More aggressive if conservative is too slow."""
    
    num_steps_per_env = 24
    max_iterations = 2500
    save_interval = 250
    experiment_name = "quadcopter_patrol_deterministic_v2"
    empirical_normalization = True
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )