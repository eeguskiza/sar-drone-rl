# Copyright (c) 2024 Erik Eguskiza - SAR Drone RL Project

import gymnasium as gym

from . import agents
from .quadcopter_patrol_env import QuadcopterPatrolEnv, QuadcopterPatrolEnvCfg

gym.register(
    id="Isaac-Quadcopter-Patrol-v0",
    entry_point="isaaclab_tasks.direct.quadcopter_patrol:QuadcopterPatrolEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterPatrolEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPatrolPPORunnerCfg",
    },
)