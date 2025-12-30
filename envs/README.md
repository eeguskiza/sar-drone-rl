# 🎮 Environments

This directory contains the reinforcement learning environments used in this project.

---

## 🚁 The Robot: Crazyflie 2.x

<p align="center">
  <img src="../docs/figures/crazyflie_drone.png" alt="Crazyflie in Isaac Sim" width="600">
</p>

The **Crazyflie 2.x** is a nano quadcopter developed by [Bitcraze](https://www.bitcraze.io/). It's widely used in robotics research due to its small size, open-source firmware, and extensive documentation.

### Specifications

| Property | Value |
|----------|-------|
| Weight | 27 g |
| Size | 92 mm motor-to-motor |
| Max thrust | ~58 g (thrust-to-weight ≈ 2.1) |
| Motors | 4 × coreless DC |
| Flight time | ~7 min |

### Simulated Model

In Isaac Sim, the Crazyflie is modeled as an articulated body with:

- **4 motor joints** (`m1_joint`, `m2_joint`, `m3_joint`, `m4_joint`)
- **Rigid body dynamics** with accurate mass and inertia
- **Direct force/torque control** (not motor velocity)

The simulation uses a simplified actuation model where actions map to total thrust and body moments, abstracting away individual motor control for faster learning.

---

## Base Environment: `quadcopter_base.py`

Original Isaac Lab environment for Crazyflie quadcopter navigation.

**Source:** `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/quadcopter_env.py`

### Observation Space (12D)

| Index | Observation | Description |
|-------|-------------|-------------|
| 0-2 | `root_lin_vel_b` | Linear velocity in body frame (m/s) |
| 3-5 | `root_ang_vel_b` | Angular velocity in body frame (rad/s) |
| 6-8 | `projected_gravity_b` | Gravity vector in body frame |
| 9-11 | `desired_pos_b` | Relative target position in body frame (m) |

### Action Space (4D)

| Index | Action | Range | Description |
|-------|--------|-------|-------------|
| 0 | Thrust | [-1, 1] | Normalized total thrust |
| 1 | Roll moment | [-1, 1] | Torque around X axis |
| 2 | Pitch moment | [-1, 1] | Torque around Y axis |
| 3 | Yaw moment | [-1, 1] | Torque around Z axis |

### Reward Function

```
reward = distance_reward + velocity_penalty + angular_penalty

where:
- distance_reward = (1 - tanh(distance / 0.8)) * 15.0 * dt
- velocity_penalty = -0.05 * ||linear_vel||² * dt
- angular_penalty = -0.01 * ||angular_vel||² * dt
```

### Termination Conditions

- **Death:** z < 0.1m (crashed) or z > 2.0m (too high)
- **Timeout:** 10 seconds (500 steps at 50Hz control)

### Task Parameters

| Parameter | Value |
|-----------|-------|
| Episode length | 10 s |
| Control frequency | 50 Hz (decimation=2, sim=100Hz) |
| Goal XY range | [-2, 2] m |
| Goal Z range | [0.5, 1.5] m |
| Thrust-to-weight ratio | 1.9 |

---

## Extended Environment: `quadcopter_obstacles.py`

Custom extension with static obstacles for curriculum learning.

### Added Features

- Cylindrical obstacles (pillars)
- Lidar-based distance sensing (optional)
- Collision penalties

### New Observations

| Index | Observation | Description |
|-------|-------------|-------------|
| 12-N | `lidar_distances` | Raycast distances to obstacles |

### Modified Rewards

- Additional penalty for obstacle proximity
- Bonus for successful navigation through obstacles

---

## Usage

### Register custom environment

```python
import gymnasium as gym

gym.register(
    id="Isaac-Quadcopter-Obstacles-v0",
    entry_point="envs.quadcopter_obstacles:QuadcopterObstaclesEnv",
    kwargs={
        "env_cfg_entry_point": "envs.quadcopter_obstacles:QuadcopterObstaclesCfg",
    },
)
```

### Train with custom environment

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Quadcopter-Obstacles-v0 \
    --num_envs 4096
```