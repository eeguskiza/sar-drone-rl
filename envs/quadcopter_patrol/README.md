# 🚁 Phase 3: Long-Horizon Patrol (Deterministic Forest)

Building upon the obstacle avoidance capabilities of Phase 2, this environment shifts the focus to **consistency and endurance**. The drone is tasked with executing a complex, 24-waypoint spiral patrol pattern within a fixed, dense forest environment.

Unlike previous phases focused on generalization, this stage uses a **Deterministic Environment (Seed 42)** to validate the agent's ability to memorize complex spatial trajectories and optimize a specific patrol route, serving as the foundational movement layer for the upcoming Person Detection task (Stage 4).

<p align="center">
<img src="../../docs/figures/patrol_spiral.gif" alt="Patrol Spiral Demo" width="600">

<em>Policy executing the 4-layer spiral patrol pattern. Note the increasing difficulty as the radius expands into denser obstacle clusters.</em>
</p>

## 🎯 Task Objective

The agent must complete a **4-Layer Spiral Pattern** consisting of 24 sequential waypoints.

* **Long-Horizon Navigation:** The episode length is extended to allow for a full tour of the 30x30m area.
* **Precision Flight:** Waypoint tolerance is loose (1.2m) to encourage smooth, continuous flight rather than stop-and-go hovering.
* **Static Mapping:** Navigate a fixed layout of 50 obstacles, allowing the policy to overfit/optimize for this specific terrain geometry.

## 📡 Observation Space (Streamlined Perception)

The observation space has been optimized to **33 dimensions** (removing the padding used in transfer learning). We retain the efficient Body-Frame Sensing but add a specific input for mission progress to help the Value Function estimate returns over the longer horizon.

| Index | Name | Dims | Description |
| --- | --- | --- | --- |
| `0-8` | **Base Flight State** | 9 | Linear Vel, Angular Vel, Projected Gravity (Body Frame). |
| `9-11` | **Waypoint Vector** | 3 | Normalized vector to the *current active* waypoint. |
| `12-26` | **Obstacle Directions** | 15 | Unit vectors pointing to the 5 closest obstacles. |
| `27-31` | **Obstacle Distances** | 5 | Normalized distance to the 5 closest obstacles. |
| `32` | **Patrol Progress** | 1 | Percentage of waypoints completed (). |

## 🧠 Reward Architecture & Network Optimization

Training a long-horizon task with standard PPO often leads to "lazy" agents that stay in safe zones. To counter this, we implemented a **Layered Reward Structure** and optimized the network architecture for a static environment.

### 1. Layered Curriculum Rewards

Instead of a flat bonus, the reward scales dynamically as the drone pushes into outer layers (further from the safe center). This creates a gradient of value that pulls the agent outwards.

### 2. Network Compression (Efficiency)

Since the environment is deterministic (the forest layout never changes), we reduced the Policy/Critic capacity to prevent overfitting to noise and speed up inference.

* **Phase 2 Arch:** `[256, 256, 128]`
* **Phase 3 Arch:** `[256, 128, 64]`

### 3. Conservative PPO Updates

To ensure stability over long episodes (where a crash at step 500 wipes out the return), we adopted a conservative update strategy:

* **Strict KL Divergence:** `0.008` (prevents policy collapse).
* **Lower Learning Rate:** `2.0e-4` (ensures monotonic improvement).

### 📊 Results

The specialized architecture allows the drone to clear the full 24-waypoint spiral consistently.

* **Completion Rate:** >95% on the fixed seed.
* **Flight Smoothness:** The agent learns to "slice" corners, maintaining momentum between waypoints rather than stopping at each one.
* **Layer Penetration:** The reward scaling successfully motivates the agent to reach Layer 4 (radius 13m) without getting stuck in local minima in the safe center.

<p align="center">
<img src="../../docs/figures/layer_progress.png" alt="Layer Completion Rates" width="800">

<em>Training metrics showing the sequential mastering of patrol layers. Layer 1 is solved first, followed by the gradual expansion to outer layers.</em>

</p>

## ⚙️ Training Configuration

* **Hardware:** NVIDIA RTX 5070 Ti (16GB VRAM) + Intel Core Ultra 9.
* **Concurrency:** 4,096 parallel environments (Lower count due to higher simulation stability requirements).
* **Total Timesteps:** ~2.0 Billion.

```bash
# Training command used
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Quadcopter-Patrol-v0 \
    --num_envs 4096 \
    --experiment_name=quadcopter_patrol_deterministic \
    --headless

```