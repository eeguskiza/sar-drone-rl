# 🚧 Phase 2: Obstacle Avoidance & Waypoint Navigation
Building upon the flight dynamics mastered in Phase 1, this environment introduces **environmental perception**. The drone must now navigate through a procedurally generated forest of rectangular pillars to reach a sequence of waypoints.

<p align="center">
<img src="../../docs/figures/obstacles.gif" alt="Obstacle Avoidance Demo" width="600">





<em>Policy trained with PPO navigating through random obstacles while tracking waypoints.</em>
</p>

## 🎯 Task Objective

The agent must fly through a sequence of **5 sequential waypoints**  without colliding.

* **Navigation:** Reach within 0.8m of the active waypoint to trigger the next one.
* **Perception:** Detect and evade 50 randomly placed pillars (Static Obstacles).
* **Constraint:** The obstacles are generated procedurally every reset, preventing map memorization.


## 🧠 Observation Space (Directional Perception)

The total observation size remains **98 dimensions** to maintain architectural compatibility with Phase 3. However, the active inputs are significantly more complex here.

**Key Technical Detail: Body-Frame Sensing**
Instead of a heavy Lidar point cloud, we feed the network the **5 closest obstacles**. Crucially, these positions are transformed into the **Drone's Body Frame**. This makes the policy rotation-invariant (if I turn left, the obstacle moves right relative to me).

| Index | Name | Dims | Description |
| --- | --- | --- | --- |
| `0-11` | **Base Flight State** | 12 | Lin Vel, Ang Vel, Gravity, etc. (Same as Phase 1). |
| `12-14` | **Target Vector** | 3 | Vector to the *current active* waypoint (Body Frame). |
| `15-29` | **Obstacle Directions** | 15 | Unit vectors pointing to the 5 closest obstacles . |
| `30-34` | **Obstacle Distances** | 5 | Normalized distance to the 5 closest obstacles. |
| `35` | **Mission Progress** | 1 | Percentage of waypoints completed (). |
| `36-97` | **PADDING** | 62 | Zero-filled. Reserved for Grid sensors in Phase 3. |



## 📉 Reward Function: The Safety-Speed Trade-off

The reward function is designed to balance the "Greedy" desire to reach the goal with the "Fear" of collision.

1. **Waypoint Navigation:**


* Includes a dense signal for approaching and a sparse bonus (+50.0) for reaching a waypoint.


2. **Obstacle Repulsion (Safety Bubble):**
We use an exponential penalty that activates sharply only when getting too close (< 1.0m).





## 🚀 Training Configuration

Due to the increased computational cost of calculating distances to 50 obstacles for every agent, we adjusted the parallelism slightly compared to the empty environment.

* **Hardware:** NVIDIA RTX 5070 Ti (16GB VRAM) + Intel Core Ultra 9.
* **Concurrency:** **16,384 (16k) parallel environments**.
* **Algorithm:** PPO with Adaptive Learning Rate.
* **Network:** `[256, 256, 128]` (ELU).

```bash
# Train the obstacle avoidance policy
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Quadcopter-Obstacles-v0 \
    --num_envs 16384 \
    --headless
```

> **Note:** The reduction from 64k to 16k envs ensures stable VRAM usage while still generating millions of samples per minute, sufficient for the agent to learn complex avoidance maneuvers in under 20 minutes.