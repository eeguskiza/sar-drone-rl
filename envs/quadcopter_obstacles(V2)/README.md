# 🚁 Phase 2: Obstacle Avoidance & Waypoint Navigation (Refined)

Building upon the flight dynamics mastered in Phase 1, this environment introduces **environmental perception**. The drone must navigate through a procedurally generated forest of rectangular pillars to reach a sequence of waypoints.

After an initial training phase, a **Safety Fine-Tuning** stage was applied to transition the agent from an aggressive navigator to a safety-conscious pilot.

<p align="center">
<img src="../../docs/figures/obstacles.gif" alt="Obstacle Avoidance Demo" width="600">
<em>Policy navigating through random obstacles. Note the deceleration maneuvers when approaching pillars.</em>
</p>

## 🎯 Task Objective

The agent must fly through a sequence of **5 sequential waypoints** without colliding.

* **Navigation:** Reach within 0.8m of the active waypoint to trigger the next one.
* **Perception:** Detect and evade 50 randomly placed pillars.
* **Safety:** Minimize high-speed approaches towards obstacles (Collision Risk).

## 📡 Observation Space (Directional Perception)

The total observation size is **98 dimensions**. Crucially, perception is handled via **Body-Frame Sensing**. Instead of a heavy Lidar point cloud, we feed the network the **5 closest obstacles** transformed into the Drone's local coordinate system.

| Index | Name | Dims | Description |
| --- | --- | --- | --- |
| `0-11` | **Base Flight State** | 12 | Lin Vel, Ang Vel, Gravity, etc. |
| `12-14` | **Target Vector** | 3 | Vector to the *current active* waypoint (Body Frame). |
| `15-29` | **Obstacle Directions** | 15 | Unit vectors pointing to the 5 closest obstacles. |
| `30-34` | **Obstacle Distances** | 5 | Normalized distance to the 5 closest obstacles. |
| `35` | **Mission Progress** | 1 | Percentage of waypoints completed. |

## 🧠 Fine-Tuning & Reward Shaping (The "Safety" Update)

Initial training yielded a ~76% success rate, but the drone exhibited reckless behavior (near-misses). To solve this, we applied a **Fine-Tuning stage** (starting from checkpoint 3000) with two critical changes:

### 1. Velocity Vector Penalization (Reward Shaping)
We replaced the simple proximity penalty with a **Collision Risk** calculation. The agent is punished not just for *being* close to an obstacle, but for *moving towards it*.

$$R_{risk} = - \sum (\vec{v}_{drone} \cdot \vec{d}_{obs}) \times (1 - dist_{norm})$$

* If the drone flies **parallel** to the wall → No penalty.
* If the drone flies **towards** the wall → Massive penalty.

### 2. Policy Hardening
* **Reduced Learning Rate:** 3e-4 → 5e-5 (Precision updates).
* **Reduced Entropy:** Forced the model to exploit learned safe paths rather than exploring dangerous ones.
* **Stricter Physics:** Collision threshold reduced to 2cm, forcing the "virtual skin" of the drone to be untouched.

### 📊 Results

The fine-tuning process drastically changed the flight characteristics:

* **Success Rate:** Improved from **76%** to **>91%**.
* **Collision Risk:** Reduced by ~50% (see graph below).
* **Behavior:** The drone now executes active braking maneuvers when an obstacle enters its flight path.

<p align="center">
<img src="../../docs/figures/colision_risk.png" alt="Collision Risk Reduction" width="800">

<em>TensorBoard log showing the sharp drop (improvement) in Collision Risk immediately after applying the new reward function at step 3000.</em>
</p>

## ⚙️ Training Configuration

* **Hardware:** NVIDIA RTX 5070 Ti (16GB VRAM) + Intel Core Ultra 9.
* **Concurrency:** 16,384 parallel environments.
* **Total Timesteps:** ~5.2 Billion (Base) + ~1.5 Billion (Fine-tuning).

```bash
# Fine-tuning command used
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Quadcopter-Obstacles-v0 \
    --num_envs 16384 \
    --load_run=2025-12-31_13-27-20 \
    --checkpoint=model_2999.pt \
    --experiment_name=quadcopter_obstacles_v5_finetune