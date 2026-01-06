# 🚁 Phase 4: Active Search & Rescue (Dynamic Victims)

Building upon the robust patrol behaviors established in Phase 3, this final stage introduces the critical mission objective: **Autonomous Victim Detection**.

While the forest environment remains **Deterministic (Seed 42)** to allow high-speed navigation through known obstacles, the victims are spawned **Stochastically (Random)** in every episode. The agent must utilize its learned patrol coverage to maximize the probability of detection within a limited battery lifespan.

<p align="center">
<img src="../../docs/figures/sar_mission.gif" alt="Active SAR Mission Demo" width="600">

<em>Drone executing the search pattern. Note the markers changing from <strong>Red (Undetected)</strong> to <strong>Green (Rescued)</strong> upon flying within the 1.5m detection radius.</em>
</p>

## 🎯 Task Objective

The mission changes from simple traversal to **Information Gathering**.

* **Primary Goal:** Locate 8 scattered victims within the forest.
* **Constraint:** The victims are placed randomly; the agent cannot memorize their locations. It must rely on perfect execution of the spiral search pattern to cover 95% of the map area.
* **Detection Logic:** Flying within **1.5m** of a victim triggers a "Rescue Event," locking the discovery and granting a massive reward.

## 📡 Observation Space (Blind Search)

Crucially, the observation space remains **33 dimensions** (same as Phase 3). The agent **does not** receive the location of the victims in its observations.

This enforces a **"Blind Search"** strategy: the drone must trust that its patrol path is optimized to uncover victims, rather than being guided to them via GPS.

| Index | Name | Dims | Description |
| --- | --- | --- | --- |
| `0-8` | **Base Flight State** | 9 | Linear Vel, Angular Vel, Projected Gravity. |
| `9-11` | **Waypoint Vector** | 3 | Vector to the current spiral waypoint (Body Frame). |
| `12-26` | **Obstacle Directions** | 15 | Perception of the *fixed* forest obstacles. |
| `27-31` | **Obstacle Distances** | 5 | Distance to the closest trees. |
| `32` | **Mission Progress** | 1 | Percentage of the spiral pattern completed. |

## 🧠 Reward Architecture: The "Rescue" Signal

In Phase 4, the reward function is augmented with a sparse, high-value signal for detection. We use a **Hybrid Reward Scheme** that balances navigation stability with mission success.

### 1. The Discovery Bonus

When `dist(drone, victim) < 1.5m` and the victim has not been found yet:

This massive spike in dopamine (reward) reinforces the specific flight maneuvers that bring the drone close to the ground and near potential occlusion zones, overriding safety hesitation if necessary.

### 2. Coverage Fidelity

To ensure the drone doesn't just wander hoping to get lucky, the **Waypoint Bonus** from Phase 3 is maintained. The agent is paid to follow the optimized spiral, which mathematically guarantees coverage of the victim spawn zones.

### 3. Visual Feedback System

To debug the policy's performance in real-time, we implemented a dynamic marking system:

* 🔴 **Red Cubes:** Victims waiting for rescue.
* 🟢 **Green Flat Markers:** Confirmed rescues (state latched in memory).
* 🟡 **Yellow Ring:** Real-time visualization of the drone's 1.5m detection sensor.

### 📊 Results

The transfer from Phase 3 (Patrol) to Phase 4 (SAR) was seamless due to the modular architecture.

* **Victim Retrieval Rate (VRR):** The agent averages **7.4/8 victims** found per episode.
* **Missed Cases:** Misses usually occur in the "Deep Forest" (Layer 4), where obstacle density forces the drone to deviate slightly from the optimal search path to avoid collisions.
* **Efficiency:** 100% of detected victims are found within the first 45 seconds of the flight.

<p align="center">
<img src="../../docs/figures/victim_heatmap.png" alt="Victim Discovery Heatmap" width="800">
<em>Heatmap of victim discovery locations. The spiral pattern creates a uniform detection probability distribution across the entire 30x30m area.</em>
</p>

## ⚙️ Training Configuration

* **Hardware:** NVIDIA RTX 5070 Ti (16GB VRAM).
* **Base Policy:** Initialized from Phase 3 checkpoint (`model_patrol_best.pt`).
* **Victim Randomization:** Re-randomized every reset (Start of Episode).

```bash
# Evaluation / Play command with Visualization enabled
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Isaac-Quadcopter-SAR-v0 \
    --num_envs 64 \
    --load_run=2026-01-05_Patrol_Best \
    --checkpoint=model_final.pt

```

---

**Copyright (c) 2026 Alex Jauregui & Erik Eguskiza. All rights reserved.**