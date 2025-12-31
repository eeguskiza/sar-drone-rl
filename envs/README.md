# 🤖 RL Environments: Search & Rescue Drone

This directory contains the Reinforcement Learning (RL) environments designed for the SAR Drone project. The environments are built on **Isaac Lab** and follow a progressive difficulty curriculum, ranging from basic stabilization to complex person localization missions.

The ultimate goal is to train a policy capable of running on real hardware (Crazyflie 2.1) via **Sim-to-Real** transfer and having autonomous drones for rescue purposes.

---

## 🚁 The Platform: Crazyflie 2.1

<p align="center">
<img src="../docs/figures/crazyflie_drone.png" alt="Crazyflie in Isaac Sim" width="600">





<em>Simulated model of the Crazyflie 2.x in Isaac Sim</em>
</p>

The **Crazyflie 2.x** is a nano-quadcopter developed by [Bitcraze](https://www.bitcraze.io/). It is the standard platform in robotics research due to its small size, open-source firmware, and agile dynamics. In this project, we simulate its rigid body dynamics and control the motors directly, replacing classic PID controllers with Neural Networks.

### Technical Specifications

| Property | Value | Notes |
| --- | --- | --- |
| **Weight** | 27 g | Critical for inertia modeling |
| **Size** | 92 mm | Motor-to-motor (diagonal) |
| **Max Thrust** | ~60 g | Thrust-to-weight ratio  |
| **Motors** | 4 × DC Coreless | Modeled as force actuators |
| **Simulation** | 100 Hz | Decimation=2 (Control frequency at 50 Hz) |

---

## 📂 Project Structure (Curriculum Learning)

The project is divided into three incremental phases, each represented by a specific environment folder:

### 1️⃣ Phase 1: Stabilization (`/quadcopter`)

**Goal:** Learn to fly.

* The agent must master the drone's dynamics.
* **Task:** Hover and reach target coordinates  in empty space.
* **Challenge:** Overcoming inherent instability without PIDs.

### 2️⃣ Phase 2: Navigation (`/quadcopter_obstacles`)

**Goal:** Local perception and avoidance.

* Introduces procedurally generated cylindrical obstacles (pillars).
* **New Input:** The drone receives distance information (simulating LiDAR or Depth Map).
* **Challenge:** Balancing velocity towards the goal with safety (collision avoidance).

### 3️⃣ Phase 3: SAR Mission (`/quadcopter_localization`)

**Goal:** Search & Rescue.

* A full scenario with hidden "persons" and complex obstacle layouts.
* **Logic:** Implements a *coverage grid* (memory of visited locations).
* **Reward:** Exploration (visiting new cells) + Detection (proximity to target).

---

## ⚙️ Technical Environment Details

### 🔹 Action Space (Common)

The drone is controlled via 4 continuous values normalized to .

| Index | Action | Description | Physically |
| --- | --- | --- | --- |
| **0** | `Thrust` | Total vertical thrust |  |
| **1** | `Roll` | Moment around X-axis |  |
| **2** | `Pitch` | Moment around Y-axis |  |
| **3** | `Yaw` | Moment around Z-axis |  |

### 🔹 Observation Space (Evolving)

<details>
<summary><strong>👁️ View Observation Details (Click to expand)</strong></summary>

#### Base (12 Dimensions)

Used in all phases.

1. **Linear Velocity (3):** `root_lin_vel_b` (m/s)
2. **Angular Velocity (3):** `root_ang_vel_b` (rad/s)
3. **Projected Gravity (3):** Gravity vector in body frame (indicates tilt/uprightness).
4. **Relative Position (3):** Vector towards the target .

#### Obstacles (+ N Dimensions)

Adds environmental perception.

* **Raycasts/Distances:** Normalized distance to the nearest  obstacles.

#### Localization (Grid + Sensor)

Adds memory and mission logic.

* **Coverage Grid:** Flattened map of visited cells (Boolean/Float).
* **Person Sensor:** Binary flag (detected/not) and direction vector if in range.

</details>