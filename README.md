# 🚁 SAR Drone RL

**Autonomous Drone Navigation using Deep Reinforcement Learning in NVIDIA Isaac Sim**

*A Comparative Study of PPO with Curriculum Learning*

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Isaac Sim 5.1](https://img.shields.io/badge/Isaac%20Sim-5.1-green.svg)](https://developer.nvidia.com/isaac-sim)
[![PyTorch](https://img.shields.io/badge/PyTorch-Nightly-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

This project implements and compares deep reinforcement learning algorithms for autonomous quadcopter navigation using NVIDIA Isaac Sim and Isaac Lab. The Crazyflie 2.x drone learns to navigate to target positions, with progressive difficulty through curriculum learning.

<p align="center">
<img src="docs/figures/sar_mission.gif" alt="Active SAR Mission Demo" width="600">

<em>Drone executing the search pattern. Note the markers changing from <strong>Red (Undetected)</strong> to <strong>Green (Rescued)</strong> upon flying within the 1.5m detection radius.</em>
</p>

**Key Features:**
- High-performance parallel simulation (8,192+ environments)
- Curriculum learning with obstacle avoidance
- RTX 5070 Ti (Blackwell) optimized

---


