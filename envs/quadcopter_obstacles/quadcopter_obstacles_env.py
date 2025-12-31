# Copyright (c) 2024 Erik Eguskiza - SAR Drone RL Project
# Stage 2 v5: Obstacles + Waypoints with Directional Observations

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_apply_inverse
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG

from isaaclab_assets import CRAZYFLIE_CFG


@configclass
class QuadcopterObstaclesEnvCfg(DirectRLEnvCfg):
    """Configuration for Quadcopter with Obstacles and Directional Observations."""
    
    # Episode
    episode_length_s = 45.0
    decimation = 2
    
    # Obstacles
    num_obstacles = 50
    num_closest_obstacles = 5  # Solo observamos los 5 más cercanos
    
    # Waypoints
    num_waypoints = 5
    waypoint_reach_threshold = 0.8
    
    # Observation: 12 base + 5*4 (closest obstacles: dir + dist) + 1 progress = 33
    # Padding: 98 - 33 = 65
    observation_space = 98
    action_space = 4
    state_space = 0
    debug_vis = True

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=20.0, replicate_physics=True
    )

    # Robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # Obstacle configuration
    obstacle_height = 1.5
    obstacle_radius = 0.15
    obstacle_spawn_range = 8.0
    obstacle_safe_zone = 1.0
    obstacle_min_separation = 0.6
    obstacle_detection_range = 3.0  # Rango de detección

    # Waypoint configuration
    waypoint_spawn_range = 7.0
    waypoint_min_height = 0.5
    waypoint_max_height = 1.5

    # Reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_waypoint_reward_scale = 10.0
    waypoint_reached_bonus = 50.0
    all_waypoints_bonus = 200.0
    obstacle_proximity_reward_scale = -10.0  # Más penalización
    progress_reward_scale = 5.0  # Reward por avanzar hacia waypoint


class QuadcopterObstaclesEnv(DirectRLEnv):
    """Quadcopter with directional obstacle observations."""
    
    cfg: QuadcopterObstaclesEnvCfg

    def __init__(self, cfg: QuadcopterObstaclesEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Actions and forces
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # Waypoints: (num_envs, num_waypoints, 3)
        self._waypoint_positions_local = torch.zeros(
            self.num_envs, self.cfg.num_waypoints, 3, device=self.device
        )
        
        # Waypoint tracking
        self._current_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._waypoints_completed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._all_waypoints_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Previous distance to waypoint (for progress reward)
        self._prev_dist_to_waypoint = torch.zeros(self.num_envs, device=self.device)
        
        # Obstacles: (num_envs, num_obstacles, 3) - ahora con altura
        self._obstacle_positions_local = torch.zeros(
            self.num_envs, self.cfg.num_obstacles, 3, device=self.device
        )
        
        # Initialize
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._randomize_obstacles(all_env_ids)
        self._randomize_waypoints(all_env_ids)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_waypoint",
                "waypoint_bonus",
                "obstacle_proximity",
                "progress",
            ]
        }
        
        # Robot properties
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self._collision_threshold = 0.05

        self.set_debug_vis(self.cfg.debug_vis)

        print(f"[INFO] Quadcopter Obstacles v5 - Directional Observations")
        print(f"[INFO] Obstacles: {self.cfg.num_obstacles} (observing {self.cfg.num_closest_obstacles} closest)")
        print(f"[INFO] Waypoints: {self.cfg.num_waypoints}")
        print(f"[INFO] Observation space: {self.cfg.observation_space}")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _randomize_obstacles(self, env_ids: torch.Tensor):
        """Generate random obstacle positions."""
        num_envs_to_reset = len(env_ids)
        
        angles = torch.zeros(num_envs_to_reset, self.cfg.num_obstacles, device=self.device)
        radii = torch.zeros(num_envs_to_reset, self.cfg.num_obstacles, device=self.device)
        
        for i in range(self.cfg.num_obstacles):
            base_angle = 2 * 3.14159 * i / self.cfg.num_obstacles
            angles[:, i] = base_angle + torch.empty(num_envs_to_reset, device=self.device).uniform_(-0.3, 0.3)
            radii[:, i] = torch.empty(num_envs_to_reset, device=self.device).uniform_(
                self.cfg.obstacle_safe_zone + 0.2,
                self.cfg.obstacle_spawn_range
            )
        
        x = radii * torch.cos(angles)
        y = radii * torch.sin(angles)
        z = torch.ones_like(x) * (self.cfg.obstacle_height / 2)  # Centro del obstáculo
        
        self._obstacle_positions_local[env_ids, :, 0] = x
        self._obstacle_positions_local[env_ids, :, 1] = y
        self._obstacle_positions_local[env_ids, :, 2] = z

    def _randomize_waypoints(self, env_ids: torch.Tensor):
        """Generate random waypoint positions."""
        num_envs_to_reset = len(env_ids)
        
        for i in range(self.cfg.num_waypoints):
            base_angle = 2 * 3.14159 * i / self.cfg.num_waypoints
            angle_variation = torch.empty(num_envs_to_reset, device=self.device).uniform_(-0.4, 0.4)
            angles = base_angle + angle_variation
            
            radii = torch.empty(num_envs_to_reset, device=self.device).uniform_(
                2.0, self.cfg.waypoint_spawn_range
            )
            
            x = radii * torch.cos(angles)
            y = radii * torch.sin(angles)
            z = torch.empty(num_envs_to_reset, device=self.device).uniform_(
                self.cfg.waypoint_min_height,
                self.cfg.waypoint_max_height
            )
            
            self._waypoint_positions_local[env_ids, i, 0] = x
            self._waypoint_positions_local[env_ids, i, 1] = y
            self._waypoint_positions_local[env_ids, i, 2] = z
        
        self._current_waypoint_idx[env_ids] = 0
        self._waypoints_completed[env_ids] = 0
        self._all_waypoints_done[env_ids] = False
        self._prev_dist_to_waypoint[env_ids] = 10.0  # Initial large distance

    def _get_current_waypoint_world(self) -> torch.Tensor:
        """Get current waypoint in world coordinates."""
        batch_indices = torch.arange(self.num_envs, device=self.device)
        current_wp_local = self._waypoint_positions_local[batch_indices, self._current_waypoint_idx]
        
        current_wp_world = current_wp_local.clone()
        current_wp_world[:, :2] += self._terrain.env_origins[:, :2]
        
        return current_wp_world

    def _compute_closest_obstacles_directional(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute direction and distance to N closest obstacles in BODY FRAME.
        Returns:
            directions: (num_envs, num_closest, 3) - unit vectors in body frame
            distances: (num_envs, num_closest) - normalized distances
        """
        # Drone position in world
        drone_pos_w = self._robot.data.root_pos_w  # (num_envs, 3)
        drone_quat_w = self._robot.data.root_quat_w  # (num_envs, 4)
        env_origins = self._terrain.env_origins  # (num_envs, 3)
        
        # Drone position local
        drone_pos_local = drone_pos_w - env_origins  # (num_envs, 3)
        
        # Vector from drone to each obstacle (in local/world frame)
        # drone_pos_local: (num_envs, 3) -> (num_envs, 1, 3)
        # obstacles: (num_envs, num_obstacles, 3)
        drone_expanded = drone_pos_local.unsqueeze(1)  # (num_envs, 1, 3)
        to_obstacles = self._obstacle_positions_local - drone_expanded  # (num_envs, num_obstacles, 3)
        
        # Distances (3D)
        distances = torch.linalg.norm(to_obstacles, dim=2)  # (num_envs, num_obstacles)
        distances = distances - self.cfg.obstacle_radius  # Surface distance
        
        # Get indices of N closest obstacles
        _, closest_indices = torch.topk(distances, self.cfg.num_closest_obstacles, dim=1, largest=False)
        # closest_indices: (num_envs, num_closest)
        
        # Gather closest obstacle vectors
        batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, self.cfg.num_closest_obstacles)
        closest_vectors = to_obstacles[batch_indices, closest_indices]  # (num_envs, num_closest, 3)
        closest_distances = distances[batch_indices, closest_indices]  # (num_envs, num_closest)
        
        # Convert directions to body frame
        # Normalize vectors first
        closest_vectors_norm = closest_vectors / (torch.linalg.norm(closest_vectors, dim=2, keepdim=True) + 1e-6)
        
        # Rotate to body frame using quat_rotate_inverse
        # Need to reshape for the function
        num_closest = self.cfg.num_closest_obstacles
        closest_vectors_flat = closest_vectors_norm.reshape(self.num_envs * num_closest, 3)
        drone_quat_expanded = drone_quat_w.unsqueeze(1).expand(-1, num_closest, -1).reshape(self.num_envs * num_closest, 4)
        
        directions_body = quat_apply_inverse(drone_quat_expanded, closest_vectors_flat)
        
        # Normalize distances
        distances_normalized = (closest_distances / self.cfg.obstacle_detection_range).clamp(0.0, 1.0)
        
        return directions_body, distances_normalized

    def _compute_min_obstacle_distance(self) -> torch.Tensor:
        """Compute minimum distance to any obstacle (for collision/reward)."""
        drone_pos_w = self._robot.data.root_pos_w[:, :2]
        env_origins_xy = self._terrain.env_origins[:, :2]
        drone_pos_local = drone_pos_w - env_origins_xy
        
        drone_expanded = drone_pos_local.unsqueeze(1)
        distances = torch.linalg.norm(
            drone_expanded - self._obstacle_positions_local[:, :, :2], dim=2
        )
        distances = distances - self.cfg.obstacle_radius
        
        return distances.min(dim=1).values

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        """Compute observations with directional obstacle info."""
        # Current waypoint in body frame
        current_wp_world = self._get_current_waypoint_world()
        waypoint_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, current_wp_world
        )
        
        # Waypoint progress
        progress = self._waypoints_completed.float() / self.cfg.num_waypoints
        
        # Closest obstacles (directional)
        obstacle_directions, obstacle_distances = self._compute_closest_obstacles_directional()
        # obstacle_directions: (num_envs, 5, 3)
        # obstacle_distances: (num_envs, 5)
        
        # Flatten obstacle info
        obstacle_dirs_flat = obstacle_directions.reshape(self.num_envs, -1)  # (num_envs, 15)
        
        # Build observation
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,      # 3
                self._robot.data.root_ang_vel_b,      # 3
                self._robot.data.projected_gravity_b, # 3
                waypoint_pos_b,                        # 3
                obstacle_dirs_flat,                    # 15 (5 obstacles * 3 dims)
                obstacle_distances,                    # 5
                progress.unsqueeze(1),                 # 1
            ],
            dim=-1,
        )  # Total: 33
        
        # Padding (98 - 33 = 65)
        padding = torch.zeros(self.num_envs, 65, device=self.device)
        obs = torch.cat([obs, padding], dim=-1)
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        # Velocity penalties
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        
        # Distance to current waypoint
        current_wp_world = self._get_current_waypoint_world()
        distance_to_waypoint = torch.linalg.norm(current_wp_world - self._robot.data.root_pos_w, dim=1)
        distance_reward = 1 - torch.tanh(distance_to_waypoint / 2.0)
        
        # Progress reward (getting closer to waypoint)
        progress_reward = (self._prev_dist_to_waypoint - distance_to_waypoint).clamp(-1.0, 1.0)
        self._prev_dist_to_waypoint = distance_to_waypoint.clone()
        
        # Waypoint reached check
        waypoint_reached = (distance_to_waypoint < self.cfg.waypoint_reach_threshold) & (~self._all_waypoints_done)
        
        waypoint_bonus = torch.zeros(self.num_envs, device=self.device)
        
        if waypoint_reached.any():
            waypoint_bonus[waypoint_reached] = self.cfg.waypoint_reached_bonus
            
            self._current_waypoint_idx[waypoint_reached] += 1
            self._waypoints_completed[waypoint_reached] += 1
            
            # Reset prev distance for new waypoint
            self._prev_dist_to_waypoint[waypoint_reached] = 10.0
            
            all_done_now = self._current_waypoint_idx >= self.cfg.num_waypoints
            first_time_all_done = all_done_now & (~self._all_waypoints_done)
            
            waypoint_bonus[first_time_all_done] += self.cfg.all_waypoints_bonus
            
            self._all_waypoints_done = self._all_waypoints_done | all_done_now
            self._current_waypoint_idx = self._current_waypoint_idx.clamp(0, self.cfg.num_waypoints - 1)
        
        # Obstacle proximity penalty
        min_obstacle_dist = self._compute_min_obstacle_distance()
        
        obstacle_proximity = torch.where(
            min_obstacle_dist < 1.0,
            torch.exp(-min_obstacle_dist * 3.0),  # Más agresivo
            torch.zeros_like(min_obstacle_dist)
        )
        
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_waypoint": distance_reward * self.cfg.distance_to_waypoint_reward_scale * self.step_dt,
            "waypoint_bonus": waypoint_bonus,
            "obstacle_proximity": obstacle_proximity * self.cfg.obstacle_proximity_reward_scale * self.step_dt,
            "progress": progress_reward * self.cfg.progress_reward_scale * self.step_dt,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        for key, value in rewards.items():
            self._episode_sums[key] += value
            
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        too_low = self._robot.data.root_pos_w[:, 2] < 0.1
        too_high = self._robot.data.root_pos_w[:, 2] > 2.5
        
        min_obstacle_dist = self._compute_min_obstacle_distance()
        collision = min_obstacle_dist < self._collision_threshold
        
        success = self._all_waypoints_done
        
        died = too_low | too_high | collision
        terminated = died | success
        
        if self.common_step_counter % 500 == 0:
            avg_completed = self._waypoints_completed.float().mean().item()
            print(f"[DEBUG] Step {self.common_step_counter}: died={died.sum().item()}, "
                  f"success={success.sum().item()}, collision={collision.sum().item()}, "
                  f"avg_waypoints={avg_completed:.1f}/{self.cfg.num_waypoints}")
        
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        avg_waypoints = self._waypoints_completed[env_ids].float().mean().item()
        success_rate = self._all_waypoints_done[env_ids].float().mean().item()
        
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/avg_waypoints_reached"] = avg_waypoints
        extras["Metrics/success_rate"] = success_rate
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        
        self._randomize_obstacles(env_ids)
        self._randomize_waypoints(env_ids)
        
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "waypoint_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
                marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0)
                )
                marker_cfg.prim_path = "/Visuals/CurrentWaypoint"
                self.waypoint_visualizer = VisualizationMarkers(marker_cfg)
            self.waypoint_visualizer.set_visibility(True)
            
            if not hasattr(self, "future_waypoint_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.1)
                marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 1.0, 0.0)
                )
                marker_cfg.prim_path = "/Visuals/FutureWaypoints"
                self.future_waypoint_visualizer = VisualizationMarkers(marker_cfg)
            self.future_waypoint_visualizer.set_visibility(True)
            
            if not hasattr(self, "obstacle_visualizer"):
                obs_marker_cfg = CUBOID_MARKER_CFG.copy()
                pillar_size = self.cfg.obstacle_radius * 2
                obs_marker_cfg.markers["cuboid"].size = (pillar_size, pillar_size, self.cfg.obstacle_height)
                obs_marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.2, 0.2)
                )
                obs_marker_cfg.prim_path = "/Visuals/Obstacles"
                self.obstacle_visualizer = VisualizationMarkers(obs_marker_cfg)
            self.obstacle_visualizer.set_visibility(True)
        else:
            if hasattr(self, "waypoint_visualizer"):
                self.waypoint_visualizer.set_visibility(False)
            if hasattr(self, "future_waypoint_visualizer"):
                self.future_waypoint_visualizer.set_visibility(False)
            if hasattr(self, "obstacle_visualizer"):
                self.obstacle_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "waypoint_visualizer"):
            current_wp_world = self._get_current_waypoint_world()
            self.waypoint_visualizer.visualize(current_wp_world)
        
        if hasattr(self, "future_waypoint_visualizer"):
            all_future = []
            for env_idx in range(min(64, self.num_envs)):
                current_idx = self._current_waypoint_idx[env_idx].item()
                for wp_idx in range(current_idx + 1, self.cfg.num_waypoints):
                    wp_local = self._waypoint_positions_local[env_idx, wp_idx]
                    wp_world = wp_local.clone()
                    wp_world[0] += self._terrain.env_origins[env_idx, 0]
                    wp_world[1] += self._terrain.env_origins[env_idx, 1]
                    all_future.append(wp_world)
            
            if all_future:
                future_positions = torch.stack(all_future)
                self.future_waypoint_visualizer.visualize(future_positions)
        
        if hasattr(self, "obstacle_visualizer"):
            env_origins = self._terrain.env_origins
            env_origins_expanded = env_origins.unsqueeze(1).repeat(1, self.cfg.num_obstacles, 1)
            
            obstacle_pos_w = self._obstacle_positions_local.clone()
            obstacle_pos_w[:, :, :2] += env_origins_expanded[:, :, :2]
            
            obstacle_pos_flat = obstacle_pos_w.reshape(-1, 3)
            self.obstacle_visualizer.visualize(obstacle_pos_flat)