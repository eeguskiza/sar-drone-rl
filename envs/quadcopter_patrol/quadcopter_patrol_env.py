# Copyright (c) 2026 Alex Jauregui & Erik Eguskiza.
# Stage 3: Patrol - Deterministic Forest, 24 waypoints spiral
# Preparado para Stage 4: Detección de personas

from __future__ import annotations

import gymnasium as gym
import torch
import math

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
class QuadcopterPatrolEnvCfg(DirectRLEnvCfg):
    """Configuration for Patrol with Deterministic Forest."""
    
    # === Episode ===
    episode_length_s = 60.0  # Reducido - ruta memorizable no necesita tanto
    decimation = 2
    
    # === Map ===
    map_size = 15.0  # ±15m = 30x30m total
    
    # === Deterministic Forest ===
    num_obstacles = 50
    num_closest_obstacles = 5
    obstacle_seed = 42  # Seed fijo para bosque determinista
    
    # === Patrol ===
    num_patrol_waypoints = 24
    patrol_height = 1.2
    waypoint_reach_threshold = 1.2
    
    # === Observation: 33 dims (sin padding innecesario) ===
    # lin_vel(3) + ang_vel(3) + gravity(3) + waypoint_dir(3) + 
    # obstacle_dirs(5*3=15) + obstacle_dists(5) + progress(1) = 33
    observation_space = 33
    action_space = 4
    state_space = 0
    debug_vis = True

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

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=40.0, replicate_physics=True
    )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # === Obstacle Config ===
    obstacle_height = 1.5
    obstacle_radius = 0.15
    obstacle_spawn_range = 13.0
    obstacle_safe_zone = 2.5  # Zona segura alrededor del spawn
    obstacle_detection_range = 3.0
    collision_threshold = 0.08

    # === Reward Scales ===
    # Penalizaciones de control (pequeñas)
    lin_vel_reward_scale = -0.005
    ang_vel_reward_scale = -0.003
    
    # Navegación
    distance_to_waypoint_reward_scale = 12.0
    progress_reward_scale = 8.0
    
    # Bonus escalonado por capa (incentiva capas exteriores)
    waypoint_bonus_layer1 = 20.0   # WP 1-6
    waypoint_bonus_layer2 = 30.0   # WP 7-12
    waypoint_bonus_layer3 = 45.0   # WP 13-18
    waypoint_bonus_layer4 = 60.0   # WP 19-24
    patrol_complete_bonus = 250.0
    
    # Obstáculos
    obstacle_proximity_reward_scale = -2.0


class QuadcopterPatrolEnv(DirectRLEnv):
    cfg: QuadcopterPatrolEnvCfg

    def __init__(self, cfg: QuadcopterPatrolEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # === Actions & Forces ===
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # === Patrol State ===
        self._patrol_waypoints = self._generate_patrol_pattern()
        self._current_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._waypoints_completed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._patrol_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_dist_to_waypoint = torch.zeros(self.num_envs, device=self.device)
        
        # === Deterministic Forest ===
        self._obstacle_positions_local = torch.zeros(
            self.num_envs, self.cfg.num_obstacles, 3, device=self.device
        )
        self._generate_deterministic_forest()
        
        # === Metrics ===
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["lin_vel", "ang_vel", "distance_to_waypoint", "waypoint_bonus", "obstacle_proximity", "progress"]
        }
        
        # Collision tracking
        self._collision_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # === Robot Physics ===
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        
        self.set_debug_vis(self.cfg.debug_vis)
        self._print_config_summary()

    def _print_config_summary(self):
        """Print configuration summary at startup."""
        print("\n" + "="*60)
        print("  QUADCOPTER PATROL - DETERMINISTIC FOREST")
        print("="*60)
        print(f"  Map Size:      {self.cfg.map_size * 2:.0f}m x {self.cfg.map_size * 2:.0f}m")
        print(f"  Obstacles:     {self.cfg.num_obstacles} (FIXED - seed {self.cfg.obstacle_seed})")
        print(f"  Waypoints:     {self.cfg.num_patrol_waypoints} (4-layer spiral)")
        print(f"  Environments:  {self.num_envs}")
        print("-"*60)
        print("  Waypoint Bonuses (scaled by layer):")
        print(f"    Layer 1 (WP 1-6):   +{self.cfg.waypoint_bonus_layer1}")
        print(f"    Layer 2 (WP 7-12):  +{self.cfg.waypoint_bonus_layer2}")
        print(f"    Layer 3 (WP 13-18): +{self.cfg.waypoint_bonus_layer3}")
        print(f"    Layer 4 (WP 19-24): +{self.cfg.waypoint_bonus_layer4}")
        print(f"    Patrol Complete:    +{self.cfg.patrol_complete_bonus}")
        print("-"*60)
        print("  Observations: 33 dims")
        print("    - Body velocities (lin/ang): 6")
        print("    - Projected gravity: 3")
        print("    - Waypoint direction (body): 3")
        print("    - Obstacle dirs (5x3): 15")
        print("    - Obstacle distances (5): 5")
        print("    - Progress: 1")
        print("="*60 + "\n")

    def _generate_patrol_pattern(self) -> torch.Tensor:
        """Generate 4-layer spiral pattern from center - 24 waypoints."""
        h = self.cfg.patrol_height
        
        # 4 capas concéntricas, 6 waypoints por capa
        c1, c2, c3, c4 = 2.5, 5.5, 9.0, 13.0
        
        waypoints = [
            # WP0: Start (centro)
            [0.0, 0.0, h],
            # Layer 1 - interior (WP 1-6)
            [c1, 0.0, h], [c1, c1, h], [-c1, c1, h], [-c1, -c1, h], [c1, -c1, h], [c1, 0.0, h],
            # Layer 2 (WP 7-12)
            [c2, 0.0, h], [c2, c2, h], [-c2, c2, h], [-c2, -c2, h], [c2, -c2, h], [c2, 0.0, h],
            # Layer 3 (WP 13-18)
            [c3, 0.0, h], [c3, c3, h], [-c3, c3, h], [-c3, -c3, h], [c3, -c3, h], [c3, 0.0, h],
            # Layer 4 - exterior (WP 19-24)
            [c4, 0.0, h], [c4, c4, h], [-c4, c4, h], [-c4, -c4, h], [c4, -c4, h], [c4, 0.0, h],
        ]
        
        # Calcular distancia total
        total_dist = sum(
            math.sqrt((waypoints[i][0]-waypoints[i-1][0])**2 + (waypoints[i][1]-waypoints[i-1][1])**2)
            for i in range(1, len(waypoints))
        )
        
        print(f"[PATROL] Path: {len(waypoints)} waypoints, {total_dist:.1f}m total")
        return torch.tensor(waypoints, device=self.device, dtype=torch.float32)

    def _generate_deterministic_forest(self):
        """Generate fixed obstacle positions using seed."""
        print(f"[FOREST] Generating deterministic forest (seed={self.cfg.obstacle_seed})...")
        
        # Usar generador con seed fijo
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.cfg.obstacle_seed)
        
        # Generar posiciones base (mismas para todos los envs)
        base_positions = torch.zeros(self.cfg.num_obstacles, 3, device=self.device)
        
        for i in range(self.cfg.num_obstacles):
            valid = False
            attempts = 0
            while not valid and attempts < 100:
                x = torch.empty(1, device=self.device).uniform_(
                    -self.cfg.obstacle_spawn_range, self.cfg.obstacle_spawn_range, generator=gen
                ).item()
                y = torch.empty(1, device=self.device).uniform_(
                    -self.cfg.obstacle_spawn_range, self.cfg.obstacle_spawn_range, generator=gen
                ).item()
                
                # Check safe zone around spawn
                dist_from_center = math.sqrt(x**2 + y**2)
                if dist_from_center > self.cfg.obstacle_safe_zone:
                    # Check distance from other obstacles
                    if i == 0:
                        valid = True
                    else:
                        min_dist = min(
                            math.sqrt((x - base_positions[j, 0].item())**2 + (y - base_positions[j, 1].item())**2)
                            for j in range(i)
                        )
                        valid = min_dist > self.cfg.obstacle_radius * 3  # Mínimo 3 radios entre obstáculos
                attempts += 1
            
            base_positions[i, 0] = x
            base_positions[i, 1] = y
            base_positions[i, 2] = self.cfg.obstacle_height / 2
        
        # Copiar a todos los envs (mismo bosque)
        self._obstacle_positions_local = base_positions.unsqueeze(0).expand(self.num_envs, -1, -1).clone()
        
        # Log estadísticas del bosque
        distances_from_center = torch.sqrt(base_positions[:, 0]**2 + base_positions[:, 1]**2)
        print(f"[FOREST] Obstacles placed: {self.cfg.num_obstacles}")
        print(f"[FOREST] Distance from center - min: {distances_from_center.min():.1f}m, max: {distances_from_center.max():.1f}m")
        print(f"[FOREST] Distribution by zone:")
        print(f"         < 5m:  {(distances_from_center < 5).sum().item()} obstacles")
        print(f"         5-9m:  {((distances_from_center >= 5) & (distances_from_center < 9)).sum().item()} obstacles")
        print(f"         9-13m: {((distances_from_center >= 9) & (distances_from_center < 13)).sum().item()} obstacles")
        print(f"         >13m:  {(distances_from_center >= 13).sum().item()} obstacles")

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

    def _get_waypoint_bonus(self, waypoint_idx: torch.Tensor) -> torch.Tensor:
        """Get bonus based on waypoint layer (higher bonus for outer layers).
        
        Args:
            waypoint_idx: Tensor of waypoint indices [N] for envs that reached a waypoint
        Returns:
            Tensor of bonuses [N] corresponding to each waypoint
        """
        bonus = torch.zeros_like(waypoint_idx, dtype=torch.float)
        
        # Layer 1: WP 1-6
        layer1 = (waypoint_idx >= 1) & (waypoint_idx <= 6)
        bonus[layer1] = self.cfg.waypoint_bonus_layer1
        
        # Layer 2: WP 7-12
        layer2 = (waypoint_idx >= 7) & (waypoint_idx <= 12)
        bonus[layer2] = self.cfg.waypoint_bonus_layer2
        
        # Layer 3: WP 13-18
        layer3 = (waypoint_idx >= 13) & (waypoint_idx <= 18)
        bonus[layer3] = self.cfg.waypoint_bonus_layer3
        
        # Layer 4: WP 19-24
        layer4 = (waypoint_idx >= 19) & (waypoint_idx <= 24)
        bonus[layer4] = self.cfg.waypoint_bonus_layer4
        
        return bonus

    def _reset_patrol(self, env_ids: torch.Tensor):
        self._current_waypoint_idx[env_ids] = 0
        self._waypoints_completed[env_ids] = 0
        self._patrol_complete[env_ids] = False
        self._prev_dist_to_waypoint[env_ids] = 2.5  # Distancia inicial aproximada al WP1
        self._collision_count[env_ids] = 0

    def _get_current_waypoint_world(self) -> torch.Tensor:
        current_wp_local = self._patrol_waypoints[self._current_waypoint_idx]
        current_wp_world = current_wp_local.clone()
        current_wp_world[:, :2] += self._terrain.env_origins[:, :2]
        return current_wp_world

    def _compute_closest_obstacles_directional(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute directions and distances to closest obstacles in body frame."""
        drone_pos_w = self._robot.data.root_pos_w
        drone_quat_w = self._robot.data.root_quat_w
        drone_pos_local = drone_pos_w - self._terrain.env_origins
        
        to_obstacles = self._obstacle_positions_local - drone_pos_local.unsqueeze(1)
        distances = torch.linalg.norm(to_obstacles, dim=2) - self.cfg.obstacle_radius
        
        _, closest_indices = torch.topk(distances, self.cfg.num_closest_obstacles, dim=1, largest=False)
        batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, self.cfg.num_closest_obstacles)
        
        closest_vectors = to_obstacles[batch_indices, closest_indices]
        closest_distances = distances[batch_indices, closest_indices]
        
        # Normalize directions
        closest_vectors_norm = closest_vectors / (torch.linalg.norm(closest_vectors, dim=2, keepdim=True) + 1e-6)
        
        # Transform to body frame
        num_closest = self.cfg.num_closest_obstacles
        closest_vectors_flat = closest_vectors_norm.reshape(self.num_envs * num_closest, 3)
        drone_quat_expanded = drone_quat_w.unsqueeze(1).expand(-1, num_closest, -1).reshape(self.num_envs * num_closest, 4)
        directions_body = quat_apply_inverse(drone_quat_expanded, closest_vectors_flat).reshape(self.num_envs, num_closest, 3)
        
        # Normalize distances
        distances_normalized = (closest_distances / self.cfg.obstacle_detection_range).clamp(0.0, 1.0)
        
        return directions_body, distances_normalized

    def _compute_min_obstacle_distance(self) -> torch.Tensor:
        drone_pos_local = self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:, :2]
        distances = torch.linalg.norm(drone_pos_local.unsqueeze(1) - self._obstacle_positions_local[:, :, :2], dim=2)
        return (distances - self.cfg.obstacle_radius).min(dim=1).values

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        current_wp_world = self._get_current_waypoint_world()
        waypoint_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, current_wp_world
        )
        
        # Normalize waypoint direction
        waypoint_dist = torch.linalg.norm(waypoint_pos_b, dim=1, keepdim=True) + 1e-6
        waypoint_dir_b = waypoint_pos_b / waypoint_dist
        
        progress = self._waypoints_completed.float() / self.cfg.num_patrol_waypoints
        obstacle_directions, obstacle_distances = self._compute_closest_obstacles_directional()
        
        obs = torch.cat([
            self._robot.data.root_lin_vel_b,           # 3
            self._robot.data.root_ang_vel_b,           # 3
            self._robot.data.projected_gravity_b,      # 3
            waypoint_dir_b,                            # 3
            obstacle_directions.reshape(self.num_envs, -1),  # 15
            obstacle_distances,                        # 5
            progress.unsqueeze(1),                     # 1
        ], dim=-1)  # Total: 33
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # === Control penalties ===
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        
        # === Waypoint navigation ===
        current_wp_world = self._get_current_waypoint_world()
        distance_to_waypoint = torch.linalg.norm(current_wp_world - self._robot.data.root_pos_w, dim=1)
        
        # Distance reward - exponential decay
        distance_reward = torch.exp(-distance_to_waypoint / 2.5)
        
        # Progress reward - approaching waypoint
        progress_reward = (self._prev_dist_to_waypoint - distance_to_waypoint).clamp(-0.3, 0.3)
        self._prev_dist_to_waypoint = distance_to_waypoint.clone()
        
        # === Waypoint reached ===
        waypoint_reached = (distance_to_waypoint < self.cfg.waypoint_reach_threshold) & (~self._patrol_complete)
        waypoint_bonus = torch.zeros(self.num_envs, device=self.device)
        
        if waypoint_reached.any():
            reached_idx = self._current_waypoint_idx[waypoint_reached]
            waypoint_bonus[waypoint_reached] = self._get_waypoint_bonus(reached_idx)
            
            # Update waypoint index
            self._current_waypoint_idx[waypoint_reached] += 1
            self._waypoints_completed[waypoint_reached] += 1
            
            # Check patrol complete
            patrol_done = self._current_waypoint_idx >= self.cfg.num_patrol_waypoints
            waypoint_bonus[patrol_done & (~self._patrol_complete)] += self.cfg.patrol_complete_bonus
            self._patrol_complete = self._patrol_complete | patrol_done
            self._current_waypoint_idx = self._current_waypoint_idx.clamp(0, self.cfg.num_patrol_waypoints - 1)
            
            # Update distance to new waypoint
            new_wp_world = self._get_current_waypoint_world()
            self._prev_dist_to_waypoint[waypoint_reached] = torch.linalg.norm(
                new_wp_world - self._robot.data.root_pos_w, dim=1
            )[waypoint_reached]
        
        # === Obstacle proximity ===
        min_obstacle_dist = self._compute_min_obstacle_distance()
        obstacle_proximity = torch.where(
            min_obstacle_dist < 1.0,
            torch.exp(-min_obstacle_dist * 3.0),
            torch.zeros_like(min_obstacle_dist)
        )
        
        # === Compile rewards ===
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
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Death conditions
        too_low = self._robot.data.root_pos_w[:, 2] < 0.1
        too_high = self._robot.data.root_pos_w[:, 2] > 3.0
        collision = self._compute_min_obstacle_distance() < self.cfg.collision_threshold
        
        # Track collisions
        self._collision_count[collision] += 1
        
        died = too_low | too_high | collision
        terminated = died | self._patrol_complete
        
        # === Periodic logging ===
        if self.common_step_counter % 1000 == 0 and self.common_step_counter > 0:
            self._log_training_status(died, collision)
        
        return terminated, time_out

    def _log_training_status(self, died: torch.Tensor, collision: torch.Tensor):
        """Log detailed training metrics."""
        avg_wp = self._waypoints_completed.float().mean().item()
        success_rate = self._patrol_complete.float().mean().item() * 100
        collision_rate = collision.float().mean().item() * 100
        death_rate = died.float().mean().item() * 100
        
        # Layer-wise progress
        layer_progress = [
            (self._waypoints_completed >= end).float().mean().item() * 100
            for end in [6, 12, 18, 24]
        ]
        
        print(f"\n{'='*50}")
        print(f"  Step {self.common_step_counter:,}")
        print(f"{'='*50}")
        print(f"  Waypoints:   {avg_wp:.1f}/24 ({avg_wp/24*100:.0f}%)")
        print(f"  Success:     {success_rate:.1f}%")
        print(f"  Deaths:      {death_rate:.1f}% (collisions: {collision_rate:.1f}%)")
        print(f"  Layer completion:")
        print(f"    L1 (WP 1-6):   {layer_progress[0]:5.1f}%")
        print(f"    L2 (WP 7-12):  {layer_progress[1]:5.1f}%")
        print(f"    L3 (WP 13-18): {layer_progress[2]:5.1f}%")
        print(f"    L4 (WP 19-24): {layer_progress[3]:5.1f}%")
        print(f"{'='*50}\n")

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        
        # === Log episode metrics ===
        extras = dict()
        for key in self._episode_sums.keys():
            extras["Episode_Reward/" + key] = torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        self.extras["log"]["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"]["Metrics/patrol_success_rate"] = self._patrol_complete[env_ids].float().mean().item()
        self.extras["log"]["Metrics/avg_waypoints"] = self._waypoints_completed[env_ids].float().mean().item()
        
        # Layer metrics
        for i, (start, end) in enumerate([(1, 6), (7, 12), (13, 18), (19, 24)]):
            layer_complete = (self._waypoints_completed[env_ids] >= end).float().mean().item()
            self.extras["log"][f"Metrics/layer{i+1}_complete"] = layer_complete
        
        # === Reset robot ===
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        
        self._actions[env_ids] = 0.0
        self._reset_patrol(env_ids)
        
        # Reset robot state (spawn at center)
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos[env_ids],
            self._robot.data.default_joint_vel[env_ids],
            None, env_ids
        )

    # === Debug Visualization ===
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "waypoint_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.4, 0.4, 0.4)
                marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
                marker_cfg.prim_path = "/Visuals/CurrentWaypoint"
                self.waypoint_visualizer = VisualizationMarkers(marker_cfg)
            self.waypoint_visualizer.set_visibility(True)
            
            if not hasattr(self, "obstacle_visualizer"):
                obs_marker_cfg = CUBOID_MARKER_CFG.copy()
                obs_marker_cfg.markers["cuboid"].size = (
                    self.cfg.obstacle_radius * 2,
                    self.cfg.obstacle_radius * 2,
                    self.cfg.obstacle_height
                )
                obs_marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.3, 0.1))
                obs_marker_cfg.prim_path = "/Visuals/Obstacles"
                self.obstacle_visualizer = VisualizationMarkers(obs_marker_cfg)
            self.obstacle_visualizer.set_visibility(True)
        else:
            for attr in ["waypoint_visualizer", "obstacle_visualizer"]:
                if hasattr(self, attr):
                    getattr(self, attr).set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "waypoint_visualizer"):
            self.waypoint_visualizer.visualize(self._get_current_waypoint_world())
        if hasattr(self, "obstacle_visualizer"):
            env_origins = self._terrain.env_origins.unsqueeze(1).repeat(1, self.cfg.num_obstacles, 1)
            obstacle_pos_w = self._obstacle_positions_local.clone()
            obstacle_pos_w[:, :, :2] += env_origins[:, :, :2]
            self.obstacle_visualizer.visualize(obstacle_pos_w.reshape(-1, 3))