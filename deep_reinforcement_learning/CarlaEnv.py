"""
CarlaEnv.py — Gymnasium environment for CARLA 0.9.16 PPO highway lane-following.
Observation:  (4, 66, 200) float32 semantic binary channels
Action:       Box([-1, 0], [1, 1], shape=(2,)) -> [steering, throttle]
"""
import math
import time
import threading
import random
import numpy as np
import cv2
import pygame
import carla
import gymnasium as gym
from gymnasium import spaces


# ==============================================================================
# -- Semantic Channel Decoder --------------------------------------------------
# ==============================================================================

# CARLA Class IDs for each semantic channel (4-channel subset used in IL)
CHANNEL_IDS = [
    {24},                               # C0 – RoadLines
    {1},                                # C1 – Roads
    {12, 13, 14, 15, 16, 18, 19},       # C2 – Dynamics
    {2, 5, 28},                         # C3 – Borders
]

IMG_H, IMG_W = 150, 200   # Full-frame resize target (4:3 ratio, half of 400x300)


def decode_semantic(raw_array, height, width):
    """Convert raw CARLA semantic BGRA buffer to (4, 150, 200) binary tensor."""
    arr = raw_array.reshape((height, width, 4))
    # Class IDs are stored in the Red channel of semantic segmentation frames
    class_ids = arr[:, :, 2].copy()

    # Resize full frame to (150, 200) with NEAREST to preserve IDs
    frame_resized = cv2.resize(class_ids, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

    tensor = np.zeros((4, IMG_H, IMG_W), dtype=np.float32)
    for ch, ids in enumerate(CHANNEL_IDS):
        mask = np.zeros_like(frame_resized, dtype=bool)
        for cid in ids:
            mask |= (frame_resized == cid)
        tensor[ch] = mask.astype(np.float32)

    return tensor


# ==============================================================================
# -- CarlaEnv ------------------------------------------------------------------
# ==============================================================================

class CarlaEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        host="127.0.0.1",
        port=2000,
        render=True,
        max_steps=1000,
        target_speed_kmh=40.0,
        async_mode=False,
    ):
        super().__init__()
        self.async_mode = async_mode

        # --- Gymnasium spaces ---
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4, IMG_H, IMG_W), dtype=np.float32
        )
        # [steering, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # --- Config ---
        self.render_mode = "human" if render else None
        self.max_steps = max_steps
        self.target_speed = target_speed_kmh
        self._step = 0
        self._slow_steps = 0
        self._prev_steer = 0.0

        # --- CARLA client ---
        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world("Town04")

        # --- World Settings ---
        settings = self.world.get_settings()
        if self.async_mode:
            print("[CarlaEnv] Starting in ASYNC mode...")
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
        else:
            print("[CarlaEnv] Starting in SYNC mode (Training)...")
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 Hz
        
        self.world.apply_settings(settings)
        
        if not self.async_mode:
            self.tm = self.client.get_trafficmanager(8000)
            self.tm.set_synchronous_mode(True)

        self.bp_lib = self.world.get_blueprint_library()
        self._map = self.world.get_map()

        # Pre-compute highway spawn points from Town04 waypoints
        self._highway_waypoints = self._get_highway_waypoints()
        print(f"[CarlaEnv] Found {len(self._highway_waypoints)} highway waypoints.")

        # Actor handles
        self.vehicle = None
        self.cam_sem = None
        self.cam_rgb = None
        self.col_sensor = None
        self.lane_sensor = None

        # Sensor data buffers
        self._sem_obs = np.zeros((4, IMG_H, IMG_W), dtype=np.float32)
        self._sem_raw = np.zeros((300, 400, 3), dtype=np.uint8)  # Full-res display buffer
        self._sem_vis_tmp = np.zeros((300, 400, 3), dtype=np.uint8) # Reusable mapping buffer
        self._rgb_raw = np.zeros((600, 800, 3), dtype=np.uint8)  # Latest RGB frame
        self._sem_event = threading.Event()
        self._printed_ids = False  # One-time debug flag
        self._collision_flag = False
        self._invasion_flag = False

        # --- Pygame display (Pre-allocate to avoid GC churn) ---
        self._display = None
        self._rgb_surface = None
        self._sem_surface = None
        self._pygame_initialized = False

    # --------------------------------------------------------------------------
    # Gymnasium interface
    # --------------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        print("[DEBUG] reset: starting", flush=True)
        super().reset(seed=seed)
        print("[DEBUG] reset: cleaning up actors", flush=True)
        self._cleanup_actors()
        self._collision_flag = False
        self._invasion_flag = False
        self._step = 0
        self._slow_steps = 0
        self._prev_steer = 0.0

        print("[DEBUG] reset: finding spawn point", flush=True)
        # Spawn vehicle at a random highway waypoint
        spawn_wp = random.choice(self._highway_waypoints)
        spawn_tf = spawn_wp.transform
        spawn_tf.location.z += 0.5  # Avoid spawning inside the road

        vehicle_bp = self.bp_lib.find("vehicle.lincoln.mkz_2020")
        print("[DEBUG] reset: spawning vehicle", flush=True)
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_tf)
        self.vehicle.set_autopilot(False)

        print("[DEBUG] reset: attaching sensors", flush=True)
        # Attach sensors
        self._attach_semantic_camera()
        self._attach_rgb_camera()
        self._attach_collision_sensor()
        self._attach_lane_invasion_sensor()

        # Initial warmup ticks (only needed in SYNC mode)
        self._block_sensors = True  # Stop processing 30 frames and starving Python GIL
        print("[DEBUG] reset: entering warmup ticks", flush=True)
        if not self.async_mode:
            print("[CarlaEnv] Warming up sensors (30 ticks)...", flush=True)
            for _ in range(30):
                self.world.tick()
        else:
            # In async mode just wait briefly for sensors to fire
            time.sleep(1.0)

        # Wait for first semantic frame
        print("[CarlaEnv] Waiting for first actual sensor frame...", flush=True)
        self._block_sensors = False
        self._sem_event.clear()
        if not self.async_mode:
            self.world.tick()
        self._sem_event.wait(timeout=5.0)
        self._sem_event.clear()

        # Track initial lane position to ensure strict lane-following
        veh_loc = self.vehicle.get_location()
        self.start_wp = self._map.get_waypoint(veh_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        if self.start_wp:
            self.start_lane_id = self.start_wp.lane_id
            self.start_road_id = self.start_wp.road_id
        else:
            # Fallback if spawn is weird
            self.start_lane_id = 0
            self.start_road_id = 0

        # DO NOT render the first frame here. Pygame init is deferred to the first 
        # actual step to avoid the "Not Responding" window freeze during PPO compile.

        obs = self._sem_obs.copy()
        print(f"[CarlaEnv] Episode started. Starting Lane ID: {self.start_lane_id}", flush=True)
        print("[DEBUG] reset: fully completed", flush=True)
        return obs, {}

    def step(self, action):
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))

        # If throttle is very low, apply slight brake to decelerate naturally
        brake = 0.0
        if throttle < 0.05:
            brake = 0.3

        ctrl = carla.VehicleControl(
            throttle=throttle,
            steer=steering,
            brake=brake,
        )
        print(ctrl)
        self.vehicle.apply_control(ctrl)
        if not self.async_mode:
            self.world.tick()

        # --- Reward and Status ---
        reward, terminated_reason = self._compute_reward(steering, throttle)
        terminated = False
        truncated = False

        if self._collision_flag:
            reward = -100.0
            terminated = True
            terminated_reason = "COLLISION"
            print(f"[Episode END] {terminated_reason} at step {self._step} | Reward: {reward}")

        elif self._invasion_flag:
            reward = -100.0
            terminated = True
            terminated_reason = "LANE INVASION"
            print(f"[Episode END] {terminated_reason} at step {self._step} | Reward: {reward}")
            
        elif terminated_reason:
            terminated = True
            # Penalty already applied in _compute_reward for road-exit
            print(f"[Episode END] {terminated_reason} at step {self._step} | Reward: {reward}")

        self._prev_steer = steering
        self._step += 1
        if self._step >= self.max_steps:
            truncated = True

        # Render from main thread (safe for Pygame GL context)
        if self.render_mode == "human":
            self.render()

        # Wait for semantic frame
        self._sem_event.wait(timeout=0.2)
        self._sem_event.clear()
        obs = self._sem_obs.copy()

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return
            
        if not getattr(self, '_pygame_initialized', False):
            pygame.init()
            self._display = pygame.display.set_mode((1200, 600))
            pygame.display.set_caption("CARLA — DRL Training")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont(None, 22)
            self._rgb_surface = pygame.Surface((800, 600))
            self._sem_surface = pygame.Surface((400, 300))
            self._pygame_initialized = True

        if self._display is None:
            return
        
        self._display.fill((10, 10, 10))

        # --- Left panel: RGB (800x600) ---
        rgb_surf = pygame.surfarray.make_surface(self._rgb_raw.swapaxes(0, 1))
        self._display.blit(rgb_surf, (0, 0))

        # --- Right panel: Semantic (400x300 at top) ---
        sem_surf = pygame.surfarray.make_surface(self._sem_raw.swapaxes(0, 1))
        self._display.blit(sem_surf, (800, 0))

        if self._step % 100 == 0:
            print(f"[CarlaEnv] Render Heartbeat - Step {self._step}")

        # --- HUD (Telemetry) ---
        if self.vehicle is not None:
            v = self.vehicle.get_velocity()
            spd = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
            self._display.blit(self._font.render(f"Speed: {spd:.1f} km/h", True, (255, 255, 255)), (10, 10))
            self._display.blit(self._font.render(f"Step:  {self._step}", True, (200, 200, 200)), (10, 32))

        # --- Legend (Below semantic view) ---
        colors_legend = [
            ((255, 255,  50), "C0 RoadLines (24)"),
            (( 50, 200,  50), "C1 Roads (1)"),
            ((255,  60,  60), "C2 Dynamics (12-19)"),
            (( 60, 100, 255), "C3 Borders (2,5,28)"),
        ]
        for i, (color, label) in enumerate(colors_legend):
            pygame.draw.rect(self._display, color, (810, 320 + i * 22, 14, 14))
            self._display.blit(self._font.render(label, True, (200, 200, 200)), (835, 320 + i * 22))

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    # Removed _build_sem_surface in favor of direct blit_array in render()
    
    def close(self):
        self._cleanup_actors()
        # Restore async mode
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        if self._display is not None:
            pygame.quit()
            self._display = None

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------

    def _get_highway_waypoints(self):
        """Filter Town04 waypoints keeping only straight highway driving lanes."""
        all_wp = self._map.generate_waypoints(10.0)
        highway = []
        for wp in all_wp:
            if (
                wp.lane_type == carla.LaneType.Driving
                and not wp.is_junction
                and wp.road_id in self._get_town04_highway_road_ids()
            ):
                highway.append(wp)
        # Fallback: if filter is too restrictive, use all non-junction Driving lanes
        if len(highway) < 50:
            highway = [
                wp for wp in all_wp
                if wp.lane_type == carla.LaneType.Driving and not wp.is_junction
            ]
        return highway

    def _get_town04_highway_road_ids(self):
        """
        Town04 main highway road IDs (the large oval circuit).
        These are the primary straight/curved highway segments, excluding urban zones.
        Identified by sampling waypoints with speed limits >= 90 km/h.
        """
        road_ids = set()
        for wp in self._map.generate_waypoints(20.0):
            if (
                wp.lane_type == carla.LaneType.Driving
                and not wp.is_junction
            ):
                # Town04 highway has speed limits of 90 km/h
                road_ids.add(wp.road_id)
        return road_ids

    def _compute_reward(self, steering, throttle):
        if self.vehicle is None:
            return 0.0, None

        vel = self.vehicle.get_velocity()
        spd_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        veh_loc = self.vehicle.get_location()
        veh_trans = self.vehicle.get_transform()

        # 1. Advance Reward: cap speed at target speed (e.g. 40.0)
        curr_wp = self._map.get_waypoint(veh_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if not curr_wp:
            return -100.0, "OFF-ROAD"

        # Calculate alignment (cos alpha) using forward vectors
        v_fwd = veh_trans.get_forward_vector()
        w_fwd = curr_wp.transform.get_forward_vector()
        # Dot product of normalized vectors
        cos_alpha = v_fwd.x * w_fwd.x + v_fwd.y * w_fwd.y + v_fwd.z * w_fwd.z
        cos_alpha = max(0.0, cos_alpha) # Ignore backward alignment
        
        advance_reward = (min(spd_kmh, self.target_speed) / self.target_speed) * cos_alpha
        
        # Penalize speeding
        speed_penalty = max(0.0, spd_kmh - self.target_speed) * 0.2

        # 2. Centering Bonus: +0.5 if lateral_dist < 0.3m
        lat_dist = math.sqrt((veh_loc.x - curr_wp.transform.location.x)**2 + 
                             (veh_loc.y - curr_wp.transform.location.y)**2)
        centering_bonus = 0.5 if lat_dist < 0.3 else 0.0

        # 3. Quadratic Deviation Penalty: Increased strictly
        deviation_penalty = (lat_dist ** 2) * 1.5

        # 4. Smoothness & Jerk Penalty
        steer_penalty = abs(steering) * 0.3
        jerk_penalty = abs(steering - getattr(self, '_prev_steer', 0.0)) * 1.5

        # 5. Slow Penalty: incremental si va a menos de 5 km/h
        if spd_kmh < 5.0:
            self._slow_steps += 1
        else:
            self._slow_steps = 0
        
        slow_penalty = self._slow_steps * 0.05

        reward = advance_reward + centering_bonus - deviation_penalty - steer_penalty - jerk_penalty - speed_penalty - slow_penalty

        # --- Termination Logic ---
        if self._slow_steps >= 250:
            return -50.0, "TOO SLOW"

        # Salida de carretera (dist > 1.5m)
        if lat_dist > 1.5:
            return -100.0, "ROAD EXIT"

        # Lane change / Wrong road
        if curr_wp.road_id != self.start_road_id:
            return -100.0, "LEFT HIGHWAY"
        
        if curr_wp.lane_id != self.start_lane_id:
            return -100.0, "LANE CHANGE"

        return float(reward), None

    # --------------------------------------------------------------------------
    # Sensor attachment
    # --------------------------------------------------------------------------

    def _attach_semantic_camera(self):
        """Onboard semantic segmentation camera mounted at the roof mirror position."""
        bp = self.bp_lib.find("sensor.camera.semantic_segmentation")
        bp.set_attribute("image_size_x", "400")
        bp.set_attribute("image_size_y", "300")
        bp.set_attribute("fov", "90")
        # Moving forward to x=1.0 and z=1.3 (clear view from rear-view mirror area)
        tf = carla.Transform(
            carla.Location(x=1.4, z=1.7),
            carla.Rotation(pitch=-7)
        )
        self.cam_sem = self.world.spawn_actor(bp, tf, attach_to=self.vehicle)

        def _on_sem(image):
            if not self.vehicle or getattr(self, '_block_sensors', False): return # Defensive
            raw = np.frombuffer(image.raw_data, dtype=np.uint8).copy()
            self._sem_obs = decode_semantic(raw, image.height, image.width)
            
            # Semantic tags are in the Red channel (index 2 in BGRA)
            arr = raw.reshape((image.height, image.width, 4))
            tags = arr[:, :, 2]
            
            # Palette mapping for display (RGB)
            PALETTE = {
                24: (255, 255,  50),  # RoadLines (Yellow)
                1:  ( 50, 200,  50),  # Roads (Green)
                2:  ( 60, 100, 255), 5: ( 60, 100, 255), 28: ( 60, 100, 255), # Borders (Blue)
                12: (255,  60,  60), 13: (255,  60,  60), # Dynamics (Red)
                14: (255,  60,  60), 15: (255,  60,  60), 16: (255,  60,  60), 
                18: (255,  60,  60), 19: (255,  60,  60),
            }
            
            # Re-use pre-allocated buffer for the color mapping
            self._sem_vis_tmp.fill(0)
            for tid, color in PALETTE.items():
                self._sem_vis_tmp[tags == tid] = color
            
            self._sem_raw = self._sem_vis_tmp.copy()
            self._sem_event.set()

            if not self._printed_ids:
                print(f"[DEBUG] Unique Semantic IDs found in frame: {np.unique(tags)}")
                self._printed_ids = True

        self.cam_sem.listen(_on_sem)

    def _attach_rgb_camera(self):
        """Third-person RGB camera for Pygame visualization."""
        bp = self.bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", "800")
        bp.set_attribute("image_size_y", "600")
        bp.set_attribute("fov", "90")
        tf = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-15))
        self.cam_rgb = self.world.spawn_actor(bp, tf, attach_to=self.vehicle)

        def _on_rgb(image):
            if not self.vehicle or getattr(self, '_block_sensors', False): return # Defensive
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).copy()
            arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
            self._rgb_raw = arr[:, :, ::-1] # Save as RGB numpy array

        self.cam_rgb.listen(_on_rgb)

    def _attach_collision_sensor(self):
        bp = self.bp_lib.find("sensor.other.collision")
        self.col_sensor = self.world.spawn_actor(
            bp, carla.Transform(), attach_to=self.vehicle
        )

        def _on_collision(event):
            self._collision_flag = True

        self.col_sensor.listen(_on_collision)

    def _attach_lane_invasion_sensor(self):
        bp = self.bp_lib.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(
            bp, carla.Transform(), attach_to=self.vehicle
        )

        def _on_invasion(event):
            # Only penalise solid/broken line crossings (not road edges)
            critical_types = {
                carla.LaneMarkingType.Solid,
                carla.LaneMarkingType.SolidSolid,
                carla.LaneMarkingType.SolidBroken,
                carla.LaneMarkingType.BrokenSolid,
            }
            for marking in event.crossed_lane_markings:
                if marking.type in critical_types:
                    self._invasion_flag = True
                    break

        self.lane_sensor.listen(_on_invasion)

    def _cleanup_actors(self):
        """Destroy all spawned actors safely."""
        actors = [self.lane_sensor, self.col_sensor, self.cam_rgb, self.cam_sem, self.vehicle]
        for actor in actors:
            try:
                if actor is not None and actor.is_alive:
                    actor.stop() if hasattr(actor, "stop") else None
                    actor.destroy()
            except Exception:
                pass
        self.vehicle = self.cam_sem = self.cam_rgb = self.col_sensor = self.lane_sensor = None
