import argparse
import random
import sys
import time
import math
import numpy as np
import pygame
import carla
from collections import deque
from DatasetRecorder import DatasetRecorder

# ==============================================================================
# -- PID Controller -----------------------------------------------------------
# ==============================================================================

class PIDLongitudinalController:
    def __init__(self, vehicle, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = []

    def run_step(self, target_speed):
        current_speed = self._get_forward_speed()
        error = target_speed - current_speed
        
        self._error_buffer.append(error)
        if len(self._error_buffer) > 20:
            self._error_buffer.pop(0)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_i * _ie) + (self._k_d * _de), -1.0, 1.0)

    def _get_forward_speed(self):
        velocity = self._vehicle.get_velocity()
        return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # km/h

class PIDLateralController:
    def __init__(self, vehicle, K_P=1.95, K_I=0.0, K_D=0.2, dt=0.03):
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._e_buffer = []

    def run_step(self, waypoint):
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def _pid_control(self, waypoint, vehicle_transform):
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint.transform.location.x - v_begin.x,
                          waypoint.transform.location.y - v_begin.y, 0.0])

        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec) + 1e-6), -1.0, 1.0))
        
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) > 20: 
            self._e_buffer.pop(0)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * _dot) + (self._k_i * _ie) + (self._k_d * _de), -1.0, 1.0)


class VehiclePIDController:
    """
    PID Controller for vehicle lateral and longitudinal control.
    """
    def __init__(self, vehicle, args_lateral=None, args_longitudinal=None):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        
        if not args_lateral:
            args_lateral = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': 1.0/60.0}
        if not args_longitudinal:
            args_longitudinal = {'K_P': 0.20, 'K_I': 0.05, 'K_D': 0.0, 'dt': 1.0/60.0}
            
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lat_controller = PIDLateralController(self._vehicle, **args_lateral)

    def run_step(self, target_speed, waypoint):
        """
        Execute one step of control.
        """
        acceleration = self._lon_controller.run_step(target_speed)
        current_steering = self._lat_controller.run_step(waypoint)
        
        control = carla.VehicleControl()
        
        if target_speed < 0.1:
            # Completely stopped state override
            control.throttle = 0.0
            control.brake = 1.0
        else:
            if acceleration >= 0.05:
                control.throttle = min(acceleration, 1.0)
                control.brake = 0.0
            else:
                control.throttle = 0.0
                # Simulate human driving: do not tap brakes for minor speed corrections (coasting)
                if acceleration < -0.15:
                    control.brake = min(abs(acceleration), 1.0)
                else:
                    control.brake = 0.0

        # Steering smoothing
        control.steer = current_steering
        return control


# ==============================================================================
# -- Safety Agent -------------------------------------------------------------
# ==============================================================================
class SafetyAgent:
    def __init__(self, vehicle):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self.last_cleared_stop_id = None
        # self.traffic_light_stop_time = None  # [DISABLED] Traffic light ignored



    def get_obstacle_clearance(self, current_speed, route):
        """
        Returns a tuple: (safety_speed_multiplier [0.0 - 1.0], obstacle_reason string)
        """
        vehicles = self._world.get_actors().filter('vehicle.*')
        walkers = self._world.get_actors().filter('walker.*')
        
        all_obstacles = list(vehicles) + list(walkers)
        
        ego_tf = self._vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_fwd = ego_tf.get_forward_vector()
        ego_wp = self._map.get_waypoint(ego_loc)
        
        min_factor = 1.0
        min_reason = None

        for actor in all_obstacles:
            if actor.id == self._vehicle.id:
                continue
                
            loc = actor.get_location()
            distance = ego_loc.distance(loc)
            
            # Check obstacles within 25 meters radius
            if distance < 25.0:
                diff_vec = carla.Vector3D(loc.x - ego_loc.x, loc.y - ego_loc.y, 0)
                diff_length = math.sqrt(diff_vec.x**2 + diff_vec.y**2)
                
                if diff_length < 0.1:
                    continue  

                dot = (diff_vec.x * ego_fwd.x + diff_vec.y * ego_fwd.y) / diff_length
                actor_type = "Vehicle" if 'vehicle' in actor.type_id else "Walker"

                # 1. Immediate collision hazard (too close and somewhat in front)
                if distance < 5.0 and dot > 0.5:
                    return 0.0, actor_type

                # 2. Path projection hazard (Checks if actor blocks any of our actual future waypoints)
                on_path = False
                closest_wp_dist = 999.0
                
                for wp in route:
                    # Bounding box matches roughly a 2.5m radius from our perfect path
                    if wp.transform.location.distance(loc) < 2.5:
                        on_path = True
                        closest_wp_dist = ego_loc.distance(wp.transform.location)
                        break
                        
                if on_path:
                    if closest_wp_dist < 8.0:
                        return 0.0, actor_type 
                    
                    # Gradient brake based on distance (starts slowing down at 25m)
                    factor = (closest_wp_dist - 8.0) / 17.0
                    if factor < min_factor:
                        min_factor = factor
                        min_reason = actor_type

        # [DISABLED] Traffic light check - agent ignores all traffic lights
        # if self._vehicle.is_at_traffic_light():
        #     state = self._vehicle.get_traffic_light_state()
        #     if state in [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow]:
        #         tl = self._vehicle.get_traffic_light()
        #         if tl is not None:
        #             if current_speed < 5.0:
        #                 if self.traffic_light_stop_time is None:
        #                     print(f"[DEBUG] Red traffic light detected (ID: {tl.id}). Waiting 2 seconds...")
        #                     self.traffic_light_stop_time = time.time()
        #                 elif time.time() - self.traffic_light_stop_time > 2.0:
        #                     print(f"[DEBUG] 2s passed. Forcing traffic light (ID: {tl.id}) to GREEN.")
        #                     tl.set_state(carla.TrafficLightState.Green)
        #                     tl.set_green_time(20.0)
        #                     self.traffic_light_stop_time = None
        #         return 0.0, "TrafficLight"
        #     else:
        #         self.traffic_light_stop_time = None
        # else:
        #     self.traffic_light_stop_time = None

        # Stop sign parsing via landmarks
        landmarks = ego_wp.get_landmarks(distance=10.0, stop_at_junction=False)
        for lm in landmarks:
            if not lm.is_dynamic and ('stop' in lm.name.lower() or lm.type in ['122', '206']):
                if lm.id != self.last_cleared_stop_id:
                    if current_speed < 0.1:
                        self.last_cleared_stop_id = lm.id
                    else:
                        return 0.0, "StopSign"

        min_factor = np.clip(min_factor, 0.0, 1.0)
        # Snap to 0 if the factor is too small to prevent constant crawling/jitter
        if min_factor < 0.15:
            return 0.0, min_reason
            
        return min_factor, min_reason

# ==============================================================================
# -- HUD & Rendering ----------------------------------------------------------
# ==============================================================================
def render_image(image, display, offset=(0, 0)):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, offset)

def build_intrinsic_matrix(w, h, fov):
    focal = w / (2.0 * math.tan(fov * math.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_world_to_camera_matrix(transform):
    pitch = math.radians(transform.rotation.pitch)
    yaw = math.radians(transform.rotation.yaw)
    roll = math.radians(transform.rotation.roll)
    c_y = math.cos(yaw); s_y = math.sin(yaw)
    c_r = math.cos(roll); s_r = math.sin(roll)
    c_p = math.cos(pitch); s_p = math.sin(pitch)
    
    matrix = np.identity(4)
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[0, 3] = transform.location.x
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[1, 3] = transform.location.y
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    matrix[2, 3] = transform.location.z
    
    return np.linalg.inv(matrix)

def project_3d_to_2d(p3d, K, world_to_camera):
    p_world = np.array([[p3d.x], [p3d.y], [p3d.z], [1.0]])
    p_cam = np.dot(world_to_camera, p_world)
    p_cam_std = np.array([p_cam[1, 0], -p_cam[2, 0], p_cam[0, 0]])
    if p_cam_std[2] <= 0.0: return None
    p_img = np.dot(K, p_cam_std)
    # Cast safely to integers representing pixel coordinates
    return (int(p_img[0] / p_img[2]), int(p_img[1] / p_img[2]))

def draw_hud(display, target_speed, current_speed, safety_factor, control):
    font = pygame.font.SysFont(None, 36)
    hud_surface = pygame.Surface((360, 200), pygame.SRCALPHA)
    hud_surface.fill((0, 0, 0, 180))
    
    t1 = font.render(f"Target Spd : {target_speed:.1f} km/h", True, (255, 255, 255))
    t2 = font.render(f"Actual Spd : {current_speed:.1f} km/h", True, (100, 255, 100))
    
    color = (255, 50, 50) if safety_factor < 0.1 else (255, 255, 50) if safety_factor < 0.8 else (50, 255, 50)
    status_text = "CLEAR" if safety_factor == 1.0 else "BRAKING (OBSTACLE/RED LIGHT)" if safety_factor > 0 else "STOPPED"
    t3 = font.render(f"Env Status : {status_text}", True, color)
    t4 = font.render(f"Gas: {control.throttle:.2f} | Brake: {control.brake:.2f}", True, (200, 200, 200))
    
    hud_surface.blit(t1, (15, 15))
    hud_surface.blit(t2, (15, 55))
    hud_surface.blit(t3, (15, 95))
    hud_surface.blit(t4, (15, 145))
    
    display.blit(hud_surface, (20, 20))


# ==============================================================================
# -- Main Loop ----------------------------------------------------------------
# ==============================================================================
# --- Traffic Parameters (Customizable) ---
NUM_CARS = 20
NUM_BIKES_MOTOS = 5
NUM_WALKERS = 10
# -----------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CARLA Manual PID Driving")
    parser.add_argument('--host', default='127.0.0.1', help='IP of the host server')
    parser.add_argument('--port', default=2000, type=int, help='TCP port to listen to')
    parser.add_argument('--traffic', action='store_true', help='Enable random traffic (vehicles and pedestrians)')
    parser.add_argument('--dataset', type=str, default=None, help='Directorio base donde exportar el dataset a 10Hz')
    args = parser.parse_args()

    pygame.init()
    width, height = 1200, 600
    display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA - Autonomous PID Driving")

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:
        print("Loading Town04 and setting weather to CloudyNoon ")
        world = client.load_world('Town04')
        world.set_weather(carla.WeatherParameters.CloudyNoon)
        c_map = world.get_map()
        
        blueprint_library = world.get_blueprint_library()
        all_vehicle_bps = blueprint_library.filter('vehicle.*')
        car_bps = [bp for bp in all_vehicle_bps if int(bp.get_attribute('number_of_wheels').as_int()) >= 4]
        bike_bps = [bp for bp in all_vehicle_bps if int(bp.get_attribute('number_of_wheels').as_int()) == 2]

        vehicle_bp = random.choice(blueprint_library.filter('vehicle.tesla.model3'))
        
        spawn_points = c_map.get_spawn_points()
        
        opt_spawn_points = list(range(230, 236)) + list(range(245, 251))

        spawn_index = random.choice(opt_spawn_points)
        print(f"Spawn index: {spawn_index}")
        spawn_point = spawn_points[spawn_index]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        vehicle.set_autopilot(False)

        traffic_vehicles = []
        traffic_walkers = []
        traffic_walker_controllers = []
        
        # Always spawn a CAR 6m ahead (center-to-center) to test obstacle avoidance
        ego_tf = vehicle.get_transform()
        fwd = ego_tf.get_forward_vector()
        obstacle_loc = ego_tf.location + carla.Location(x=fwd.x * 6.0, y=fwd.y * 6.0, z=0.2)
        obstacle_tf = carla.Transform(obstacle_loc, ego_tf.rotation)
        
        obs_bp = random.choice(car_bps)
        if obs_bp.has_attribute('color'):
            obs_bp.set_attribute('color', '255,0,0') # Red color
            
        print(f"Spawning obstacle car ahead: {obs_bp.id}")
        obstacle_vehicle = world.try_spawn_actor(obs_bp, obstacle_tf)
        if obstacle_vehicle is not None:
            obstacle_vehicle.set_autopilot(True)
            traffic_vehicles.append(obstacle_vehicle)
        
        if args.traffic:
            print(f"Spawning {NUM_CARS} cars and {NUM_BIKES_MOTOS} bikes/motos...")
            
            sp = c_map.get_spawn_points()
            random.shuffle(sp)
            
            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            
            spawn_index = 0
            
            # Spawn Cars
            for _ in range(NUM_CARS):
                if spawn_index >= len(sp): break
                if sp[spawn_index].location.distance(spawn_point.location) > 5.0:
                    bp = random.choice(car_bps)
                    if bp.has_attribute('color'):
                        color = random.choice(bp.get_attribute('color').recommended_values)
                        bp.set_attribute('color', color)
                    npc = world.try_spawn_actor(bp, sp[spawn_index])
                    if npc is not None:
                        npc.set_autopilot(True, traffic_manager.get_port())
                        traffic_vehicles.append(npc)
                spawn_index += 1
                
            # Spawn Bikes/Motos
            for _ in range(NUM_BIKES_MOTOS):
                if spawn_index >= len(sp): break
                if sp[spawn_index].location.distance(spawn_point.location) > 5.0:
                    bp = random.choice(bike_bps)
                    if bp.has_attribute('color'):
                        color = random.choice(bp.get_attribute('color').recommended_values)
                        bp.set_attribute('color', color)
                    npc = world.try_spawn_actor(bp, sp[spawn_index])
                    if npc is not None:
                        npc.set_autopilot(True, traffic_manager.get_port())
                        traffic_vehicles.append(npc)
                spawn_index += 1
            
            print(f"Spawning {NUM_WALKERS} pedestrians...")
            walker_bps = blueprint_library.filter('walker.pedestrian.*')
            walker_controller_bp = blueprint_library.find('controller.ai.walker')
            
            for i in range(NUM_WALKERS):
                loc = world.get_random_location_from_navigation()
                if loc is None: continue
                spawn_loc = carla.Transform(loc)
                bp = random.choice(walker_bps)
                npc = world.try_spawn_actor(bp, spawn_loc)
                if npc is not None:
                    traffic_walkers.append(npc)
                    controller = world.try_spawn_actor(walker_controller_bp, carla.Transform(), npc)
                    if controller is not None:
                        traffic_walker_controllers.append(controller)
                        controller.start()
                        controller.go_to_location(world.get_random_location_from_navigation())
                        controller.set_max_speed(1 + random.random())

        # State Dictionary for async callbacks
        scenario_state = {'needs_restart': False, 'reason': 'none'}
        restart_reason = 'none'
        
        # Collision Sensor
        col_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
        
        def collision_handler(event):
            print(f"COLLISION DETECTED with {event.other_actor.type_id}! Scheduling restart...")
            scenario_state['needs_restart'] = True
            scenario_state['reason'] = 'collision'
            
        collision_sensor.listen(collision_handler)

        # 1. 3rd Person Camera (Left side)
        cam_3rd_bp = blueprint_library.find('sensor.camera.rgb')
        cam_3rd_bp.set_attribute('image_size_x', '800')
        cam_3rd_bp.set_attribute('image_size_y', '600')
        cam_3rd_bp.set_attribute('fov', '90')
        cam_3rd_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15.0))
        camera_3rd = world.spawn_actor(cam_3rd_bp, cam_3rd_transform, attach_to=vehicle)

        # 2. Onboard RGB (Top Right)
        cam_rgb_bp = blueprint_library.find('sensor.camera.rgb')
        cam_rgb_bp.set_attribute('image_size_x', '400')
        cam_rgb_bp.set_attribute('image_size_y', '300')
        cam_rgb_bp.set_attribute('fov', '90')
        onboard_transform = carla.Transform(carla.Location(x=1.5, z=1.6), carla.Rotation(pitch=0.0))
        camera_rgb = world.spawn_actor(cam_rgb_bp, onboard_transform, attach_to=vehicle)

        # 3. Onboard Semantic (Bottom Right)
        cam_sem_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        cam_sem_bp.set_attribute('image_size_x', '400')
        cam_sem_bp.set_attribute('image_size_y', '300')
        cam_sem_bp.set_attribute('fov', '90')
        camera_sem = world.spawn_actor(cam_sem_bp, onboard_transform, attach_to=vehicle)

        image_data = {'3rd': None, 'rgb': None, 'sem': None, 'sem_raw': None}
        
        camera_3rd.listen(lambda image: image_data.update({'3rd': image}))
        camera_rgb.listen(lambda image: image_data.update({'rgb': image}))
        
        def process_semantic(image):
            # Extract raw 8-bit map for the Dataset (CARLA encodes Class ID inside the Red channel of the raw byte buffer)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            class_ids = array[:, :, 2].copy() # (300, 400) uint8 array of Class IDs
            image_data['sem_raw'] = class_ids
            
            # Convert the original object to CityScapes for human Pygame HUD visually
            image.convert(carla.ColorConverter.CityScapesPalette)
            image_data['sem'] = image
            
        camera_sem.listen(process_semantic)

        dt = 1.0 / 60.0
        controller = VehiclePIDController(vehicle, 
                                          args_lateral={'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': dt},
                                          args_longitudinal={'K_P': 0.20, 'K_I': 0.05, 'K_D': 0.0, 'dt': dt})
        safety_agent = SafetyAgent(vehicle)

        # Waypoint tracking queue
        route = deque()
        current_wp = c_map.get_waypoint(vehicle.get_location())
        route.append(current_wp)
        
        # Build initial lookahead route (e.g. 10 waypoints, 2 meters apart)
        for _ in range(10):
            next_wps = route[-1].next(2.0)
            if not next_wps: break
            route.append(random.choice(next_wps))

        dataset_recorder = None
        if args.dataset:
            dataset_recorder = DatasetRecorder(args.dataset, capture_hz=10.0)

        print("\nStarting Automated PID Driving Loop in Town03.")
        print("Weather: ClearNoon. Using ground-truth collision detection.")
        print("Press ESC to exit.")
        clock = pygame.time.Clock()
        
        control = carla.VehicleControl()

        while True:
            clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE):
                    return

            if scenario_state['needs_restart']:
                needs_restart = True
                restart_reason = scenario_state['reason']
                break

            car_loc = vehicle.get_location()

            # Remove waypoints we've passed or are very close to
            while len(route) > 0 and car_loc.distance(route[0].transform.location) < 2.5:
                route.popleft()
                
            # Replenish route to keep ~10 waypoints ahead
            while len(route) < 10:
                next_wps = route[-1].next(2.0)
                if not next_wps: break
                
                options = list(next_wps)
                # If we are in an intersection/roundabout, occasionally consider the right lane 
                # to progressively drift out and prevent infinite loops in roundabout inner lanes.
                if route[-1].is_junction:
                    right_lane = route[-1].get_right_lane()
                    if right_lane and right_lane.lane_type == carla.LaneType.Driving:
                        rl_next = right_lane.next(2.0)
                        if rl_next:
                            options.append(rl_next[0])

                route.append(random.choice(options))

            # Check if we are far off route (> 4.5m)
            if len(route) > 0:
                min_dist_to_route = min([car_loc.distance(wp.transform.location) for wp in route])
                if min_dist_to_route > 4.5:
                    if 'off_route_start_time' not in locals():
                        off_route_start_time = time.time()
                    elif time.time() - off_route_start_time > 3.0:
                        print("Vehicle off trajectory > 3s. Restarting scenario...")
                        needs_restart = True
                        restart_reason = 'off_route'
                        break
                else:
                    if 'off_route_start_time' in locals():
                        del off_route_start_time

            # The target for the PID is the waypoint roughly 4-6 meters ahead
            target_waypoint = route[min(2, len(route)-1)] if len(route) > 0 else current_wp

            # Draw trajectory in CARLA (This will be visible on Pygame screen)
            # CARLA debug lines render in the physical 3D world, meaning they contaminate all 
            # attached cameras (RGB, Semantic). To keep the dataset sensors clean, it is disabled.
            # for i in range(3, len(route) - 1):
            #     p1 = route[i].transform.location + carla.Location(z=0.2)
            #     p2 = route[i+1].transform.location + carla.Location(z=0.2)
            #     world.debug.draw_line(p1, p2, thickness=0.08, color=carla.Color(0, 100, 255), life_time=0.05)

            v = vehicle.get_velocity()
            current_spd = math.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6
            
            # Query environment limits
            speed_limit = vehicle.get_speed_limit()  # typically returns 30+ km/h
            if speed_limit <= 0.0:
                speed_limit = 30.0

            # Safety adjustments (Traffic lights & Obstacles)
            safety_factor, obstacle_reason = safety_agent.get_obstacle_clearance(current_spd, route)
            desired_speed = speed_limit * safety_factor
            
            # --- TIMEOUT RESTART LOGIC ---
            # Verify absolute blockage (Car ahead and Ego vehicle almost stopped)
            if safety_factor < 0.1 and obstacle_reason == "Vehicle" and current_spd < 2.0:
                if 'stuck_start_time' not in locals():
                    stuck_start_time = time.time()
                
                stuck_duration = time.time() - stuck_start_time
                                
                if stuck_duration > 60.0:
                    print("Stuck for over 60 seconds without a safe gap. Restarting scenario...")
                    needs_restart = True
                    restart_reason = 'stuck'
                    break
            else:
                if 'stuck_start_time' in locals():
                    del stuck_start_time
            # ---------------------------

            # Calculate control using custom PID
            control = controller.run_step(desired_speed, target_waypoint)
            vehicle.apply_control(control)

            # Render Screens
            if image_data['3rd'] is not None:
                render_image(image_data['3rd'], display, offset=(0, 0))
                
                # --- 2D Trajectory Projection over Pygame ---
                K = build_intrinsic_matrix(800, 600, 90.0)
                world_to_cam = get_world_to_camera_matrix(camera_3rd.get_transform())
                points_2d = []
                for i in range(1, len(route)):
                    loc = route[i].transform.location + carla.Location(z=0.2)
                    pt = project_3d_to_2d(loc, K, world_to_cam)
                    if pt: points_2d.append(pt)
                
                if len(points_2d) > 1:
                    pygame.draw.lines(display, (0, 100, 255), False, points_2d, 4)
                # ---------------------------------------------
                
            if image_data['rgb'] is not None:
                render_image(image_data['rgb'], display, offset=(800, 0))
            if image_data['sem'] is not None:
                render_image(image_data['sem'], display, offset=(800, 300))

            # Draw HUD
            draw_hud(display, desired_speed, current_spd, safety_factor, control)

            pygame.display.flip()

            # --- DATASET RECORDING LOOP ---
            if control.throttle > 0.01:
                if dataset_recorder:
                    dataset_recorder.resume()
                if 'stopped_start_time' in locals():
                    del stopped_start_time
            elif current_spd < 0.5:
                if 'stopped_start_time' not in locals():
                    stopped_start_time = time.time()
                elif time.time() - stopped_start_time > 0.5:
                    if dataset_recorder:
                        dataset_recorder.pause()
            else:
                if 'stopped_start_time' in locals():
                    del stopped_start_time

            if dataset_recorder and image_data['rgb'] is not None and image_data['sem_raw'] is not None:
                # Time.time() forces absolute epoch timestamps for strict inter-file exactitude.
                dataset_recorder.record(time.time(), 
                                        image_data['rgb'], 
                                        image_data['sem_raw'], 
                                        control.throttle, 
                                        control.steer, 
                                        control.brake, 
                                        current_spd)

    finally:
        print("Cleaning up...")
        if 'dataset_recorder' in locals() and dataset_recorder:
            if 'restart_reason' in locals() and restart_reason != 'none':
                dataset_recorder.delete_last_seconds(2.0)
            dataset_recorder.close()
        if 'collision_sensor' in locals(): collision_sensor.destroy()
        if 'camera_3rd' in locals(): camera_3rd.destroy()
        if 'camera_rgb' in locals(): camera_rgb.destroy()
        if 'camera_sem' in locals(): camera_sem.destroy()
        if 'vehicle' in locals():
            vehicle.destroy()
            
        print("Destroying traffic actors...")
        if 'traffic_walker_controllers' in locals():
            for c in traffic_walker_controllers:
                c.stop()
                c.destroy()
        if 'traffic_walkers' in locals():
            for w in traffic_walkers:
                w.destroy()
        if 'traffic_vehicles' in locals():
            for v in traffic_vehicles:
                if v is not None:
                    v.destroy()
                
        pygame.quit()
        if 'needs_restart' in locals() and needs_restart:
            import os
            os.execv(sys.executable, ['python'] + sys.argv)

if __name__ == '__main__':
    main()
