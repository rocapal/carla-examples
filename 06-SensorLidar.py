import argparse
import random
import carla
import pygame
import numpy as np

# --- Configuration Parameters for LiDAR (Customizable) ---
LIDAR_CHANNELS = '16'
LIDAR_RANGE = '50.0'         # meters
LIDAR_POINTS_PER_SEC = '100000'
LIDAR_ROTATION_FREQ = '60.0'
LIDAR_UPPER_FOV = '10.0'
LIDAR_LOWER_FOV = '-30.0'
# ---------------------------------------------------------

def render_image(image, display, offset):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, offset)

def render_lidar(lidar_measurement, display, offset, lidar_range):
    points = np.frombuffer(lidar_measurement.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))

    # Pygame surface dims (right pane)
    w, h = 800, 600
    lidar_img = np.zeros((w, h, 3), dtype=np.uint8)
    
    # Draw ego-vehicle center as a red dot
    lidar_img[w//2 - 2:w//2 + 2, h//2 - 4:h//2 + 4] = (255, 0, 0) 

    scale = min(w, h) / (2.0 * float(lidar_range))

    x = points[:, 0]
    y = points[:, 1]
    
    # Map points to 2D grid
    u = np.array(w/2 + y * scale, dtype=np.int32)
    v = np.array(h/2 - x * scale, dtype=np.int32)

    # Filter out bounds
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    
    # Color active points green
    lidar_img[u[valid], v[valid]] = (0, 255, 0)
    
    surface = pygame.surfarray.make_surface(lidar_img)
    display.blit(surface, offset)

def main():
    parser = argparse.ArgumentParser(description="CARLA Sensor & Lidar")
    parser.add_argument('--host', default='127.0.0.1', help='IP of the host server')
    parser.add_argument('--port', default=2000, type=int, help='TCP port to listen to')
    args = parser.parse_args()

    pygame.init()
    width, height = 800, 600
    # 3 columns layout: 1st (3rd person), 2nd (RGB / Semantic), 3rd (LiDAR)
    display = pygame.display.set_mode((width * 3, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA - Left: 3rd Person | Center: RGB/Semantic | Right: LiDAR")

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:
        available_maps = client.get_available_maps()
        selected_map = random.choice(available_maps)
        print(f"Loading random map: {selected_map}")
        world = client.load_world(selected_map)

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.tesla.model3'))
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        print(f"Spawning vehicle: {vehicle_bp.id}")
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True)

        # 1. Third person RGB
        cam_bp_3rd = blueprint_library.find('sensor.camera.rgb')
        cam_bp_3rd.set_attribute('image_size_x', str(width))
        cam_bp_3rd.set_attribute('image_size_y', str(height))
        cam_bp_3rd.set_attribute('fov', '90')
        third_person_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15.0))
        camera_third_person = world.spawn_actor(cam_bp_3rd, third_person_transform, attach_to=vehicle)

        # 2. Onboard RGB (Top center)
        cam_bp_rgb = blueprint_library.find('sensor.camera.rgb')
        cam_bp_rgb.set_attribute('image_size_x', str(width))
        cam_bp_rgb.set_attribute('image_size_y', str(height // 2))
        cam_bp_rgb.set_attribute('fov', '90')
        onboard_transform = carla.Transform(carla.Location(x=1.5, z=1.4), carla.Rotation(pitch=0.0))
        camera_onboard_rgb = world.spawn_actor(cam_bp_rgb, onboard_transform, attach_to=vehicle)

        # 3. Onboard Semantic Segmentation (Bottom center)
        cam_bp_sem = blueprint_library.find('sensor.camera.semantic_segmentation')
        cam_bp_sem.set_attribute('image_size_x', str(width))
        cam_bp_sem.set_attribute('image_size_y', str(height // 2))
        cam_bp_sem.set_attribute('fov', '90')
        camera_onboard_sem = world.spawn_actor(cam_bp_sem, onboard_transform, attach_to=vehicle)

        # 4. LiDAR (Right)
        cam_bp_lidar = blueprint_library.find('sensor.lidar.ray_cast')
        cam_bp_lidar.set_attribute('channels', LIDAR_CHANNELS)
        cam_bp_lidar.set_attribute('range', LIDAR_RANGE)
        cam_bp_lidar.set_attribute('points_per_second', LIDAR_POINTS_PER_SEC)
        cam_bp_lidar.set_attribute('rotation_frequency', LIDAR_ROTATION_FREQ)
        cam_bp_lidar.set_attribute('upper_fov', LIDAR_UPPER_FOV)
        cam_bp_lidar.set_attribute('lower_fov', LIDAR_LOWER_FOV)
        
        # Position LiDAR on the roof
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=3))
        camera_lidar = world.spawn_actor(cam_bp_lidar, lidar_transform, attach_to=vehicle)

        sensors_data = {'third_person': None, 'onboard_rgb': None, 'onboard_sem': None, 'lidar': None}
        
        camera_third_person.listen(lambda image: sensors_data.update({'third_person': image}))
        camera_onboard_rgb.listen(lambda image: sensors_data.update({'onboard_rgb': image}))
        camera_onboard_sem.listen(lambda image: image.convert(carla.ColorConverter.CityScapesPalette) or sensors_data.update({'onboard_sem': image}))
        camera_lidar.listen(lambda data: sensors_data.update({'lidar': data}))

        print("Simulation running with Autopilot, Cameras, and LiDAR. Press ESC to exit.")
        clock = pygame.time.Clock()

        while True:
            clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE):
                    return

            if sensors_data['third_person'] is not None:
                render_image(sensors_data['third_person'], display, (0, 0))
                
            if sensors_data['onboard_rgb'] is not None:
                render_image(sensors_data['onboard_rgb'], display, (width, 0))
                
            if sensors_data['onboard_sem'] is not None:
                render_image(sensors_data['onboard_sem'], display, (width, height // 2))
                
            if sensors_data['lidar'] is not None:
                render_lidar(sensors_data['lidar'], display, (width * 2, 0), float(LIDAR_RANGE))

            pygame.display.flip()

    finally:
        print("Cleaning up actors...")
        if 'camera_third_person' in locals():
            camera_third_person.destroy()
        if 'camera_onboard_rgb' in locals():
            camera_onboard_rgb.destroy()
        if 'camera_onboard_sem' in locals():
            camera_onboard_sem.destroy()
        if 'camera_lidar' in locals():
            camera_lidar.destroy()
        if 'vehicle' in locals():
            vehicle.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()
