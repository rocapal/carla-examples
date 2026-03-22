import argparse
import random
import carla
import pygame
import numpy as np

def render_image(image, display, offset):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, offset)

def main():
    parser = argparse.ArgumentParser(description="CARLA Semantic Sensor & Autopilot")
    parser.add_argument('--host', default='127.0.0.1', help='IP of the host server')
    parser.add_argument('--port', default=2000, type=int, help='TCP port to listen to')
    args = parser.parse_args()

    pygame.init()
    width, height = 800, 600
    display = pygame.display.set_mode((width * 2, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA - Left: 3rd Person | Right Top: Onboard RGB | Right Bottom: Onboard Semantic")

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

        # 1. Third person RGB (Left side, full height)
        cam_bp_3rd = blueprint_library.find('sensor.camera.rgb')
        cam_bp_3rd.set_attribute('image_size_x', str(width))
        cam_bp_3rd.set_attribute('image_size_y', str(height))
        cam_bp_3rd.set_attribute('fov', '90')
        third_person_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15.0))
        camera_third_person = world.spawn_actor(cam_bp_3rd, third_person_transform, attach_to=vehicle)

        # 2. Onboard RGB (Right side, top half: 800x300)
        cam_bp_rgb = blueprint_library.find('sensor.camera.rgb')
        cam_bp_rgb.set_attribute('image_size_x', str(width))
        cam_bp_rgb.set_attribute('image_size_y', str(height // 2))
        cam_bp_rgb.set_attribute('fov', '90')
        onboard_transform = carla.Transform(carla.Location(x=1.5, z=1.4), carla.Rotation(pitch=0.0))
        camera_onboard_rgb = world.spawn_actor(cam_bp_rgb, onboard_transform, attach_to=vehicle)

        # 3. Onboard Semantic Segmentation (Right side, bottom half: 800x300)
        cam_bp_sem = blueprint_library.find('sensor.camera.semantic_segmentation')
        cam_bp_sem.set_attribute('image_size_x', str(width))
        cam_bp_sem.set_attribute('image_size_y', str(height // 2))
        cam_bp_sem.set_attribute('fov', '90')
        camera_onboard_sem = world.spawn_actor(cam_bp_sem, onboard_transform, attach_to=vehicle)

        images = {'third_person': None, 'onboard_rgb': None, 'onboard_sem': None}
        
        camera_third_person.listen(lambda image: images.update({'third_person': image}))
        camera_onboard_rgb.listen(lambda image: images.update({'onboard_rgb': image}))
        
        # For semantic camera, we must convert it using CityScapesPalette before displaying
        camera_onboard_sem.listen(lambda image: image.convert(carla.ColorConverter.CityScapesPalette) or images.update({'onboard_sem': image}))

        print("Simulation running with Autopilot and 3 Cameras. Press ESC to exit.")
        clock = pygame.time.Clock()

        while True:
            clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE):
                    return

            if images['third_person'] is not None:
                render_image(images['third_person'], display, (0, 0))
                
            if images['onboard_rgb'] is not None:
                render_image(images['onboard_rgb'], display, (width, 0))
                
            if images['onboard_sem'] is not None:
                render_image(images['onboard_sem'], display, (width, height // 2))

            pygame.display.flip()

    finally:
        print("Cleaning up actors...")
        if 'camera_third_person' in locals():
            camera_third_person.destroy()
        if 'camera_onboard_rgb' in locals():
            camera_onboard_rgb.destroy()
        if 'camera_onboard_sem' in locals():
            camera_onboard_sem.destroy()
        if 'vehicle' in locals():
            vehicle.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()
