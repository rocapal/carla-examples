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
    parser = argparse.ArgumentParser(description="CARLA Autopilot")
    parser.add_argument('--host', default='127.0.0.1', help='IP of the host server')
    parser.add_argument('--port', default=2000, type=int, help='TCP port to listen to')
    args = parser.parse_args()

    pygame.init()
    width, height = 800, 600
    display = pygame.display.set_mode((width * 2, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA Autopilot - Onboard & 3rd Person")

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:
        available_maps = client.get_available_maps()
        selected_map = random.choice(available_maps)
        print(f"Loading random map: {selected_map}")
        world = client.load_world(selected_map)

        blueprint_library = world.get_blueprint_library()
        #print(blueprint_library.filter('vehicle.tesla.*'))
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.tesla.model3'))
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        print(f"Spawning vehicle: {vehicle_bp.id}")
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True)

        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(width))
        cam_bp.set_attribute('image_size_y', str(height))
        cam_bp.set_attribute('fov', '90')

        onboard_transform = carla.Transform(carla.Location(x=1.5, z=1.4), carla.Rotation(pitch=0.0))
        camera_onboard = world.spawn_actor(cam_bp, onboard_transform, attach_to=vehicle)

        third_person_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15.0))
        camera_third_person = world.spawn_actor(cam_bp, third_person_transform, attach_to=vehicle)

        images = {'onboard': None, 'third_person': None}
        camera_onboard.listen(lambda image: images.update({'onboard': image}))
        camera_third_person.listen(lambda image: images.update({'third_person': image}))

        print("Simulation running. Autopilot Enabled. Press ESC to exit.")
        clock = pygame.time.Clock()

        while True:
            clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE):
                    return

            if images['onboard'] is not None:
                render_image(images['onboard'], display, (0, 0))
            if images['third_person'] is not None:
                render_image(images['third_person'], display, (width, 0))

            pygame.display.flip()

    finally:
        print("Cleaning up actors...")
        if 'camera_onboard' in locals():
            camera_onboard.destroy()
        if 'camera_third_person' in locals():
            camera_third_person.destroy()
        if 'vehicle' in locals():
            vehicle.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()
