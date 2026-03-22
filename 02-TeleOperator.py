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

def draw_hud(display, control):
    font = pygame.font.SysFont(None, 24)
    # HUD surface with transparency
    hud_surface = pygame.Surface((240, 160), pygame.SRCALPHA)
    hud_surface.fill((0, 0, 0, 180))
    
    t_text = font.render(f"Throttle: {control.throttle:.2f}", True, (255, 255, 255))
    b_text = font.render(f"Brake: {control.brake:.2f}", True, (255, 255, 255))
    s_text = font.render(f"Steer: {control.steer:.2f}", True, (255, 255, 255))
    r_text = font.render(f"Reverse (R): {'ON' if control.reverse else 'OFF'}", True, (255, 255, 255))
    
    hud_surface.blit(t_text, (10, 20))
    hud_surface.blit(b_text, (10, 50))
    hud_surface.blit(s_text, (10, 80))
    hud_surface.blit(r_text, (10, 120))

    # Throttle bar
    pygame.draw.rect(hud_surface, (50, 50, 50), (120, 25, 100, 12))
    pygame.draw.rect(hud_surface, (0, 255, 0), (120, 25, int(100 * control.throttle), 12))

    # Brake bar
    pygame.draw.rect(hud_surface, (50, 50, 50), (120, 55, 100, 12))
    pygame.draw.rect(hud_surface, (255, 0, 0), (120, 55, int(100 * control.brake), 12))

    # Steer bar (centered)
    pygame.draw.rect(hud_surface, (50, 50, 50), (120, 85, 100, 12))
    pygame.draw.line(hud_surface, (255, 255, 255), (170, 80), (170, 100), 2)
    if control.steer < 0:
        pygame.draw.rect(hud_surface, (0, 0, 255), (170 + int(50 * control.steer), 85, int(-50 * control.steer), 12))
    elif control.steer > 0:
        pygame.draw.rect(hud_surface, (0, 0, 255), (170, 85, int(50 * control.steer), 12))

    display.blit(hud_surface, (20, 20))


def main():
    parser = argparse.ArgumentParser(description="CARLA TeleOperator")
    parser.add_argument('--host', default='127.0.0.1', help='IP of the host server')
    parser.add_argument('--port', default=2000, type=int, help='TCP port to listen to')
    args = parser.parse_args()

    pygame.init()
    width, height = 800, 600
    display = pygame.display.set_mode((width * 2, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA TeleOperator - Keyboard Control")

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
        # Autopilot NOT enabled for teleoperation

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

        print("Simulation running.")
        print("CONTROLS:")
        print("  UP/DOWN      : Throttle / Brake")
        print("  LEFT/RIGHT   : Steer")
        print("  R Key        : Toggle Reverse")
        print("  ESC          : Exit")

        clock = pygame.time.Clock()
        
        control = carla.VehicleControl()

        while True:
            clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE):
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    control.reverse = not control.reverse

            keys = pygame.key.get_pressed()
            
            # Throttle control
            if keys[pygame.K_UP]:
                control.throttle = min(control.throttle + 0.05, 1.0)
            else:
                control.throttle = max(control.throttle - 0.1, 0.0)

            # Brake control
            if keys[pygame.K_DOWN]:
                control.brake = min(control.brake + 0.1, 1.0)
            else:
                control.brake = max(control.brake - 0.1, 0.0)

            # Steer control
            if keys[pygame.K_LEFT]:
                control.steer = max(control.steer - 0.05, -1.0)
            elif keys[pygame.K_RIGHT]:
                control.steer = min(control.steer + 0.05, 1.0)
            else:
                if control.steer > 0:
                    control.steer = max(control.steer - 0.05, 0.0)
                elif control.steer < 0:
                    control.steer = min(control.steer + 0.05, 0.0)

            # Apply limits just in case
            control.throttle = 0.5 #max(0.0, min(1.0, control.throttle))
            control.brake = max(0.0, min(1.0, control.brake))
            control.steer = max(-1.0, min(1.0, control.steer))

            vehicle.apply_control(control)

            if images['onboard'] is not None:
                render_image(images['onboard'], display, (0, 0))
            if images['third_person'] is not None:
                render_image(images['third_person'], display, (width, 0))

            # Render the Head-Up Display (HUD)
            draw_hud(display, control)

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
