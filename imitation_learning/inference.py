import carla
import pygame
import numpy as np
import argparse
import torch
from torchvision import transforms
from PIL import Image

# Importamos directamente la arquitectura purista que escribiste
from train import PilotNetEnhancedConditional, PilotNetSemanticConditional
from torchvision.transforms import InterpolationMode 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Ruta absoluta local al peso il_best_pilotnet_*.pth')
    parser.add_argument('--image_type', type=str, choices=['rgb', 'segsem'], default='rgb', help='Elige la camara para inferir')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=2000, type=int)
    parser.add_argument('--grad', action='store_true', help='Activa la visualizacion de atencion Grad-CAM a la izquierda')
    args = parser.parse_args()

    # =========================================================================
    # 1. ARRANQUE PYTORCH NEURAL ENGINE
    # =========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.image_type == 'segsem':
        model = PilotNetSemanticConditional().to(device)
    else:
        model = PilotNetEnhancedConditional().to(device)
    
    # Inyectamos los pesos musculares entrenados (strict=False para ignorar el viejo ln_1 si existe)
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.eval() # Congelar gradientes y normalizaciones para inferencia ciega
    print(f"[PyTorch] Modelo restaurado al 100% sobre GPU: {args.model_path}")

    # Este pipeline de transformación matemático DEBE SER IDÉNTICO AL DEL TRAIN.PY
    interp_mode = InterpolationMode.NEAREST if args.image_type == 'segsem' else InterpolationMode.BILINEAR
    data_transform = transforms.Compose([
        # Recorte 400x132 (mantiene el aspecto 3:1 de PilotNet)
        transforms.Lambda(lambda img: img.crop((0, 150, 400, 242))), 
        
        # CAMBIO CRUCIAL: Dinamico
        transforms.Resize((66, 200), interpolation=interp_mode),
        
        transforms.ToTensor()
    ])

    # =========================================================================
    # 1.5. GRAD-CAM (OPCIONAL)
    # =========================================================================
    if args.grad:
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            import cv2
        except ImportError:
            print("[!] Módulo 'grad-cam' o 'cv2' no encontrado. Instalar con: pip install grad-cam opencv-python")
            raise

        class PilotNetWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.current_speed = None
            def forward(self, img_tensor):
                return self.model(img_tensor, self.current_speed)
                
        class SteeringTarget:
            def __call__(self, model_outputs):
                # Índice 1 corresponde a la decisión de Steering (Volante). Salida es 1D (3,)
                return model_outputs[1]
                
        wrapped_model = PilotNetWrapper(model)
        # Cambiamos cn_5 a cn_3. cn_5 colapsa la altura a 1 pixel (1x18), lo que generaba lineas verticales al hacer upsampling.
        # cn_3 retiene una resolución bidimensional de 5x22, ideal para proyectar atención vertical y horizontal.
        target_layers = [model.cn_3] 
        cam_extractor = GradCAM(model=wrapped_model, target_layers=target_layers)
        cam_targets = [SteeringTarget()]

    # =========================================================================
    # 2. SANDBOX CARLA ASINCRONO
    # =========================================================================
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.load_world('Town04')
    
    bp_lib = world.get_blueprint_library()

    print("Reseteando entorno y haciendo Spawn del Ego Vehicle...")
    vehicle_bp = bp_lib.find('vehicle.tesla.model3')
    spawn_points = world.get_map().get_spawn_points()

    opt_spawn_points = list(range(230, 236)) + list(range(245, 251))

    import random
    spawn_index = random.choice(opt_spawn_points)
    print(f"Spawn index: {spawn_index}")
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[spawn_index])

    # =========================================================================
    # 3. SETUP DE CÁMARAS DUALES
    # =========================================================================
    
    # Cámara A: Visión Testigo en 3º Persona para que tu Pygame monitorice (No va a la IA)
    cam_bp_3rd = bp_lib.find('sensor.camera.rgb')
    cam_bp_3rd.set_attribute('image_size_x', '800')
    cam_bp_3rd.set_attribute('image_size_y', '600')
    cam_bp_3rd.set_attribute('fov', '90')
    transform_3rd = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-15))
    camera_3rd = world.spawn_actor(cam_bp_3rd, transform_3rd, attach_to=vehicle)

    # Cámara B: Foco Ocular Integrado al Capó para la Red Neuronal (400x300)
    if args.image_type == 'rgb':
        cam_bp_hood = bp_lib.find('sensor.camera.rgb')
    else:
        cam_bp_hood = bp_lib.find('sensor.camera.semantic_segmentation')
        
    cam_bp_hood.set_attribute('image_size_x', '400')
    cam_bp_hood.set_attribute('image_size_y', '300')
    cam_bp_hood.set_attribute('fov', '90')
    transform_hood = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0))
    camera_hood = world.spawn_actor(cam_bp_hood, transform_hood, attach_to=vehicle)

    # Variables transaccionales de punteros para almacenar el flujo asíncrono
    image_data = {'3rd': None, 'hood': None}

    def process_3rd(image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1] # BGR a RGB para Pygame visual
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        image_data['3rd'] = surface

    def process_hood(image):
        if args.image_type == 'segsem':
            # Extracción pura de los IDs categóricos matemáticos (Canal Rojo del buffer en bruto)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            class_ids = array[:, :, 2].copy() # Extraemos la escala de grises intacta
            image_data['hood'] = class_ids
        else:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3] 
            array = array[:, :, ::-1] # BGR a RGB -> PIL lo exige inherentemente así.
            image_data['hood'] = array

    camera_3rd.listen(process_3rd)
    camera_hood.listen(process_hood)

    # =========================================================================
    # 4. LOOP DE PYGAME MASTER 
    # =========================================================================
    pygame.init()
    if args.grad:
        display = pygame.display.set_mode((1200, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
    else:
        display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("PilotNet Autonomous Driver - INFERENCE")
    clock = pygame.time.Clock()

    print("\n[>>] Activando bucle neuronal asíncrono puro a 60 FPS... (Pulsa ESC para abortar)")
    
    try:
        while True:
            clock.tick(15) 
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt

            # Pintar la pantalla principal Pygame con la vista de espectador
            if image_data['3rd'] is not None:
                if args.grad:
                    display.blit(image_data['3rd'], (400, 0))
                else:
                    display.blit(image_data['3rd'], (0, 0))
            
            # -----------------------------------------------------------------
            # LATIDO NEURONAL VIVO (INFERENCIA)
            # -----------------------------------------------------------------
            if image_data['hood'] is not None:
                if args.image_type == 'segsem':
                    pil_img = Image.fromarray(image_data['hood'], mode='L')
                    img_t = data_transform(pil_img) # Reduce y crea 1 Tensor Flotante (1, 66, 200)
                    
                    class_ids = (img_t * 255.0).round().long().squeeze(0)
                    sem_tensor = torch.zeros((5, class_ids.shape[0], class_ids.shape[1]), dtype=torch.float32)
                    import cv2
                    line_mask = (class_ids == 24).float().numpy()
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    sem_tensor[0] = torch.from_numpy(cv2.dilate(line_mask, kernel, iterations=1)) # C0 (RoadLines Engrosado)
                    sem_tensor[1] = (class_ids == 1).float() * 0.4
                    sem_tensor[2] = ((class_ids == 12) | (class_ids == 13) | (class_ids == 14) | (class_ids == 15) | (class_ids == 16) | (class_ids == 18) | (class_ids == 19)).float() * 0.5
                    sem_tensor[3] = ((class_ids == 2) | (class_ids == 5) | (class_ids == 28)).float() * 0.5
                    sem_tensor[4] = ((class_ids == 6) | (class_ids == 7) | (class_ids == 8)).float() * 0.5
                    
                    img_tensor = sem_tensor.unsqueeze(0).to(device)
                else:
                    pil_img = Image.fromarray(image_data['hood'])
                    img_tensor = data_transform(pil_img).unsqueeze(0).to(device)

                # 3. Leer Velocidad en crudo del simulador dinámicamente y normalizar su tensor
                velocity = vehicle.get_velocity()
                speed_kmh = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                speed_norm = max(0.0, min(speed_kmh / 50.0, 1.0))
                speed_tensor = torch.tensor([[speed_norm]], dtype=torch.float32).to(device)
                
                # 4. Magia Pura (La IA predice los movimientos instantáneos)
                with torch.no_grad():
                    outputs = model(img_tensor, speed_tensor)
                    throttle_pred = outputs[0][0].item()
                    steering_pred = outputs[0][1].item()
                    brake_pred    = outputs[0][2].item()
                
                if args.grad:
                    # Rellenar fondo negro a la izquierda y dibujar heatmap centrado verticalmente
                    pygame.draw.rect(display, (0, 0, 0), (0, 0, 400, 600))
                    
                    # Grad-CAM requiere computar gradientes temporalmente
                    with torch.enable_grad():
                        wrapped_model.current_speed = speed_tensor
                        grayscale_cam = cam_extractor(input_tensor=img_tensor, targets=cam_targets)[0]
                        
                    # grayscale_cam es (66, 200) float32 entre 0 y 1.
                    # Redimensionar solo al crop válido 400x92 (ROI de PilotNet)
                    cam_resized_to_crop = cv2.resize(grayscale_cam, (400, 92))
                    
                    # Rellenar con ceros la zona superior (150px) e inferior (58px) para igualar 400x300
                    full_cam = np.zeros((300, 400), dtype=np.float32)
                    
                    cam_resized_to_crop[cam_resized_to_crop < 0.3] = 0
                    full_cam[150:242, :] = cam_resized_to_crop
                    full_cam = cv2.GaussianBlur(full_cam, (11, 11), 0)
                    
                    if args.image_type == 'rgb':
                        rgb_img = image_data['hood'].astype(np.float32) / 255.0
                    else:
                        # Para segsem, pasarlo a base RGB simulada:
                        cam_vis_base = image_data['hood'].astype(np.float32) / 27.0
                        rgb_img = np.stack((cam_vis_base,)*3, axis=-1)
                        
                    # Asegurar estrictamente el rango [0,1] (Carla SegSem IDs pueden pasar de 27)
                    rgb_img = np.clip(rgb_img, 0.0, 1.0)
                    cam_image = show_cam_on_image(rgb_img, full_cam, use_rgb=True)
                    cam_surface = pygame.surfarray.make_surface(cam_image.swapaxes(0, 1))
                    
                    display.blit(cam_surface, (0, 150))
                    # Contorno opcional para enmarcar la atención
                    pygame.draw.rect(display, (255, 255, 255), (0, 150, 400, 300), 2)
                
                # 5. Clamp a los valores físicos del hardware de CARLA
                throttle_cmd = max(0.0, min(1.0, throttle_pred))
                steering_cmd = max(-1.0, min(1.0, steering_pred))
                brake_cmd    = max(0.0, min(1.0, brake_pred))
                
                # 6. Despachar la decisión mecánica al Chasis directamente omitiendo PIDs
                control = carla.VehicleControl(throttle=throttle_cmd, steer=steering_cmd, brake=brake_cmd)
                if control.throttle > 0.1:
                    control.brake = 0.0
                vehicle.apply_control(control)
                
                # Overlay Visual de Pygame para ver las entrañas del cerebro IA
                font = pygame.font.SysFont(None, 36)
                offset_x = 420 if args.grad else 20
                display.blit(font.render(f"Ego Speed: {speed_kmh:.1f} km/h", True, (255, 255, 255)), (offset_x, 20))
                display.blit(font.render(f"[AI] Throttle: {throttle_pred:.3f}", True, (0, 255, 0)), (offset_x, 60))
                display.blit(font.render(f"[AI] Steer:    {steering_pred:.3f}", True, (255, 100, 100)), (offset_x, 95))
                display.blit(font.render(f"[AI] Brake:    {brake_pred:.3f}", True, (255, 50, 50)), (offset_x, 130))

            pygame.display.flip()

    except KeyboardInterrupt:
        print("\n\n[!!] Bucle Maestro IA Interrumpido por el Usuario.")
    finally:
        print("Borrando clones de simulacion virtuales en servidor...")
        if 'camera_3rd' in locals(): camera_3rd.destroy()
        if 'camera_hood' in locals(): camera_hood.destroy()
        if 'vehicle' in locals(): vehicle.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()
