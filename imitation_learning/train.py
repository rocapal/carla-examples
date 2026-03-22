import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
import random

class CarlaDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, image_type='rgb', data_aug=False):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.image_type = image_type
        self.data_aug = data_aug
        self.samples = []
        
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Dataset directory '{dataset_dir}' does not exist.")
            
        for run_folder in os.listdir(dataset_dir):
            run_path = os.path.join(dataset_dir, run_folder)
            csv_path = os.path.join(run_path, 'data.csv')
            
            if os.path.isdir(run_path) and os.path.exists(csv_path):
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    next(reader, None) # Skip Header
                    for row in reader:
                        if len(row) < 7: continue
                        
                        rel_path = row[1] if self.image_type == 'rgb' else row[2]
                        throttle = float(row[3])
                        steering = float(row[4])
                        brake = float(row[5])
                        ego_speed = float(row[6]) # Recuperado
                        
                        abs_path = os.path.abspath(os.path.join(run_path, rel_path))
                        if os.path.exists(abs_path):
                            self.samples.append((abs_path, throttle, steering, brake, ego_speed))
                            
        print(f"[Dataset] Gathered {len(self.samples)} valid samples (Type: {self.image_type.upper()}) from {dataset_dir}")

    def __len__(self):
        # Multiplicar lógicamente el dataset x3 nativamente solo si el flag está activo
        return len(self.samples) * 3 if self.data_aug else len(self.samples)
    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        img_path, throttle, steering, brake, speed = self.samples[real_idx]
        
        if self.image_type == 'rgb':
            img = Image.open(img_path).convert('RGB')
            if self.data_aug and idx >= len(self.samples) and idx < len(self.samples) * 2:
                img = ImageOps.mirror(img)
                steering = -steering
            elif self.data_aug and idx >= len(self.samples) * 2:
                angle = random.uniform(-5.0, 5.0)
                img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0,0,0))
                steering += (angle * -0.015)
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(random.uniform(0.6, 1.4))
                
            steering = max(-1.0, min(1.0, steering))
            throttle = max(0.0, min(1.0, throttle))
            brake = max(0.0, min(1.0, brake))
            
            if self.transform:
                img = self.transform(img)
                
            if self.data_aug and idx >= len(self.samples) * 2:
                noise = torch.randn_like(img) * 0.05
                img = torch.clamp(img + noise, 0.0, 1.0)
        else:
            # SEMANTIC BIFURCATION (5-Channel OHE)
            img = Image.open(img_path).convert('L')
            
            if self.data_aug and idx >= len(self.samples) and idx < len(self.samples) * 2:
                img = ImageOps.mirror(img)
                steering = -steering
            elif self.data_aug and idx >= len(self.samples) * 2:
                angle = random.uniform(-5.0, 5.0)
                img = img.rotate(angle, resample=Image.NEAREST, fillcolor=0)
                steering += (angle * -0.015)
                
            steering = max(-1.0, min(1.0, steering))
            throttle = max(0.0, min(1.0, throttle))
            brake = max(0.0, min(1.0, brake))
            
            if self.transform:
                img = self.transform(img)
                
            # Decodificar flotante 0-1 a Class IDs reales
            class_ids = (img * 255.0).round().long().squeeze(0) # (66, 200)
            semantic_tensor = torch.zeros((5, class_ids.shape[0], class_ids.shape[1]), dtype=torch.float32)
            
            # C0 (RoadLines): 24 (Engrosado con OpenCV Dilation)
            import cv2
            line_mask = (class_ids == 24).float().numpy()
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
            semantic_tensor[0] = torch.from_numpy(cv2.dilate(line_mask, kernel, iterations=1)) # Mantiene 1.0
            # C1 (Roads): 1
            semantic_tensor[1] = (class_ids == 1).float() * 0.3
            # C2 (Dynamics): 12, 13, 14, 15, 16, 18, 19
            semantic_tensor[2] = ((class_ids == 12) | (class_ids == 13) | (class_ids == 14) | (class_ids == 15) | (class_ids == 16) | (class_ids == 18) | (class_ids == 19)).float() * 0.3
            # C3 (Borders): 2, 5, 28
            semantic_tensor[3] = ((class_ids == 2) | (class_ids == 5) | (class_ids == 28)).float() * 0.3
            # C4 (Pathing): 6, 7, 8
            semantic_tensor[4] = ((class_ids == 6) | (class_ids == 7) | (class_ids == 8)).float() * 0.3
            
            img = semantic_tensor

        # Normalizar velocidad (Target = 50km/h max)
        speed_norm = max(0.0, min(speed / 50.0, 1.0))
        speed_tensor = torch.tensor([speed_norm], dtype=torch.float32)

        target_tensor = torch.tensor([throttle, steering, brake], dtype=torch.float32)
        
        return (img, speed_tensor), target_tensor


class PilotNetEnhancedConditional(nn.Module):
    def __init__(self):
        super(PilotNetEnhancedConditional, self).__init__()
        
        # Original PilotNetEnhanced Architecture (urjc-deepracer)
        self.img_height = 66
        self.img_width = 200
        self.num_channels = 3
        
        # Capa extra inicial
        self.ln_1 = nn.BatchNorm2d(self.num_channels, eps=1e-03)

        self.cn_1 = nn.Conv2d(self.num_channels, 24, kernel_size=5, stride=2)
        self.relu1 = nn.ReLU()
        self.cn_2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.relu2 = nn.ReLU()
        self.cn_3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.relu3 = nn.ReLU()
        self.cn_4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.relu4 = nn.ReLU()
        self.cn_5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu5 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        
        # Su capa fc_1 extendida (1152 -> 1164)
        self.fc_1 = nn.Linear(1 * 18 * 64, 1164)
        self.relu_fc1 = nn.ReLU()
        
        # Nuestra rama secundaria híbrida para asimilar la Velocidad (Custom)
        self.speed_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )
        
        # Capas Densas post-fusión
        #self.fc_2 = nn.Linear(1164 + 16, 100)
        self.relu_fc2 = nn.ReLU()
        #self.fc_3 = nn.Linear(100, 50)
        self.relu_fc3 = nn.ReLU()
        #self.fc_4 = nn.Linear(50, 10)
        self.relu_fc4 = nn.ReLU()
        #self.fc_5 = nn.Linear(10, 3) # [Throttle, Steering, Brake]

        # Fusión más gradual y con Dropout para generalizar mejor
        self.fc_2 = nn.Linear(1164 + 16, 256) # Intermedio
        self.drop = nn.Dropout(0.1) 
        self.fc_3 = nn.Linear(256, 100)
        self.fc_4 = nn.Linear(100, 50)
        self.fc_5 = nn.Linear(50, 10)
        self.fc_out = nn.Linear(10, 3)


    def forward(self, img, speed):
        # 1. Extractor visual con normalización local
        out = self.ln_1(img)

        out = self.cn_1(out)
        out = self.relu1(out)
        out = self.cn_2(out)
        out = self.relu2(out)
        out = self.cn_3(out)
        out = self.relu3(out)
        out = self.cn_4(out)
        out = self.relu4(out)
        out = self.cn_5(out)
        out = self.relu5(out)

        out = self.flatten(out)

        out = self.fc_1(out)
        img_features = self.relu_fc1(out)

        # 2. Extractor de flujo asimétrico de velocidad
        speed_emb = self.speed_branch(speed)

        # 3. Concatenación paralela
        combined = torch.cat((img_features, speed_emb), dim=1)

        # 4. Decodificador Físico
        #out = self.fc_2(combined)
        #out = self.relu_fc2(out)
        #out = self.fc_3(out)
        #out = self.relu_fc3(out)
        #out = self.fc_4(out)
        #out = self.relu_fc4(out)
        #out = self.fc_5(out)    
        #return out

        x = self.relu_fc2(self.fc_2(combined))    
        x = self.drop(x)    
        x = self.relu_fc3(self.fc_3(x))
        x = self.relu_fc4(self.fc_4(x))
        x = self.fc_out(self.fc_5(x))
        
        # Split y activación (Crucial para control físico)
        throttle = torch.sigmoid(x[:, 0:1])
        steering = torch.tanh(x[:, 1:2])
        brake = torch.sigmoid(x[:, 2:3])
        
        return torch.cat((throttle, steering, brake), dim=1)

class PilotNetSemanticConditional(nn.Module):
    def __init__(self):
        super(PilotNetSemanticConditional, self).__init__()
        
        self.img_height = 66
        self.img_width = 200
        self.num_channels = 5 # 5 Semantic Layers (RoadLines, Roads, Dynamics, Borders, Pathing)
        
        # Desactivamos BatchNorm2d para que aplique puramente la ponderación atencional estática
        # self.ln_1 = nn.BatchNorm2d(self.num_channels, eps=1e-03)

        self.cn_1 = nn.Conv2d(self.num_channels, 24, kernel_size=5, stride=2)
        self.relu1 = nn.ReLU()
        self.cn_2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.relu2 = nn.ReLU()
        self.cn_3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.relu3 = nn.ReLU()
        self.cn_4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.relu4 = nn.ReLU()
        self.cn_5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu5 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(1 * 18 * 64, 1164)
        self.relu_fc1 = nn.ReLU()
        
        self.speed_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )
        
        self.fc_2 = nn.Linear(1164 + 16, 256) # Intermedio
        self.relu_fc2 = nn.ReLU()
        self.drop = nn.Dropout(0.1) 
        self.fc_3 = nn.Linear(256, 100)
        self.relu_fc3 = nn.ReLU()
        self.fc_4 = nn.Linear(100, 50)
        self.relu_fc4 = nn.ReLU()
        self.fc_5 = nn.Linear(50, 10)
        self.fc_out = nn.Linear(10, 3)

    def forward(self, img, speed):
        out = img # Omitimos: out = self.ln_1(img)
        out = self.cn_1(out)
        out = self.relu1(out)
        out = self.cn_2(out)
        out = self.relu2(out)
        out = self.cn_3(out)
        out = self.relu3(out)
        out = self.cn_4(out)
        out = self.relu4(out)
        out = self.cn_5(out)
        out = self.relu5(out)

        out = self.flatten(out)
        out = self.fc_1(out)
        img_features = self.relu_fc1(out)

        speed_emb = self.speed_branch(speed)
        combined = torch.cat((img_features, speed_emb), dim=1)

        x = self.relu_fc2(self.fc_2(combined))    
        x = self.drop(x)    
        x = self.relu_fc3(self.fc_3(x))
        x = self.relu_fc4(self.fc_4(x))
        x = self.fc_out(self.fc_5(x))
        
        throttle = torch.sigmoid(x[:, 0:1])
        steering = torch.tanh(x[:, 1:2])
        brake = torch.sigmoid(x[:, 2:3])
        
        return torch.cat((throttle, steering, brake), dim=1)

def weighted_mse_loss(prediction, target, weights):
    # Error al cuadrado por cada elemento multi-task
    loss = (prediction - target) ** 2
    # Tolerancia asimétrica escalada por cada canal
    weighted_loss = loss * weights
    return weighted_loss.mean()

def main():
    from torchvision.transforms import InterpolationMode 

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='Root directory containing the generated /run_*')
    parser.add_argument('--image_type', type=str, choices=['rgb', 'segsem'], default='rgb', help='Select input imaging architecture')
    parser.add_argument('--data_aug', action='store_true', help='Activar multiplicador x3 de Online Data Augmentation')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (images per iteration)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for Adam optimizer')
    args = parser.parse_args()

    # Extract native 3:1 ROI keeping 100% width (400x132). Then exactly downscale 50% to 200x66.
    interp_mode = InterpolationMode.NEAREST if args.image_type == 'segsem' else InterpolationMode.BILINEAR
    data_transform = transforms.Compose([
        # Recorte 400x132 (mantiene el aspecto 3:1 de PilotNet)
        transforms.Lambda(lambda img: img.crop((0, 150, 400, 242))), 
        
        # CAMBIO CRUCIAL: Interpolation escalable
        transforms.Resize((66, 200), interpolation=interp_mode),
        
        transforms.ToTensor()
    ])

    print("\nInitializing Dataset Collection...")
    full_dataset = CarlaDataset(args.dataset_dir, transform=data_transform, image_type=args.image_type, data_aug=args.data_aug)
    
    if len(full_dataset) == 0:
        print("[!] No valid samples in your CSV to train. Exiting.")
        return

    # --- DEBUG PREVIEW ---
    DEBUG_SEMANTIC_PLOT = True
    print(f"Generating visual mathematical representations...")
    try:
        if args.image_type == 'rgb' or not DEBUG_SEMANTIC_PLOT:
            sample_img_path = full_dataset.samples[2000][0] 
            orig_img = Image.open(sample_img_path).convert('RGB')
            
            transformed_tensor = data_transform(orig_img)
            transformed_img = transformed_tensor.permute(1, 2, 0).numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(orig_img)
            axes[0].set_title(f"Original RGB")
            axes[0].axis('off')
            
            axes[1].imshow(transformed_img)
            axes[1].set_title(f"PilotNet Tensor Crop")
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'crop_preview_{args.image_type}.png')
            plt.close()
            print(f"  [*] RGB Preview guardada como 'crop_preview_{args.image_type}.png'")
        else:
            print(f"  [>] Extrayendo desglose Tensor Semantico 5-D...")
            (sem_tensor, _), _ = full_dataset[600] 
            
            sample_img_path = full_dataset.samples[600][0] 
            orig_img_l = Image.open(sample_img_path).convert('L') 
                        
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            
            # Subplot 1: Original base classes (L Channel)
            axes[0, 0].imshow(orig_img_l, cmap='nipy_spectral', vmin=0, vmax=27)
            axes[0, 0].set_title("Raw Class IDs (PNG Original)")
            axes[0, 0].axis('off')
            
            channels_names = [
                "C0 (RoadLines: 24)", 
                "C1 (Roads: 1)", 
                "C2 (Dynamics: 12,13,14,15,16,18,19)", 
                "C3 (Borders: 2,5,28)", 
                "C4 (Pathing: 6,7,8)"
            ]
            
            for i in range(5):
                row = (i + 1) // 3
                col = (i + 1) % 3
                axes[row, col].imshow(sem_tensor[i].numpy(), cmap='gray')
                axes[row, col].set_title(channels_names[i])
                axes[row, col].axis('off')
                
            plt.tight_layout()
            plt.savefig('semantic_channels_debug.png')
            plt.close()
            print("  [*] Tensor 5D Ploteado guardado como 'semantic_channels_debug.png'")
            
    except Exception as e:
        print(f"Failed to generate preview: {e}")
    # ---------------------

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Dataset successfully partitioned: {train_size} Training | {val_size} Validation (Random)")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting hardware Tensor Engine on: {device}\n")

    if args.image_type == 'segsem':
        model = PilotNetSemanticConditional().to(device)
    else:
        model = PilotNetEnhancedConditional().to(device)
        
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Definimos pesos VRAM asimétricos: 0.2 para throttle, 0.7 para steering, 0.1 para brake
    weights = torch.tensor([0.2, 0.7, 0.1], dtype=torch.float32).to(device)

    best_val_loss = float('inf')
    hist_train_loss = []
    hist_val_loss = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            (imgs, speeds), targets = batch
            imgs, speeds, targets = imgs.to(device), speeds.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs, speeds)
            loss = weighted_mse_loss(outputs, targets, weights) 
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                (imgs, speeds), targets = batch
                imgs, speeds, targets = imgs.to(device), speeds.to(device), targets.to(device)
                
                outputs = model(imgs, speeds)
                loss = weighted_mse_loss(outputs, targets, weights)
                val_loss += loss.item() * imgs.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        hist_train_loss.append(train_loss)
        hist_val_loss.append(val_loss)

        print(f"Epoch [{epoch:02d}/{args.epochs}] -> Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f'il_best_pilotnet_{args.image_type}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"  [*] New Validation record. Saved to -> {model_path}")

    print("\nTraining completed. Generating loss plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), hist_train_loss, label='Train MSE Loss', color='blue', linewidth=2)
    plt.plot(range(1, args.epochs + 1), hist_val_loss, label='Validation MSE Loss', color='orange', linewidth=2)
    plt.title('PilotNet Training vs Validation Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plot_path = f'training_losses_plot_{args.image_type}.png'
    plt.savefig(plot_path)
    print(f"Plot saved successfully to '{plot_path}'.")

if __name__ == '__main__':
    main()
