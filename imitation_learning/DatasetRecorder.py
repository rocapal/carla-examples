import os
import time
import csv
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    from PIL import Image

class DatasetRecorder:
    def __init__(self, base_folder, capture_hz=10.0):
        self.base_folder = os.path.abspath(base_folder)
        self.capture_interval = 1.0 / capture_hz if capture_hz > 0 else 0.0
        self.last_record_time = 0.0
        self.is_paused = False
        
        # Generar la carpeta del experimento actual (run_$timestamp_epoch)
        self.run_id = f"run_{int(time.time())}"
        self.run_folder = os.path.join(self.base_folder, self.run_id)
        
        # Subcarpetas RGB y Segmentación Semántica
        self.rgb_folder = os.path.join(self.run_folder, "rgb")
        self.segsem_folder = os.path.join(self.run_folder, "segsem")
        
        os.makedirs(self.rgb_folder, exist_ok=True)
        os.makedirs(self.segsem_folder, exist_ok=True)
        
        # Inicializar Base de Datos CSV
        self.csv_path = os.path.join(self.run_folder, "data.csv")
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp", "path_rgb", "path_segsem", "throttle", "steering", "brake", "ego_speed"])
        
        print(f"[DatasetRecorder] Inicializado entorno de grabacion: {self.run_folder} (Frecuencia: {capture_hz}Hz)")

    def _save_image(self, carla_image, path):
        # Transformar RAW bytes de memoria a NumPy tensorial (4 canales BGRA)
        array = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (carla_image.height, carla_image.width, 4))
        # Recortar el canal Alfa y quedarse con puramente Visual (BGR final)
        array = array[:, :, :3] 
        
        if HAS_CV2:
            # OpenCV respeta el protocolo colorimétrico BGR de origen.
            cv2.imwrite(path, array)
        else:
            # Fallback a PIL: Exige invertir los canales BGR a estándar universal RGB
            array = array[:, :, ::-1]
            img = Image.fromarray(array)
            img.save(path, format='JPEG', quality=95)

    def _save_numpy_mask(self, array, path):
        if HAS_CV2:
            cv2.imwrite(path, array) # CV2 automáticamente guarda en 8-bit PNG si la matriz es bidimensional y la extensión es `.png`
        else:
            img = Image.fromarray(array, mode='L')
            img.save(path, format='PNG')

    def pause(self):
        if not self.is_paused:
            self.is_paused = True
            print(f"[DatasetRecorder] PAUSA: Grabación de dataset temporalmente detenida.")

    def resume(self):
        if self.is_paused:
            self.is_paused = False
            self.last_record_time = time.time()
            print(f"[DatasetRecorder] REANUDADO: Grabación de dataset activada.")

    def record(self, timestamp, rgb_image, sem_image, throttle, steering, brake, speed):
        if self.is_paused:
            return
            
        current_time = time.time()
        
        # Comprobar el filtro de tiempo 10Hz parameterizado (cada 0.1s físico del programa)
        if current_time - self.last_record_time >= self.capture_interval:
            self.last_record_time = current_time
            
            # Castear string fijo a .6 para evitar errores de sistema de ficheros
            ts_str = f"{timestamp:.6f}"
            
            # Generar localizaciones absolutas en disco (RGB es JPG, SEGSEM es PNG 8-bits)
            rgb_path = os.path.join(self.rgb_folder, f"{ts_str}.jpg")
            sem_path = os.path.join(self.segsem_folder, f"{ts_str}.png")
            
            # Volcar las matrices renderizadas al disco (Usando métodos dedicados)
            self._save_image(rgb_image, rgb_path)
            self._save_numpy_mask(sem_image, sem_path)
            
            # Montar rutas relativas seguras para el CSV (exigido)
            rel_rgb_path = os.path.join("rgb", f"{ts_str}.jpg")
            rel_sem_path = os.path.join("segsem", f"{ts_str}.png")
            
            # Volcar vector meta-data por fila y purgar búfer interno para evitar pérdidas
            self.csv_writer.writerow([ts_str, rel_rgb_path, rel_sem_path, 
                                      f"{throttle:.4f}", f"{steering:.4f}", 
                                      f"{brake:.4f}", f"{speed:.2f}"])
            self.csv_file.flush() 
                                      
    def close(self):
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
            print(f"[DatasetRecorder] Descriptor CSV de disco desvinculado con éxito ({self.run_folder}).")
            
    def delete_last_seconds(self, seconds=2.0):
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()

        if not os.path.exists(self.csv_path):
            return

        with open(self.csv_path, 'r') as f:
            lines = list(csv.reader(f))

        if len(lines) <= 1:
            return

        header = lines[0]
        data_rows = lines[1:]

        try:
            latest_timestamp = float(data_rows[-1][0])
        except ValueError:
            return

        cutoff_time = latest_timestamp - seconds

        kept_rows = []
        deleted_count = 0

        for row in data_rows:
            try:
                ts = float(row[0])
                if ts < cutoff_time:
                    kept_rows.append(row)
                else:
                    rgb_path = os.path.join(self.base_folder, self.run_id, row[1])
                    sem_path = os.path.join(self.base_folder, self.run_id, row[2])
                    
                    if os.path.exists(rgb_path):
                        os.remove(rgb_path)
                    if os.path.exists(sem_path):
                        os.remove(sem_path)
                    deleted_count += 1
            except Exception:
                kept_rows.append(row)

        print(f"[DatasetRecorder] Eliminando ultimos {seconds}s de archivo: {deleted_count} frames borrados por anomalía en simulación.")

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(kept_rows)
