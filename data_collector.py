"""
Data Collector - Captura datos de conducción manual para entrenar la IA
"""
import csv
import os
from datetime import datetime

class DataCollector:
    def __init__(self):
        """Inicializa el colector de datos"""
        self.data_buffer = []
        self.is_recording = False
        self.filename = None
        self.frames_recorded = 0
        self.min_speed_threshold = 1.0  # Solo guardar cuando el auto se mueve
        
    def start_recording(self):
        """Inicia una nueva sesión de grabación"""
        # Crear nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"training_data_{timestamp}.csv"
        
        self.data_buffer = []
        self.is_recording = True
        self.frames_recorded = 0
        
        print(f"📹 Grabación iniciada: {self.filename}")
        print("   Conduce una vuelta completa sin chocar para mejores resultados")
        
    def stop_recording(self):
        """Detiene la grabación y guarda los datos"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if len(self.data_buffer) == 0:
            print("⚠ No hay datos para guardar")
            return
        
        # Guardar a archivo CSV
        self._save_to_csv()
        
        print(f"✅ Grabación guardada: {self.filename}")
        print(f"   Frames capturados: {self.frames_recorded}")
        print(f"   Tiempo aproximado: {self.frames_recorded / 60:.1f} segundos")
        
    def record_frame(self, car, steering, throttle):
        """
        Captura los datos de un frame
        
        Args:
            car: Objeto Car con sensores y estado
            steering: Valor de dirección (-1 a 1)
            throttle: Valor de aceleración (-1 a 1)
        """
        if not self.is_recording:
            return
        
        # Solo grabar cuando el auto se está moviendo y no ha chocado
        if abs(car.speed) < self.min_speed_threshold or car.crashed:
            return
        
        # Construir registro: [16 sensores, velocidad, steering, throttle]
        # Total: 19 valores
        record = []
        
        # Agregar distancias de los 16 sensores (normalizadas)
        for dist in car.sensor_distances:
            normalized_dist = dist / car.sensor_length  # 0 a 1
            record.append(normalized_dist)
        
        # Agregar velocidad normalizada
        normalized_speed = car.speed / car.max_speed  # -1 a 1
        record.append(normalized_speed)
        
        # Agregar acciones del jugador
        record.append(steering)
        record.append(throttle)
        
        self.data_buffer.append(record)
        self.frames_recorded += 1
        
    def _save_to_csv(self):
        """Guarda los datos en un archivo CSV"""
        # Crear directorio si no existe
        os.makedirs("training_data", exist_ok=True)
        filepath = os.path.join("training_data", self.filename)
        
        # Escribir CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Encabezado
            header = []
            for i in range(16):
                header.append(f"sensor_{i}")
            header.append("velocity")
            header.append("steering")
            header.append("throttle")
            writer.writerow(header)
            
            # Datos
            writer.writerows(self.data_buffer)
        
        print(f"   Archivo guardado en: {filepath}")
        
    def get_status(self):
        """Retorna el estado actual de la grabación"""
        if self.is_recording:
            return f"🔴 REC [{self.frames_recorded} frames]"
        else:
            return "⚪ Listo para grabar (presiona G)"
