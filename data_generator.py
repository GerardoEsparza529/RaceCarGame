"""
Generador de Datos de Entrenamiento
Usa el controlador difuso como "maestro" para generar ejemplos
"""
import numpy as np
import pickle
import os
from car import Car
from track import Track
from fuzzy_controller import FuzzyController
import pygame

class DataGenerator:
    def __init__(self):
        """Inicializa el generador de datos"""
        self.data_X = []  # Estados (entrada)
        self.data_y = []  # Acciones (salida)
        
    def generate_training_data(self, num_samples=5000, save_path='data/training_data.pkl'):
        """
        Genera datos de entrenamiento usando el controlador difuso
        
        Args:
            num_samples: Número de muestras a generar
            save_path: Ruta donde guardar los datos
        """
        print(f"\n📊 Generando {num_samples} muestras de entrenamiento...")
        
        # Inicializar Pygame (sin ventana visible)
        pygame.init()
        screen = pygame.Surface((1200, 800))
        
        # Crear objetos del juego
        track = Track(1200, 800)
        car = Car(600, 150, (0, 120, 255), is_player=True)
        fuzzy = FuzzyController()
        
        # Resetear auto a posición inicial
        start_x, start_y, start_angle = track.get_start_position(0)
        car.reset(start_x, start_y, start_angle)
        
        samples_collected = 0
        episode = 0
        max_episodes = 100  # Número máximo de episodios
        
        while samples_collected < num_samples and episode < max_episodes:
            episode += 1
            car.reset(start_x, start_y, start_angle)
            steps_without_crash = 0
            max_steps_per_episode = 500
            
            print(f"  Episodio {episode}/{max_episodes} - Muestras: {samples_collected}/{num_samples}")
            
            for step in range(max_steps_per_episode):
                # Actualizar sensores
                car.update_sensors(track)
                
                # Obtener acción del controlador difuso
                steering, throttle = fuzzy.compute(car)
                
                # Guardar estado y acción
                state = car.get_state_vector()
                action = np.array([steering, throttle])
                
                self.data_X.append(state)
                self.data_y.append(action)
                samples_collected += 1
                
                # Aplicar control y física
                car.update_ai_control(steering, throttle)
                car.apply_physics()
                
                # Verificar colisión
                if track.check_collision(car):
                    steps_without_crash = 0
                    break  # Reiniciar episodio
                else:
                    steps_without_crash += 1
                
                # Si llegamos a num_samples, terminar
                if samples_collected >= num_samples:
                    break
            
            # Añadir variación aleatoria ocasional para diversidad
            if episode % 10 == 0:
                # Generar algunas muestras con ruido
                for _ in range(50):
                    if samples_collected >= num_samples:
                        break
                    
                    # Estado aleatorio pero plausible (velocidad + 16 sensores = 17 valores)
                    random_state = np.random.rand(17)
                    random_state[0] = (random_state[0] - 0.5) * 2  # Velocidad en [-1, 1]
                    
                    # Crear car temporal con estos sensores
                    temp_car = Car(600, 150, (0, 0, 0))
                    temp_car.speed = random_state[0] * temp_car.max_speed
                    temp_car.sensor_distances = list(random_state[1:] * 150)
                    
                    # Obtener acción del difuso
                    steering, throttle = fuzzy.compute(temp_car)
                    action = np.array([steering, throttle])
                    
                    self.data_X.append(random_state)
                    self.data_y.append(action)
                    samples_collected += 1
        
        # Convertir a arrays de numpy
        X = np.array(self.data_X)
        y = np.array(self.data_y)
        
        print(f"\n✓ Generación completada:")
        print(f"   Total de muestras: {X.shape[0]}")
        print(f"   Forma de X: {X.shape}")
        print(f"   Forma de y: {y.shape}")
        
        # Estadísticas
        print(f"\n📈 Estadísticas de los datos:")
        print(f"   Steering - Min: {y[:, 0].min():.3f}, Max: {y[:, 0].max():.3f}, Mean: {y[:, 0].mean():.3f}")
        print(f"   Throttle - Min: {y[:, 1].min():.3f}, Max: {y[:, 1].max():.3f}, Mean: {y[:, 1].mean():.3f}")
        
        # Guardar datos
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({'X': X, 'y': y}, f)
        
        print(f"✓ Datos guardados en {save_path}")
        
        pygame.quit()
        return X, y
    
    def load_training_data(self, load_path='data/training_data.pkl'):
        """
        Carga datos de entrenamiento previamente generados
        
        Args:
            load_path: Ruta de donde cargar los datos
            
        Returns:
            Tupla (X, y) con los datos
        """
        if not os.path.exists(load_path):
            print(f"⚠ No se encontraron datos en {load_path}")
            return None, None
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        X = data['X']
        y = data['y']
        
        print(f"✓ Datos cargados desde {load_path}")
        print(f"   Muestras: {X.shape[0]}")
        
        return X, y

def split_data(X, y, train_ratio=0.8, val_ratio=0.1):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba
    
    Args:
        X: Datos de entrada
        y: Datos de salida
        train_ratio: Proporción para entrenamiento
        val_ratio: Proporción para validación
        
    Returns:
        Tupla (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Mezclar datos
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Calcular tamaños
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Dividir
    X_train = X_shuffled[:n_train]
    y_train = y_shuffled[:n_train]
    
    X_val = X_shuffled[n_train:n_train + n_val]
    y_val = y_shuffled[n_train:n_train + n_val]
    
    X_test = X_shuffled[n_train + n_val:]
    y_test = y_shuffled[n_train + n_val:]
    
    print(f"\n📊 División de datos:")
    print(f"   Entrenamiento: {X_train.shape[0]} muestras ({train_ratio*100:.0f}%)")
    print(f"   Validación: {X_val.shape[0]} muestras ({val_ratio*100:.0f}%)")
    print(f"   Prueba: {X_test.shape[0]} muestras ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    # Generar datos
    generator = DataGenerator()
    X, y = generator.generate_training_data(num_samples=5000)
    
    # Dividir datos
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    
    print("\n✓ Proceso completado")
