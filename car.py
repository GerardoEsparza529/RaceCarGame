"""
Clase Car - Representa un auto en el juego de carreras
"""
import pygame
import math
import numpy as np

class Car:
    def __init__(self, x, y, color, is_player=True, image_path=None):
        """
        Inicializa un auto
        
        Args:
            x: Posición inicial en X
            y: Posición inicial en Y
            color: Color del auto (tuple RGB)
            is_player: Si es el auto del jugador o el oponente
            image_path: Ruta opcional a una imagen PNG/JPG del carro
        """
        self.x = x
        self.y = y
        self.color = color
        self.is_player = is_player
        
        # Dimensiones del auto
        self.width = 40
        self.height = 60
        
        # Cargar imagen si se proporciona
        self.car_image = None
        self.use_image = False
        if image_path:
            try:
                self.car_image = pygame.image.load(image_path).convert_alpha()
                # Escalar imagen al tamaño del auto
                self.car_image = pygame.transform.scale(self.car_image, (self.width, self.height))
                self.use_image = True
            except (pygame.error, FileNotFoundError) as e:
                print(f"No se pudo cargar la imagen del carro: {e}")
                print("Se usará el dibujo por defecto.")
                self.use_image = False
        
        # Física del auto
        self.angle = 0  # Ángulo en grados (0 = arriba, 90 = derecha, 180 = abajo, 270 = izquierda)
        self.speed = 0
        self.max_speed = 8
        self.acceleration = 0.3
        self.friction = 0.05
        self.turn_speed = 4
        
        # Sensores para detección (16 rayos alrededor del auto - cada 22.5 grados)
        self.sensor_distances = [0] * 16
        self.sensor_angles = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 
                             180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]
        self.sensor_length = 150
        
        # Estadísticas
        self.lap_count = 0
        self.checkpoint_count = 0
        self.total_distance = 0
        self.crashed = False
        
        # Seguimiento de posición para detección de checkpoints
        self.prev_x = x
        self.prev_y = y
        
    def update_manual(self, keys):
        """Control manual con teclas de flecha"""
        if keys[pygame.K_UP]:
            self.speed = min(self.speed + self.acceleration, self.max_speed)
        if keys[pygame.K_DOWN]:
            self.speed = max(self.speed - self.acceleration, -self.max_speed / 2)
        if keys[pygame.K_LEFT] and abs(self.speed) > 0.5:
            self.angle -= self.turn_speed
        if keys[pygame.K_RIGHT] and abs(self.speed) > 0.5:
            self.angle += self.turn_speed
            
    def update_ai_control(self, steering, throttle):
        """
        Control por IA (difuso o red neuronal)
        
        Args:
            steering: Valor entre -1 (izquierda) y 1 (derecha)
            throttle: Valor entre -1 (reversa) y 1 (acelerar)
        """
        # Aplicar aceleración/frenado
        target_speed = throttle * self.max_speed
        if self.speed < target_speed:
            self.speed = min(self.speed + self.acceleration, target_speed)
        elif self.speed > target_speed:
            self.speed = max(self.speed - self.acceleration, target_speed)
        
        # Aplicar dirección solo si hay velocidad significativa
        if abs(self.speed) > 0.5:
            self.angle += steering * self.turn_speed
            
    def apply_physics(self):
        """Aplica física básica del auto"""
        # Guardar posición previa para detección de checkpoints
        self.prev_x = self.x
        self.prev_y = self.y
        
        # Fricción
        if self.speed > 0:
            self.speed = max(0, self.speed - self.friction)
        elif self.speed < 0:
            self.speed = min(0, self.speed + self.friction)
        
        # Actualizar posición basada en velocidad y ángulo
        rad = math.radians(self.angle)
        self.x += math.sin(rad) * self.speed
        self.y -= math.cos(rad) * self.speed
        
        # Actualizar distancia total
        self.total_distance += abs(self.speed)
        
    def get_corners(self):
        """Obtiene las coordenadas de las esquinas del auto para colisiones"""
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        # Esquinas relativas al centro
        corners_local = [
            (-self.width/2, -self.height/2),
            (self.width/2, -self.height/2),
            (self.width/2, self.height/2),
            (-self.width/2, self.height/2)
        ]
        
        # Rotar y trasladar esquinas
        corners = []
        for lx, ly in corners_local:
            rx = lx * cos_a - ly * sin_a + self.x
            ry = lx * sin_a + ly * cos_a + self.y
            corners.append((rx, ry))
            
        return corners
    
    def update_sensors(self, track):
        """
        Actualiza los sensores de distancia del auto
        
        Args:
            track: Objeto Track para detectar colisiones
        """
        for i, sensor_angle in enumerate(self.sensor_angles):
            # Ángulo absoluto del sensor
            angle = math.radians(self.angle + sensor_angle)
            
            # Buscar distancia hasta el borde
            distance = 0
            step = 5
            while distance < self.sensor_length:
                distance += step
                sx = self.x + math.cos(angle) * distance
                sy = self.y + math.sin(angle) * distance
                
                # Verificar si el punto está fuera de la pista
                if not track.is_on_track(sx, sy):
                    break
                    
            self.sensor_distances[i] = min(distance, self.sensor_length)
    
    def get_state_vector(self):
        """
        Obtiene el vector de estado del auto para los controladores IA
        
        Returns:
            numpy array con: [velocidad_normalizada, sensores_normalizados...]
        """
        # Normalizar velocidad entre -1 y 1
        speed_norm = self.speed / self.max_speed
        
        # Normalizar distancias de sensores entre 0 y 1
        sensors_norm = [d / self.sensor_length for d in self.sensor_distances]
        
        return np.array([speed_norm] + sensors_norm)
    
    def draw(self, screen):
        """Dibuja el auto en la pantalla"""
        if self.use_image and self.car_image:
            # Usar imagen cargada
            rotated = pygame.transform.rotate(self.car_image, -self.angle)
            rect = rotated.get_rect(center=(self.x, self.y))
            screen.blit(rotated, rect.topleft)
        else:
            # Dibujar forma del carro por defecto
            car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            
            # Colores para detalles
            dark_color = tuple(max(0, c - 50) for c in self.color)
            light_color = tuple(min(255, c + 30) for c in self.color)
            
            # Cuerpo principal del auto (forma más aerodinámica)
            body_points = [
                (self.width//2, 0),  # Frente puntiagudo
                (self.width - 5, 10),  # Lateral derecho superior
                (self.width, self.height//2),  # Lateral derecho medio
                (self.width - 5, self.height - 10),  # Lateral derecho inferior
                (self.width//2, self.height),  # Parte trasera
                (5, self.height - 10),  # Lateral izquierdo inferior
                (0, self.height//2),  # Lateral izquierdo medio
                (5, 10)  # Lateral izquierdo superior
            ]
            pygame.draw.polygon(car_surface, self.color, body_points)
            pygame.draw.polygon(car_surface, (0, 0, 0), body_points, 2)
            
            # Parabrisas delantero
            windshield_points = [
                (self.width//2, 8),
                (self.width - 10, 18),
                (self.width - 10, 28),
                (10, 28),
                (10, 18)
            ]
            pygame.draw.polygon(car_surface, (100, 150, 200), windshield_points)
            pygame.draw.polygon(car_surface, (0, 0, 0), windshield_points, 1)
            
            # Ventana trasera
            rear_window_points = [
                (self.width//2, self.height - 8),
                (self.width - 10, self.height - 18),
                (self.width - 10, self.height - 28),
                (10, self.height - 28),
                (10, self.height - 18)
            ]
            pygame.draw.polygon(car_surface, (100, 150, 200), rear_window_points)
            pygame.draw.polygon(car_surface, (0, 0, 0), rear_window_points, 1)
            
            # Luces delanteras
            pygame.draw.circle(car_surface, (255, 255, 150), (10, 5), 4)
            pygame.draw.circle(car_surface, (255, 255, 150), (self.width - 10, 5), 4)
            
            # Luces traseras
            pygame.draw.circle(car_surface, (255, 50, 50), (10, self.height - 5), 4)
            pygame.draw.circle(car_surface, (255, 50, 50), (self.width - 10, self.height - 5), 4)
            
            # Líneas decorativas laterales
            pygame.draw.line(car_surface, light_color, (8, 20), (8, self.height - 20), 2)
            pygame.draw.line(car_surface, light_color, (self.width - 8, 20), (self.width - 8, self.height - 20), 2)
            
            # Spoiler trasero (pequeño detalle)
            pygame.draw.rect(car_surface, dark_color, (5, self.height - 3, self.width - 10, 3))
            
            # Rotar el auto
            rotated = pygame.transform.rotate(car_surface, -self.angle)
            rect = rotated.get_rect(center=(self.x, self.y))
            
            # Dibujar en pantalla
            screen.blit(rotated, rect.topleft)
        
    def draw_sensors(self, screen):
        """Dibuja los sensores del auto (para debugging)"""
        for i, sensor_angle in enumerate(self.sensor_angles):
            angle = math.radians(self.angle + sensor_angle)
            distance = self.sensor_distances[i]
            
            end_x = self.x + math.cos(angle) * distance
            end_y = self.y + math.sin(angle) * distance
            
            # Color según distancia (rojo cerca, verde lejos)
            intensity = int(255 * (distance / self.sensor_length))
            color = (255 - intensity, intensity, 0)
            
            pygame.draw.line(screen, color, (self.x, self.y), (end_x, end_y), 1)
            pygame.draw.circle(screen, color, (int(end_x), int(end_y)), 3)
    
    def reset(self, x, y, angle=0):
        """Reinicia el auto a una posición inicial"""
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0
        self.lap_count = 0
        self.checkpoint_count = 0
        self.total_distance = 0
        self.crashed = False
