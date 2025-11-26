"""
Clase Track - Representa la pista de carreras recta de 2 carriles
"""
import pygame
import math

class Track:
    def __init__(self, width, height):
        """
        Inicializa la pista recta de 2 carriles
        
        Args:
            width: Ancho de la ventana
            height: Alto de la ventana
        """
        self.width = width
        self.height = height
        
        # Pista recta vertical de 2 carriles
        self.track_width = 200  # Ancho total de la pista (2 carriles)
        self.lane_width = self.track_width // 2  # Ancho de cada carril
        self.track_length = height - 100  # Largo de la pista
        
        # Posición de la pista (centrada horizontalmente)
        self.track_x = (width - self.track_width) // 2
        self.track_y = 50
        
        # Colores
        self.grass_color = (34, 139, 34)
        self.track_color = (70, 70, 70)
        self.line_color = (255, 255, 255)
        self.finish_line_color = (255, 215, 0)  # Dorado
        
        # Línea de inicio (parte superior)
        self.start_line_y = self.track_y + 30
        
        # Línea de meta (parte inferior)
        self.finish_line_y = self.track_y + self.track_length - 30
        
        # Checkpoints para detectar progreso
        self.checkpoints = self.create_checkpoints()
        
    def create_checkpoints(self):
        """Crea puntos de control en la pista recta"""
        checkpoints = []
        
        # Dividir la pista en 5 checkpoints intermedios + 1 meta
        num_checkpoints = 5
        segment_length = (self.finish_line_y - self.start_line_y) / num_checkpoints
        
        x1 = self.track_x
        x2 = self.track_x + self.track_width
        
        # Crear checkpoints intermedios
        for i in range(1, num_checkpoints + 1):
            y = self.start_line_y + (segment_length * i)
            checkpoints.append((x1, y, x2, y, 'horizontal'))
        
        return checkpoints
    
    def is_on_track(self, x, y):
        """
        Verifica si un punto está sobre la pista
        
        Args:
            x: Coordenada X
            y: Coordenada Y
            
        Returns:
            True si está en la pista, False si no
        """
        # Verificar si está dentro de los límites de la pista recta
        in_x = self.track_x <= x <= self.track_x + self.track_width
        in_y = self.track_y <= y <= self.track_y + self.track_length
        
        return in_x and in_y
    
    def check_collision(self, car):
        """
        Verifica si un auto colisionó con el borde de la pista
        
        Args:
            car: Objeto Car a verificar
            
        Returns:
            True si hay colisión, False si no
        """
        corners = car.get_corners()
        
        # Verificar cada esquina del auto
        for corner in corners:
            if not self.is_on_track(corner[0], corner[1]):
                return True
                
        return False
    
    def check_checkpoint(self, car, checkpoint_index, prev_position=None):
        """
        Verifica si un auto pasó por un checkpoint (línea) con detección de cruce
        
        Args:
            car: Objeto Car a verificar
            checkpoint_index: Índice del checkpoint a verificar
            prev_position: Posición anterior (x, y) para detectar cruce
            
        Returns:
            True si pasó el checkpoint, False si no
        """
        if checkpoint_index >= len(self.checkpoints):
            return False
            
        x1, y1, x2, y2, orientation = self.checkpoints[checkpoint_index]
        
        # Margen de tolerancia más amplio
        tolerance = 25
        
        # Posición actual
        curr_x, curr_y = car.x, car.y
        
        if orientation == 'horizontal':
            # Verificar si está dentro del rango X
            if min(x1, x2) - tolerance <= curr_x <= max(x1, x2) + tolerance:
                # Verificar cruce de la línea Y
                if prev_position:
                    prev_x, prev_y = prev_position
                    # Detectar si cruzó la línea (cambio de lado)
                    if (prev_y < y1 - tolerance and curr_y >= y1 - tolerance) or \
                       (prev_y > y1 + tolerance and curr_y <= y1 + tolerance):
                        return True
                else:
                    # Sin posición previa, usar proximidad simple
                    if abs(curr_y - y1) < tolerance:
                        return True
        else:  # vertical
            # Verificar si está dentro del rango Y
            if min(y1, y2) - tolerance <= curr_y <= max(y1, y2) + tolerance:
                # Verificar cruce de la línea X
                if prev_position:
                    prev_x, prev_y = prev_position
                    # Detectar si cruzó la línea (cambio de lado)
                    if (prev_x < x1 - tolerance and curr_x >= x1 - tolerance) or \
                       (prev_x > x1 + tolerance and curr_x <= x1 + tolerance):
                        return True
                else:
                    # Sin posición previa, usar proximidad simple
                    if abs(curr_x - x1) < tolerance:
                        return True
        
        return False
    
    def get_start_position(self, lane=0):
        """
        Obtiene la posición inicial para un auto
        
        Args:
            lane: Carril (0 para jugador izquierdo, 1 para oponente derecho)
            
        Returns:
            Tupla (x, y, angle)
        """
        # X: Centrado en cada carril
        x = self.track_x + (self.lane_width // 2) + (lane * self.lane_width)
        
        # Y: Justo después de la línea de inicio
        y = self.start_line_y + 10
        
        # Ángulo: 180 grados = mirando hacia abajo (sur)
        angle = 180
        
        return (x, y, angle)
    
    def draw(self, screen):
        """Dibuja la pista recta en la pantalla"""
        # Fondo de pasto
        screen.fill(self.grass_color)
        
        # Pista principal
        track_rect = pygame.Rect(self.track_x, self.track_y, self.track_width, self.track_length)
        pygame.draw.rect(screen, self.track_color, track_rect)
        
        # Bordes de la pista (blancos)
        pygame.draw.rect(screen, self.line_color, track_rect, 4)
        
        # Línea central divisoria de carriles (discontinua)
        dash_length = 30
        gap_length = 20
        center_x = self.track_x + self.lane_width
        
        y = self.track_y
        while y < self.track_y + self.track_length:
            end_y = min(y + dash_length, self.track_y + self.track_length)
            pygame.draw.line(screen, self.line_color, (center_x, y), (center_x, end_y), 3)
            y += dash_length + gap_length
        
        # Línea de INICIO (patrón de cuadros blanco/negro)
        checker_size = 10
        for i in range(self.track_width // checker_size):
            for j in range(4):  # 4 filas de cuadros
                x = self.track_x + (i * checker_size)
                y = self.start_line_y - 20 + (j * checker_size)
                color = (255, 255, 255) if (i + j) % 2 == 0 else (0, 0, 0)
                pygame.draw.rect(screen, color, (x, y, checker_size, checker_size))
        
        # Texto "INICIO"
        font_start = pygame.font.Font(None, 32)
        text_start = font_start.render("INICIO", True, (255, 255, 255))
        text_rect = text_start.get_rect(center=(self.track_x + self.track_width // 2, self.start_line_y - 40))
        # Fondo oscuro
        bg_rect = text_rect.inflate(20, 10)
        pygame.draw.rect(screen, (50, 50, 50), bg_rect)
        pygame.draw.rect(screen, (255, 255, 255), bg_rect, 2)
        screen.blit(text_start, text_rect)
        
        # Línea de META (patrón de cuadros blanco/negro - más grande y dorado)
        checker_size_finish = 15
        for i in range(self.track_width // checker_size_finish + 1):
            for j in range(4):  # 4 filas de cuadros
                x = self.track_x + (i * checker_size_finish)
                y = self.finish_line_y - 30 + (j * checker_size_finish)
                if x < self.track_x + self.track_width:
                    color = self.finish_line_color if (i + j) % 2 == 0 else (0, 0, 0)
                    pygame.draw.rect(screen, color, (x, y, checker_size_finish, checker_size_finish))
        
        # Texto "META"
        font_finish = pygame.font.Font(None, 48)
        text_finish = font_finish.render("META", True, self.finish_line_color)
        text_rect_finish = text_finish.get_rect(center=(self.track_x + self.track_width // 2, self.finish_line_y + 40))
        # Fondo oscuro
        bg_rect_finish = text_rect_finish.inflate(30, 15)
        pygame.draw.rect(screen, (50, 50, 50), bg_rect_finish)
        pygame.draw.rect(screen, self.finish_line_color, bg_rect_finish, 3)
        screen.blit(text_finish, text_rect_finish)
        
        # Flechas indicadoras de dirección (hacia abajo)
        arrow_color = (255, 255, 100)
        arrow_size = 30
        
        # Dibujar 3 flechas espaciadas
        for i in range(3):
            y_arrow = self.track_y + 150 + (i * 150)
            if y_arrow < self.finish_line_y - 100:
                # Flecha en carril izquierdo
                self.draw_arrow(screen, self.track_x + self.lane_width // 2, y_arrow, 180, arrow_size, arrow_color)
                # Flecha en carril derecho
                self.draw_arrow(screen, self.track_x + self.lane_width + self.lane_width // 2, y_arrow, 180, arrow_size, arrow_color)
        
        # Checkpoints visibles (opcionales, para debug)
        # for i, (x1, y1, x2, y2, orientation) in enumerate(self.checkpoints):
        #     pygame.draw.line(screen, (100, 255, 100), (int(x1), int(y1)), (int(x2), int(y2)), 2)
    
    def draw_arrow(self, screen, x, y, angle, size, color):
        """
        Dibuja una flecha indicadora de dirección
        
        Args:
            screen: Superficie de pygame
            x, y: Posición central de la flecha
            angle: Ángulo en grados (0=arriba, 90=derecha, 180=abajo, 270=izquierda)
            size: Tamaño de la flecha
            color: Color de la flecha
        """
        # Definir puntos de la flecha (apuntando hacia arriba)
        arrow_points = [
            (0, -size),      # Punta
            (-size//2, 0),   # Izquierda base
            (-size//4, 0),   # Izquierda cuello
            (-size//4, size//2),  # Izquierda cola
            (size//4, size//2),   # Derecha cola
            (size//4, 0),    # Derecha cuello
            (size//2, 0)     # Derecha base
        ]
        
        # Rotar y trasladar puntos
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        rotated_points = []
        for px, py in arrow_points:
            rx = px * cos_a - py * sin_a + x
            ry = px * sin_a + py * cos_a + y
            rotated_points.append((rx, ry))
        
        # Dibujar flecha con borde
        pygame.draw.polygon(screen, color, rotated_points)
        pygame.draw.polygon(screen, (0, 0, 0), rotated_points, 2)
