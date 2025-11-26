"""
Clase Game - Gestiona el juego principal
"""
import pygame
import sys
from car import Car
from track import Track
from fuzzy_controller import FuzzyController
from neural_controller import NeuralController
from opponent_controller import OpponentController
from data_collector import DataCollector

class Game:
    def __init__(self):
        """Inicializa el juego"""
        pygame.init()
        
        # Configuración de ventana
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Carrera de Autos IA - Proyecto")
        
        # Reloj y FPS
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # Fuentes
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Estado del juego
        self.state = 'menu'  # 'menu', 'playing', 'finished'
        self.control_mode = None  # 'manual', 'fuzzy', 'neural'
        
        # Crear objetos del juego
        self.track = Track(self.width, self.height)
        
        # Autos
        self.player_car = None
        self.opponent_car = None
        
        # Controladores
        self.fuzzy_controller = None
        self.neural_controller = None
        
        # Sistema de niveles progresivos
        self.current_level = 1  # Nivel actual (1, 2, 3)
        self.max_level = 3
        self.opponent_controller = self.create_opponent_for_level(1)
        
        # Colector de datos para entrenamiento
        self.data_collector = DataCollector()
        
        # Variables de juego
        self.winner = None
        self.game_time = 0
        self.show_sensors = False
        
        # Colores
        self.COLOR_PLAYER = (0, 120, 255)
        self.COLOR_OPPONENT = (255, 80, 80)
    
    def create_opponent_for_level(self, level):
        """Crea un oponente con dificultad según el nivel"""
        if level == 1:
            return OpponentController(difficulty='easy')
        elif level == 2:
            return OpponentController(difficulty='medium')
        else:  # level 3
            return OpponentController(difficulty='hard')
    
    def advance_level(self):
        """Avanza al siguiente nivel si es posible"""
        if self.current_level < self.max_level:
            self.current_level += 1
            self.opponent_controller = self.create_opponent_for_level(self.current_level)
            return True
        return False
        
    def reset_race(self):
        """Reinicia la carrera"""
        # Obtener posiciones de inicio desde la pista
        player_pos = self.track.get_start_position(lane=0)
        opponent_pos = self.track.get_start_position(lane=1)
        
        # Auto del jugador (lane 0)
        self.player_car = Car(player_pos[0], player_pos[1], self.COLOR_PLAYER, is_player=True)
        self.player_car.angle = player_pos[2]
        
        # Auto oponente (lane 1)
        self.opponent_car = Car(opponent_pos[0], opponent_pos[1], self.COLOR_OPPONENT, is_player=False)
        self.opponent_car.angle = opponent_pos[2]
        
        # Inicializar controladores si es necesario
        if self.control_mode == 'fuzzy' and self.fuzzy_controller is None:
            self.fuzzy_controller = FuzzyController()
        
        if self.control_mode == 'neural' and self.neural_controller is None:
            self.neural_controller = NeuralController()
            if not self.neural_controller.is_trained:
                print("⚠ Red neuronal no entrenada. Ejecuta train_network.py primero.")
        
        # Reiniciar variables
        self.winner = None
        self.game_time = 0
        self.state = 'playing'
        
        # Iniciar grabación automática en modo manual
        if self.control_mode == 'manual':
            if not self.data_collector.is_recording:
                self.data_collector.start_recording()
                print("📹 Grabación automática iniciada")
        
    def run(self):
        """Loop principal del juego"""
        running = True
        
        while running:
            # Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self.handle_keydown(event.key)
            
            # Actualizar
            if self.state == 'menu':
                self.update_menu()
            elif self.state == 'playing':
                self.update_game()
            elif self.state == 'level_complete':
                self.update_level_complete()
            elif self.state == 'finished':
                self.update_finished()
            
            # Dibujar
            self.draw()
            
            # Control de FPS
            self.clock.tick(self.fps)
        
        pygame.quit()
        sys.exit()
    
    def handle_keydown(self, key):
        """Maneja eventos de teclado"""
        if self.state == 'menu':
            if key == pygame.K_1:
                self.control_mode = 'manual'
                self.reset_race()
            elif key == pygame.K_2:
                self.control_mode = 'fuzzy'
                self.reset_race()
            elif key == pygame.K_3:
                self.control_mode = 'neural'
                self.reset_race()
            elif key == pygame.K_ESCAPE:
                return False
        
        elif self.state == 'playing':
            if key == pygame.K_ESCAPE:
                self.state = 'menu'
            elif key == pygame.K_s:
                self.show_sensors = not self.show_sensors
            elif key == pygame.K_r:
                self.reset_race()
            elif key == pygame.K_g and self.control_mode == 'manual':
                # Toggle grabación en modo manual
                if self.data_collector.is_recording:
                    self.data_collector.stop_recording()
                else:
                    self.data_collector.start_recording()
        
        elif self.state == 'level_complete':
            if key == pygame.K_SPACE or key == pygame.K_RETURN:
                self.reset_race()  # Continuar al siguiente nivel
        
        elif self.state == 'finished':
            if key == pygame.K_SPACE or key == pygame.K_RETURN:
                # Reiniciar desde el nivel 1
                self.current_level = 1
                self.opponent_controller = self.create_opponent_for_level(1)
                self.reset_race()
            elif key == pygame.K_ESCAPE:
                self.state = 'menu'
                self.current_level = 1  # Reset al volver al menú
                self.opponent_controller = self.create_opponent_for_level(1)
        
        return True
    
    def update_menu(self):
        """Actualiza el menú"""
        pass  # El menú es estático
    
    def update_game(self):
        """Actualiza el estado del juego"""
        keys = pygame.key.get_pressed()
        
        # === ACTUALIZAR AUTO DEL JUGADOR ===
        steering = 0
        throttle = 0
        
        if self.control_mode == 'manual':
            # Actualizar sensores para captura de datos
            self.player_car.update_sensors(self.track)
            
            # Control manual
            self.player_car.update_manual(keys)
            
            # Calcular steering y throttle equivalentes para grabación
            if keys[pygame.K_LEFT]:
                steering = -1.0
            elif keys[pygame.K_RIGHT]:
                steering = 1.0
            
            if keys[pygame.K_UP]:
                throttle = 1.0
            elif keys[pygame.K_DOWN]:
                throttle = -0.5
            
            # Grabar datos si está activo
            self.data_collector.record_frame(self.player_car, steering, throttle)
            
        elif self.control_mode == 'fuzzy':
            self.player_car.update_sensors(self.track)
            steering, throttle = self.fuzzy_controller.compute(self.player_car)
            self.player_car.update_ai_control(steering, throttle)
        elif self.control_mode == 'neural':
            self.player_car.update_sensors(self.track)
            steering, throttle = self.neural_controller.compute(self.player_car)
            self.player_car.update_ai_control(steering, throttle)
        
        self.player_car.apply_physics()
        
        # === ACTUALIZAR AUTO OPONENTE ===
        self.opponent_car.update_sensors(self.track)
        
        # El oponente (auto rojo) SIEMPRE usa el OpponentController simple
        # que solo avanza recto a velocidad constante
        steering, throttle = self.opponent_controller.compute(self.opponent_car)
        
        self.opponent_car.update_ai_control(steering, throttle)
        self.opponent_car.apply_physics()
        
        # === VERIFICAR COLISIONES ===
        if self.track.check_collision(self.player_car):
            self.player_car.crashed = True
            self.player_car.speed *= 0.5  # Ralentizar
        
        if self.track.check_collision(self.opponent_car):
            self.opponent_car.crashed = True
            self.opponent_car.speed *= 0.5
        
        # === VERIFICAR CHECKPOINTS ===
        self.check_progress(self.player_car)
        self.check_progress(self.opponent_car)
        
        # === VERIFICAR CONDICIONES DE VICTORIA (llegó a la meta) ===
        if self.player_car.y >= self.track.finish_line_y and not self.winner:
            self.winner = 'player'
            
            # Guardar grabación automática en modo manual
            if self.control_mode == 'manual' and self.data_collector.is_recording:
                self.data_collector.stop_recording()
                print("💾 Grabación guardada automáticamente")
            
            # Avanzar de nivel si ganó
            if self.advance_level():
                self.state = 'level_complete'
                print(f"🏁 ¡NIVEL {self.current_level - 1} COMPLETADO! Avanzando a Nivel {self.current_level}")
            else:
                self.state = 'finished'  # Completó todos los niveles
                print("🏆 ¡FELICITACIONES! ¡COMPLETASTE TODOS LOS NIVELES!")
        elif self.opponent_car.y >= self.track.finish_line_y and not self.winner:
            self.winner = 'opponent'
            
            # Guardar grabación incluso si perdió (datos útiles)
            if self.control_mode == 'manual' and self.data_collector.is_recording:
                self.data_collector.stop_recording()
                print("💾 Grabación guardada (carrera perdida)")
            
            self.state = 'finished'  # Perdió, no avanza de nivel
            print("🏁 ¡OPONENTE GANÓ! Intenta nuevamente")
        
        # Actualizar tiempo
        self.game_time += 1 / self.fps
    
    def check_progress(self, car):
        """Verifica el progreso de un auto (checkpoints)"""
        # Verificar checkpoint actual con posición previa
        prev_pos = (car.prev_x, car.prev_y)
        if self.track.check_checkpoint(car, car.checkpoint_count, prev_pos):
            car.checkpoint_count += 1
            progress_percent = (car.checkpoint_count / len(self.track.checkpoints)) * 100
            print(f"{'🔵 Jugador' if car.is_player else '🔴 Oponente'} - Checkpoint {car.checkpoint_count}/{len(self.track.checkpoints)} ({progress_percent:.0f}%)")
    
    def update_level_complete(self):
        """Actualiza pantalla de nivel completado"""
        pass  # La pantalla es estática
    
    def update_finished(self):
        """Actualiza pantalla de finalización"""
        pass  # La pantalla de fin es estática
    
    def draw(self):
        """Dibuja todo en la pantalla"""
        if self.state == 'menu':
            self.draw_menu()
        elif self.state == 'playing':
            self.draw_game()
        elif self.state == 'level_complete':
            self.draw_level_complete()
        elif self.state == 'finished':
            self.draw_finished()
        
        pygame.display.flip()
    
    def draw_menu(self):
        """Dibuja el menú principal"""
        self.screen.fill((20, 20, 40))
        
        # Título
        title = self.font_large.render("CARRERA DE AUTOS IA", True, (255, 255, 255))
        title_rect = title.get_rect(center=(self.width // 2, 100))
        self.screen.blit(title, title_rect)
        
        # Subtítulo
        subtitle = self.font_small.render("Proyecto de Control Inteligente", True, (200, 200, 200))
        subtitle_rect = subtitle.get_rect(center=(self.width // 2, 150))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Opciones
        options_y = 250
        options = [
            "1 - Control MANUAL (Flechas del teclado)",
            "2 - Control DIFUSO (Lógica Fuzzy)",
            "3 - Control RED NEURONAL (Deep Learning)",
            "",
            "ESC - Salir"
        ]
        
        for i, option in enumerate(options):
            color = (255, 255, 100) if option and option[0].isdigit() else (180, 180, 180)
            text = self.font_medium.render(option, True, color)
            text_rect = text.get_rect(center=(self.width // 2, options_y + i * 60))
            self.screen.blit(text, text_rect)
        
        # Información adicional
        info_y = 600
        info_texts = [
            "OBJETIVO: Llega a la META antes que tu oponente",
            "Durante el juego: [S] Mostrar sensores | [R] Reiniciar | [ESC] Menú",
            "Modo Manual: [G] Grabar datos para entrenar IA"
        ]
        
        for i, info in enumerate(info_texts):
            text = self.font_small.render(info, True, (150, 150, 150))
            text_rect = text.get_rect(center=(self.width // 2, info_y + i * 30))
            self.screen.blit(text, text_rect)
    
    def draw_game(self):
        """Dibuja el juego en ejecución"""
        # Dibujar pista
        self.track.draw(self.screen)
        
        # Dibujar sensores si está activado
        if self.show_sensors:
            self.player_car.draw_sensors(self.screen)
            self.opponent_car.draw_sensors(self.screen)
        
        # Dibujar autos
        self.player_car.draw(self.screen)
        self.opponent_car.draw(self.screen)
        
        # Dibujar HUD
        self.draw_hud()
    
    def draw_hud(self):
        """Dibuja el HUD (información en pantalla)"""
        # Panel semi-transparente
        hud_surface = pygame.Surface((self.width, 120))
        hud_surface.set_alpha(180)
        hud_surface.fill((20, 20, 40))
        self.screen.blit(hud_surface, (0, 0))
        
        # Información del jugador (izquierda)
        y_offset = 10
        
        # Nivel actual
        level_text = self.font_large.render(f"NIVEL {self.current_level}/{self.max_level}", True, (255, 215, 0))
        level_rect = level_text.get_rect(center=(self.width // 2, 25))
        self.screen.blit(level_text, level_rect)
        
        # Modo de control
        mode_names = {
            'manual': 'MANUAL',
            'fuzzy': 'DIFUSO',
            'neural': 'RED NEURONAL'
        }
        mode_text = self.font_medium.render(f"Modo: {mode_names[self.control_mode]}", True, self.COLOR_PLAYER)
        self.screen.blit(mode_text, (20, y_offset))
        
        # Indicador de grabación en modo manual
        if self.control_mode == 'manual' and self.data_collector.is_recording:
            rec_text = self.font_small.render("● REC", True, (255, 50, 50))
            self.screen.blit(rec_text, (220, y_offset + 8))
        
        # Progreso del jugador
        player_progress = (self.player_car.checkpoint_count / len(self.track.checkpoints)) * 100 if len(self.track.checkpoints) > 0 else 0
        progress_text = self.font_small.render(f"Progreso: {player_progress:.0f}%", True, (255, 255, 255))
        self.screen.blit(progress_text, (20, y_offset + 40))
        
        # Velocidad del jugador
        speed_text = self.font_small.render(f"Velocidad: {abs(self.player_car.speed):.1f}", True, (255, 255, 255))
        self.screen.blit(speed_text, (20, y_offset + 65))
        
        # Estado
        if self.player_car.crashed:
            crash_text = self.font_small.render("¡COLISIÓN!", True, (255, 100, 100))
            self.screen.blit(crash_text, (20, y_offset + 90))
        
        # Información del oponente (derecha)
        opp_title = self.font_medium.render("Oponente", True, self.COLOR_OPPONENT)
        opp_rect = opp_title.get_rect(topright=(self.width - 20, y_offset))
        self.screen.blit(opp_title, opp_rect)
        
        opp_progress = (self.opponent_car.checkpoint_count / len(self.track.checkpoints)) * 100 if len(self.track.checkpoints) > 0 else 0
        opp_progress_text = self.font_small.render(f"Progreso: {opp_progress:.0f}%", True, (255, 255, 255))
        opp_progress_rect = opp_progress_text.get_rect(topright=(self.width - 20, y_offset + 40))
        self.screen.blit(opp_progress_text, opp_progress_rect)
        
        opp_speed = self.font_small.render(f"Velocidad: {abs(self.opponent_car.speed):.1f}", True, (255, 255, 255))
        opp_speed_rect = opp_speed.get_rect(topright=(self.width - 20, y_offset + 65))
        self.screen.blit(opp_speed, opp_speed_rect)
        
        # Tiempo (centro)
        time_text = self.font_medium.render(f"Tiempo: {self.game_time:.1f}s", True, (255, 255, 100))
        time_rect = time_text.get_rect(center=(self.width // 2, 30))
        self.screen.blit(time_text, time_rect)
        
        # Estado de grabación (solo en modo manual)
        if self.control_mode == 'manual':
            recording_status = self.data_collector.get_status()
            color = (255, 80, 80) if self.data_collector.is_recording else (150, 150, 150)
            rec_text = self.font_small.render(recording_status, True, color)
            rec_rect = rec_text.get_rect(center=(self.width // 2, 85))
            self.screen.blit(rec_text, rec_rect)
    
    def draw_level_complete(self):
        """Dibuja la pantalla de nivel completado"""
        # Dibujar juego de fondo
        self.draw_game()
        
        # Overlay semi-transparente
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(200)
        overlay.fill((0, 50, 0))  # Verde oscuro
        self.screen.blit(overlay, (0, 0))
        
        # Mensaje de nivel completado
        message = f"¡NIVEL {self.current_level - 1} COMPLETADO!"
        title = self.font_large.render(message, True, (100, 255, 100))
        title_rect = title.get_rect(center=(self.width // 2, self.height // 2 - 120))
        self.screen.blit(title, title_rect)
        
        # Siguiente nivel
        next_level_text = f"SIGUIENTE: NIVEL {self.current_level}"
        next_level = self.font_large.render(next_level_text, True, (255, 215, 0))
        next_level_rect = next_level.get_rect(center=(self.width // 2, self.height // 2 - 50))
        self.screen.blit(next_level, next_level_rect)
        
        # Descripción de dificultad
        difficulties = {1: "FÁCIL", 2: "MEDIO", 3: "DIFÍCIL"}
        diff_text = f"Dificultad: {difficulties[self.current_level]}"
        diff = self.font_medium.render(diff_text, True, (255, 255, 255))
        diff_rect = diff.get_rect(center=(self.width // 2, self.height // 2 + 20))
        self.screen.blit(diff, diff_rect)
        
        # Instrucción
        instruction = "PRESIONA ESPACIO PARA CONTINUAR"
        inst_text = self.font_medium.render(instruction, True, (255, 255, 255))
        inst_rect = inst_text.get_rect(center=(self.width // 2, self.height // 2 + 100))
        self.screen.blit(inst_text, inst_rect)
    
    def draw_finished(self):
        """Dibuja la pantalla de finalización"""
        # Dibujar juego de fondo
        self.draw_game()
        
        # Overlay semi-transparente
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Mensaje de victoria/derrota
        if self.winner == 'player':
            message = "🏆 ¡COMPLETASTE TODOS LOS NIVELES! 🏆"
            color = (255, 215, 0)  # Dorado
        else:
            message = "DERROTA"
            color = (255, 100, 100)
        
        title = self.font_large.render(message, True, color)
        title_rect = title.get_rect(center=(self.width // 2, self.height // 2 - 100))
        self.screen.blit(title, title_rect)
        
        # Estadísticas
        if self.winner == 'player':
            stats = [
                f"¡FELICITACIONES!",
                f"Completaste los 3 niveles en modo {self.control_mode.upper()}",
                "",
                "ESPACIO - Jugar de nuevo (Nivel 1)",
                "ESC - Menú principal"
            ]
        else:
            stats = [
                f"Nivel alcanzado: {self.current_level}",
                f"Modo de control: {self.control_mode.upper()}",
                "",
                "ESPACIO - Reintentar (Nivel 1)",
                "ESC - Menú principal"
            ]
        
        y_offset = self.height // 2
        for i, stat in enumerate(stats):
            text = self.font_medium.render(stat, True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.width // 2, y_offset + i * 50))
            self.screen.blit(text, text_rect)

if __name__ == "__main__":
    game = Game()
    game.run()
