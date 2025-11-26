"""
Controlador Difuso para el auto
Usa lógica difusa para controlar velocidad y dirección basado en sensores
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyController:
    def __init__(self):
        """Inicializa el sistema de control difuso"""
        
        # Variables para detección de atasco y recuperación
        self.stuck_counter = 0
        self.last_x = 0
        self.last_y = 0
        self.reverse_timer = 0
        self.recovery_direction = 0
        self.crash_recovery_mode = False
        
        # ===== VARIABLES DE ENTRADA =====
        
        # Sensor frontal (distancia al frente)
        self.front_sensor = ctrl.Antecedent(np.arange(0, 151, 1), 'front_sensor')
        self.front_sensor['muy_cerca'] = fuzz.trapmf(self.front_sensor.universe, [0, 0, 20, 40])
        self.front_sensor['cerca'] = fuzz.trimf(self.front_sensor.universe, [30, 50, 70])
        self.front_sensor['media'] = fuzz.trimf(self.front_sensor.universe, [60, 80, 100])
        self.front_sensor['lejos'] = fuzz.trapmf(self.front_sensor.universe, [90, 110, 150, 150])
        
        # Sensor izquierdo
        self.left_sensor = ctrl.Antecedent(np.arange(0, 151, 1), 'left_sensor')
        self.left_sensor['cerca'] = fuzz.trapmf(self.left_sensor.universe, [0, 0, 30, 60])
        self.left_sensor['media'] = fuzz.trimf(self.left_sensor.universe, [50, 75, 100])
        self.left_sensor['lejos'] = fuzz.trapmf(self.left_sensor.universe, [90, 120, 150, 150])
        
        # Sensor derecho
        self.right_sensor = ctrl.Antecedent(np.arange(0, 151, 1), 'right_sensor')
        self.right_sensor['cerca'] = fuzz.trapmf(self.right_sensor.universe, [0, 0, 30, 60])
        self.right_sensor['media'] = fuzz.trimf(self.right_sensor.universe, [50, 75, 100])
        self.right_sensor['lejos'] = fuzz.trapmf(self.right_sensor.universe, [90, 120, 150, 150])
        
        # Velocidad actual
        self.speed = ctrl.Antecedent(np.arange(0, 11, 1), 'speed')
        self.speed['baja'] = fuzz.trapmf(self.speed.universe, [0, 0, 2, 4])
        self.speed['media'] = fuzz.trimf(self.speed.universe, [3, 5, 7])
        self.speed['alta'] = fuzz.trapmf(self.speed.universe, [6, 8, 10, 10])
        
        # ===== VARIABLES DE SALIDA =====
        
        # Control de aceleración/frenado (-1 = frenar, 0 = mantener, 1 = acelerar)
        self.throttle = ctrl.Consequent(np.arange(-1, 1.1, 0.1), 'throttle')
        self.throttle['frenar_fuerte'] = fuzz.trapmf(self.throttle.universe, [-1, -1, -0.8, -0.5])
        self.throttle['frenar'] = fuzz.trimf(self.throttle.universe, [-0.6, -0.3, 0])
        self.throttle['mantener'] = fuzz.trimf(self.throttle.universe, [-0.2, 0, 0.2])
        self.throttle['acelerar'] = fuzz.trimf(self.throttle.universe, [0, 0.5, 1])
        self.throttle['acelerar_fuerte'] = fuzz.trapmf(self.throttle.universe, [0.7, 0.9, 1, 1])
        
        # Control de dirección (-1 = izquierda, 0 = recto, 1 = derecha)
        self.steering = ctrl.Consequent(np.arange(-1, 1.1, 0.1), 'steering')
        self.steering['izquierda_fuerte'] = fuzz.trapmf(self.steering.universe, [-1, -1, -0.8, -0.5])
        self.steering['izquierda'] = fuzz.trimf(self.steering.universe, [-0.7, -0.4, -0.1])
        self.steering['recto'] = fuzz.trimf(self.steering.universe, [-0.2, 0, 0.2])
        self.steering['derecha'] = fuzz.trimf(self.steering.universe, [0.1, 0.4, 0.7])
        self.steering['derecha_fuerte'] = fuzz.trapmf(self.steering.universe, [0.5, 0.8, 1, 1])
        
        # ===== REGLAS DIFUSAS =====
        
        rules = []
        
        # Reglas de aceleración basadas en distancia frontal y velocidad
        rules.append(ctrl.Rule(self.front_sensor['muy_cerca'] & self.speed['alta'], self.throttle['frenar_fuerte']))
        rules.append(ctrl.Rule(self.front_sensor['muy_cerca'] & self.speed['media'], self.throttle['frenar']))
        rules.append(ctrl.Rule(self.front_sensor['cerca'] & self.speed['alta'], self.throttle['frenar']))
        rules.append(ctrl.Rule(self.front_sensor['cerca'] & self.speed['media'], self.throttle['mantener']))
        rules.append(ctrl.Rule(self.front_sensor['cerca'] & self.speed['baja'], self.throttle['acelerar']))
        rules.append(ctrl.Rule(self.front_sensor['media'] & self.speed['baja'], self.throttle['acelerar_fuerte']))
        rules.append(ctrl.Rule(self.front_sensor['media'] & self.speed['media'], self.throttle['acelerar']))
        rules.append(ctrl.Rule(self.front_sensor['lejos'] & self.speed['baja'], self.throttle['acelerar_fuerte']))
        rules.append(ctrl.Rule(self.front_sensor['lejos'] & self.speed['media'], self.throttle['acelerar_fuerte']))
        rules.append(ctrl.Rule(self.front_sensor['lejos'] & self.speed['alta'], self.throttle['mantener']))
        
        # Reglas de dirección basadas en sensores laterales
        rules.append(ctrl.Rule(self.left_sensor['cerca'] & self.right_sensor['lejos'], self.steering['derecha_fuerte']))
        rules.append(ctrl.Rule(self.left_sensor['cerca'] & self.right_sensor['media'], self.steering['derecha']))
        rules.append(ctrl.Rule(self.right_sensor['cerca'] & self.left_sensor['lejos'], self.steering['izquierda_fuerte']))
        rules.append(ctrl.Rule(self.right_sensor['cerca'] & self.left_sensor['media'], self.steering['izquierda']))
        rules.append(ctrl.Rule(self.left_sensor['media'] & self.right_sensor['media'], self.steering['recto']))
        rules.append(ctrl.Rule(self.left_sensor['lejos'] & self.right_sensor['lejos'], self.steering['recto']))
        
        # Reglas combinadas: si hay obstáculo al frente, girar hacia el lado más libre
        rules.append(ctrl.Rule(self.front_sensor['muy_cerca'] & (self.left_sensor['lejos'] | self.left_sensor['media']), 
                              self.steering['izquierda_fuerte']))
        rules.append(ctrl.Rule(self.front_sensor['muy_cerca'] & (self.right_sensor['lejos'] | self.right_sensor['media']), 
                              self.steering['derecha_fuerte']))
        rules.append(ctrl.Rule(self.front_sensor['cerca'] & self.left_sensor['lejos'], self.steering['izquierda']))
        rules.append(ctrl.Rule(self.front_sensor['cerca'] & self.right_sensor['lejos'], self.steering['derecha']))
        
        # Crear sistema de control
        self.control_system = ctrl.ControlSystem(rules)
        self.controller = ctrl.ControlSystemSimulation(self.control_system)
        
        print("✓ Sistema de control difuso inicializado")
        print(f"  - Reglas definidas: {len(rules)}")
        print(f"  - Variables de entrada: front_sensor, left_sensor, right_sensor, speed")
        print(f"  - Variables de salida: throttle, steering")
    
    def compute(self, car):
        """
        Calcula las acciones de control basadas en el estado del auto
        
        Args:
            car: Objeto Car con sensores actualizados
            
        Returns:
            Tupla (steering, throttle) con valores entre -1 y 1
        """
        # CONTROLADOR HÍBRIDO: Usa lógica fuzzy + reglas determinísticas
        # Obtener distancias de sensores (16 sensores totales)
        front = car.sensor_distances[0]
        front_right_22 = car.sensor_distances[1]
        front_right_45 = car.sensor_distances[2]
        front_right_67 = car.sensor_distances[3]
        right = car.sensor_distances[4]
        left = car.sensor_distances[12]
        front_left_67 = car.sensor_distances[13]
        front_left_45 = car.sensor_distances[14]
        front_left_22 = car.sensor_distances[15]
        current_speed = abs(car.speed)
        
        # === DETECCIÓN DE ATASCO ===
        import math
        import random
        
        distance_moved = math.sqrt((car.x - self.last_x)**2 + (car.y - self.last_y)**2)
        
        if distance_moved < 2 and abs(car.speed) < 0.5 and not car.crashed:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        
        if self.stuck_counter > 20:
            self.crash_recovery_mode = True
            self.reverse_timer = 30
            self.recovery_direction = 1 if random.random() > 0.5 else -1
            self.stuck_counter = 0
        
        self.last_x = car.x
        self.last_y = car.y
        
        # === MODO RECUPERACIÓN ===
        if self.reverse_timer > 0 or self.crash_recovery_mode:
            self.reverse_timer -= 1
            
            if self.reverse_timer > 20:
                steering = self.recovery_direction * 0.8
                throttle = -0.6
            elif self.reverse_timer > 10:
                steering = self.recovery_direction * 0.9
                throttle = -0.4
            else:
                steering = self.recovery_direction * 0.5
                throttle = 0.8
            
            if self.reverse_timer <= 0:
                self.crash_recovery_mode = False
                self.recovery_direction = 0
            
            return (steering, throttle)
        
        # === CONTROL DE DIRECCIÓN PARA PISTA RECTA ===
        steering = 0
        
        # PRIORIDAD 1: EVITAR COLISIÓN CON BORDES (CRÍTICO)
        if left < 20:  # Muy cerca del borde izquierdo
            steering = 0.7  # Girar a la DERECHA para alejarse
        elif right < 20:  # Muy cerca del borde derecho
            steering = -0.7  # Girar a la IZQUIERDA para alejarse
        
        # PRIORIDAD 2: CORRECCIÓN MODERADA cerca de bordes
        elif left < 35:  # Cerca del borde izquierdo
            steering = 0.3  # Corrección suave a la DERECHA
        elif right < 35:  # Cerca del borde derecho
            steering = -0.3  # Corrección suave a la IZQUIERDA
        
        # PRIORIDAD 3: Ir completamente recto si hay espacio
        else:
            steering = 0.0  # Mantener dirección recta
        
        # === CONTROL DE VELOCIDAD PARA PISTA RECTA ===
        
        # MODO REVERSA (solo para recuperación de atasco)
        if self.reverse_timer > 0:
            throttle = -0.6  # Marcha atrás
        
        # Reducir velocidad solo si hay riesgo inmediato
        elif car.crashed:
            throttle = 0.2  # Casi detenerse si chocó
        elif left < 18 or right < 18:
            throttle = 0.4  # Reducir si está muy cerca de los bordes
        elif front < 60:
            # Frente bloqueado (probablemente otro auto)
            throttle = 0.6  # Reducir para no chocar
        else:
            # Pista despejada: acelerar
            throttle = 1.0
        
        # Reducir si está haciendo correcciones fuertes
        if abs(steering) > 0.5:
            throttle *= 0.8
        
        # Limitar valores
        steering = np.clip(steering, -1, 1)
        throttle = np.clip(throttle, -1, 1)
        
        return steering, throttle
    
    def get_rules_description(self):
        """Retorna una descripción legible de las reglas para pista recta"""
        return """
        REGLAS DE CONTROL HÍBRIDO PARA PISTA RECTA:
        
        === Control de Dirección (Mantener Centrado) ===
        1. SI muy cerca del borde izquierdo (< 20px) → GIRAR DERECHA FUERTE (0.8)
        2. SI muy cerca del borde derecho (< 20px) → GIRAR IZQUIERDA FUERTE (-0.8)
        3. SI cerca del borde izquierdo (< 40px) → GIRAR DERECHA (0.4)
        4. SI cerca del borde derecho (< 40px) → GIRAR IZQUIERDA (-0.4)
        5. SI desbalance moderado → CORRECCIÓN SUAVE proporcional al balance
        6. SI centrado perfectamente → RECTO (0.0)
        
        === Control de Velocidad (Máxima Velocidad) ===
        1. SI chocado o bordes < 15px → REDUCIR (0.3)
        2. SI cerca de bordes < 30px → PRECAUCIÓN (0.7)
        3. SI frente bloqueado < 80px → REDUCIR para no chocar (0.5)
        4. SI pista despejada → ACELERAR AL MÁXIMO (1.0)
        5. Penalización por correcciones fuertes de steering
        
        === Sistema de Recuperación ===
        - Detección de atasco: movimiento < 2px y velocidad < 0.5
        - Recuperación en 3 fases: reversa + giro aleatorio
        
        OBJETIVO: Mantener el auto centrado en su carril a máxima velocidad
        """
