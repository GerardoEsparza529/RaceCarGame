"""
Controlador Simple para el Auto Oponente (Auto Rojo - CPU)
Avanza recto a velocidad moderada con corrección mínima para mantenerse centrado
"""

class OpponentController:
    def __init__(self, difficulty='medium'):
        """
        Inicializa el controlador del oponente (CPU)
        
        Args:
            difficulty: Nivel de dificultad ('easy', 'medium', 'hard')
        """
        self.difficulty = difficulty
        
        # Velocidad según dificultad
        if difficulty == 'easy':
            self.target_throttle = 0.35  # 35% velocidad
        elif difficulty == 'medium':
            self.target_throttle = 0.5  # 50% velocidad
        else:  # hard
            self.target_throttle = 0.65  # 65% velocidad
        
    def compute(self, car):
        """
        Controlador simple: avanza completamente recto sin corrección
        El auto rojo (CPU) solo va hacia adelante en línea recta
        
        Args:
            car: Objeto Car del oponente
            
        Returns:
            Tupla (steering, throttle)
        """
        
        # Sin corrección de dirección - completamente recto
        steering = 0
        
        # Velocidad constante según dificultad
        throttle = self.target_throttle
        
        return steering, throttle
