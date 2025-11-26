"""
Main - Punto de entrada del juego
"""
from game import Game

def main():
    """Función principal"""
    print("="*60)
    print("  CARRERA DE AUTOS CON IA")
    print("  Proyecto de Control Inteligente")
    print("="*60)
    print("\nIniciando juego...\n")
    
    # Crear y ejecutar juego
    game = Game()
    game.run()

if __name__ == "__main__":
    main()
