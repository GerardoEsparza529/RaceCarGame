"""
Script para entrenar la red neuronal
"""
import numpy as np
import pandas as pd
import glob
from data_generator import DataGenerator, split_data
from neural_controller import NeuralController
import matplotlib.pyplot as plt
import os

def load_real_data():
    """Carga datos reales capturados del modo manual"""
    training_files = glob.glob("training_data/training_data_*.csv")
    
    if not training_files:
        return None, None
    
    print(f"\n📂 Encontrados {len(training_files)} archivos de entrenamiento:")
    
    all_X = []
    all_y = []
    
    for filepath in training_files:
        print(f"   - {os.path.basename(filepath)}")
        df = pd.read_csv(filepath)
        
        # Extraer características (16 sensores + velocidad)
        X = df.iloc[:, :17].values  # Primeras 17 columnas
        
        # Extraer etiquetas (steering, throttle)
        y = df.iloc[:, 17:19].values  # Últimas 2 columnas
        
        all_X.append(X)
        all_y.append(y)
        
        print(f"      {len(X)} muestras cargadas")
    
    # Combinar todos los datos
    X = np.vstack(all_X)
    y = np.vstack(all_y)
    
    print(f"\n✓ Total: {len(X)} muestras de datos reales")
    
    return X, y

def train_neural_network(use_real_data=True, combine_with_synthetic=False):
    """Entrena la red neuronal con datos reales y/o sintéticos"""
    
    print("="*60)
    print("🧠 ENTRENAMIENTO DE RED NEURONAL")
    print("="*60)
    
    X_real, y_real = None, None
    X_synth, y_synth = None, None
    
    # Intentar cargar datos reales
    if use_real_data:
        X_real, y_real = load_real_data()
        
        if X_real is None:
            print("\n⚠ No se encontraron datos reales en training_data/")
            print("   Usa el modo manual y presiona [G] para grabar datos")
            
            if not combine_with_synthetic:
                print("\n❌ No hay datos disponibles para entrenar")
                return
    
    # Generar datos sintéticos si es necesario
    if combine_with_synthetic or X_real is None:
        print(f"\n📊 Generando datos sintéticos...")
        generator = DataGenerator()
        data_path = 'data/training_data.pkl'
        
        if os.path.exists(data_path):
            print(f"   Cargando datos sintéticos de {data_path}...")
            X_synth, y_synth = generator.load_training_data(data_path)
        else:
            X_synth, y_synth = generator.generate_training_data(num_samples=5000)
    
    # Combinar datos
    if X_real is not None and X_synth is not None:
        print("\n🔀 Combinando datos reales y sintéticos...")
        X = np.vstack([X_real, X_synth])
        y = np.vstack([y_real, y_synth])
        print(f"   Datos reales: {len(X_real)}")
        print(f"   Datos sintéticos: {len(X_synth)}")
        print(f"   Total: {len(X)}")
    elif X_real is not None:
        print("\n✓ Usando solo datos reales")
        X, y = X_real, y_real
    else:
        print("\n✓ Usando solo datos sintéticos")
        X, y = X_synth, y_synth
    
    if X is None or y is None:
        print("❌ Error: No se pudieron obtener datos de entrenamiento")
        return
    
    # Dividir datos
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, 
                                                                  train_ratio=0.8, 
                                                                  val_ratio=0.1)
    
    # Crear y entrenar modelo
    controller = NeuralController(model_path='models/neural_controller.h5')
    
    history = controller.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32
    )
    
    # Evaluar modelo
    print("\n📊 Evaluando modelo con datos de prueba...")
    controller.evaluate(X_test, y_test)
    
    # Graficar resultados del entrenamiento
    plot_training_history(history)
    
    # Análisis de predicciones
    analyze_predictions(controller, X_test, y_test)
    
    print("\n" + "="*60)
    print("✓ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"\n📁 Modelo guardado en: models/neural_controller.h5")
    print("📁 Gráficas guardadas en: models/")
    print("\n💡 Ahora puedes ejecutar el juego con: python main.py")

def plot_training_history(history):
    """Grafica la historia del entrenamiento"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Entrenamiento')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validación')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Pérdida durante el Entrenamiento')
    axes[0].legend()
    axes[0].grid(True)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Entrenamiento')
    if 'val_mae' in history.history:
        axes[1].plot(history.history['val_mae'], label='Validación')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Error Absoluto Medio')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # Guardar
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/training_history.png', dpi=150)
    print(f"✓ Gráfica guardada en models/training_history.png")
    
    # Mostrar (opcional)
    # plt.show()
    plt.close()

def analyze_predictions(controller, X_test, y_test, num_samples=1000):
    """Analiza las predicciones vs valores reales"""
    
    # Tomar muestra aleatoria
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]
    
    # Predecir
    y_pred = controller.model.predict(X_sample, verbose=0)
    
    # Graficar
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Steering
    axes[0].scatter(y_sample[:, 0], y_pred[:, 0], alpha=0.5, s=10)
    axes[0].plot([-1, 1], [-1, 1], 'r--', label='Ideal')
    axes[0].set_xlabel('Steering Real')
    axes[0].set_ylabel('Steering Predicho')
    axes[0].set_title('Predicción de Dirección (Steering)')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].axis('equal')
    
    # Throttle
    axes[1].scatter(y_sample[:, 1], y_pred[:, 1], alpha=0.5, s=10)
    axes[1].plot([-1, 1], [-1, 1], 'r--', label='Ideal')
    axes[1].set_xlabel('Throttle Real')
    axes[1].set_ylabel('Throttle Predicho')
    axes[1].set_title('Predicción de Aceleración (Throttle)')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].axis('equal')
    
    plt.tight_layout()
    plt.savefig('models/predictions_analysis.png', dpi=150)
    print(f"✓ Análisis de predicciones guardado en models/predictions_analysis.png")
    
    plt.close()
    
    # Estadísticas de error
    errors_steering = np.abs(y_sample[:, 0] - y_pred[:, 0])
    errors_throttle = np.abs(y_sample[:, 1] - y_pred[:, 1])
    
    print(f"\n📊 Estadísticas de Error (sobre {num_samples} muestras):")
    print(f"   Steering - MAE: {errors_steering.mean():.4f}, Max: {errors_steering.max():.4f}")
    print(f"   Throttle - MAE: {errors_throttle.mean():.4f}, Max: {errors_throttle.max():.4f}")

if __name__ == "__main__":
    import sys
    
    # Configuración por defecto
    use_real_data = True
    combine_with_synthetic = False
    
    # Argumentos de línea de comandos
    if len(sys.argv) > 1:
        if '--synthetic-only' in sys.argv:
            use_real_data = False
            print("🔧 Modo: Solo datos sintéticos")
        elif '--combined' in sys.argv:
            combine_with_synthetic = True
            print("🔧 Modo: Datos reales + sintéticos")
        else:
            print("🔧 Modo: Solo datos reales (predeterminado)")
    
    print("\n💡 Opciones disponibles:")
    print("   python train_network.py              # Solo datos reales")
    print("   python train_network.py --combined   # Reales + sintéticos")
    print("   python train_network.py --synthetic-only  # Solo sintéticos")
    print()
    
    train_neural_network(use_real_data=use_real_data, 
                        combine_with_synthetic=combine_with_synthetic)
