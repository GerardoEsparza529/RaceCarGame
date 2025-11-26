"""
Controlador de Red Neuronal para el auto
Usa una red neuronal entrenada con datos del controlador difuso
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os

class NeuralController:
    def __init__(self, model_path='models/neural_controller.h5'):
        """
        Inicializa el controlador de red neuronal
        
        Args:
            model_path: Ruta al archivo del modelo entrenado
        """
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        
        # Intentar cargar modelo existente
        if os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path)
                self.is_trained = True
                print(f"✓ Modelo de red neuronal cargado desde {model_path}")
            except Exception as e:
                print(f"⚠ No se pudo cargar el modelo: {e}")
                self.create_model()
        else:
            self.create_model()
    
    def create_model(self):
        """Crea la arquitectura de la red neuronal"""
        # Entrada: [velocidad_norm, 16 sensores normalizados] = 17 valores
        inputs = keras.Input(shape=(17,), name='sensor_inputs')
        
        # Capa oculta 1: 32 neuronas con activación ReLU (más neuronas para más sensores)
        x = layers.Dense(32, activation='relu', name='hidden1')(inputs)
        x = layers.Dropout(0.2)(x)
        
        # Capa oculta 2: 24 neuronas con activación ReLU
        x = layers.Dense(24, activation='relu', name='hidden2')(x)
        x = layers.Dropout(0.2)(x)
        
        # Capa oculta 3: 16 neuronas con activación ReLU
        x = layers.Dense(16, activation='relu', name='hidden3')(x)
        
        # Salida: 2 neuronas (steering, throttle) con activación tanh para rango [-1, 1]
        outputs = layers.Dense(2, activation='tanh', name='control_outputs')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='CarNeuralController')
        
        # Compilar modelo
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("✓ Arquitectura de red neuronal creada")
        print(self.model.summary())
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """
        Entrena la red neuronal con datos de entrenamiento
        
        Args:
            X_train: Datos de entrada (estados del auto)
            y_train: Datos de salida (acciones de control)
            X_val: Datos de validación (opcional)
            y_val: Salidas de validación (opcional)
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del lote
            
        Returns:
            Historia del entrenamiento
        """
        print("\n🧠 Iniciando entrenamiento de red neuronal...")
        print(f"   Datos de entrenamiento: {X_train.shape}")
        print(f"   Épocas: {epochs}, Batch size: {batch_size}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        # Entrenar
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # Guardar modelo
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        print(f"✓ Modelo guardado en {self.model_path}")
        
        return history
    
    def compute(self, car):
        """
        Calcula las acciones de control basadas en el estado del auto
        
        Args:
            car: Objeto Car con sensores actualizados
            
        Returns:
            Tupla (steering, throttle) con valores entre -1 y 1
        """
        if not self.is_trained:
            print("⚠ Red neuronal no entrenada, usando valores por defecto")
            return 0.0, 0.5
        
        try:
            # Obtener vector de estado del auto
            state = car.get_state_vector()
            
            # Predecir acción (necesita dimensión de batch)
            state_batch = np.expand_dims(state, axis=0)
            action = self.model.predict(state_batch, verbose=0)[0]
            
            # Extraer steering y throttle
            steering = float(action[0])
            throttle = float(action[1])
            
            # Limitar valores al rango [-1, 1]
            steering = np.clip(steering, -1, 1)
            throttle = np.clip(throttle, -1, 1)
            
            return steering, throttle
            
        except Exception as e:
            print(f"Error en control neuronal: {e}")
            return 0.0, 0.5
    
    def evaluate(self, X_test, y_test):
        """
        Evalúa el desempeño del modelo con datos de prueba
        
        Args:
            X_test: Datos de entrada de prueba
            y_test: Datos de salida de prueba
            
        Returns:
            Métricas de evaluación
        """
        if not self.is_trained:
            print("⚠ Red neuronal no entrenada")
            return None
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        print("\n📊 Evaluación de la red neuronal:")
        print(f"   Loss: {results[0]:.4f}")
        print(f"   MAE: {results[1]:.4f}")
        
        return results
    
    def get_architecture_info(self):
        """Retorna información sobre la arquitectura de la red"""
        if self.model is None:
            return "Modelo no inicializado"
        
        return """
        ARQUITECTURA DE RED NEURONAL:
        
        Capa de Entrada: 9 neuronas
          - 1 neurona: velocidad normalizada [-1, 1]
          - 8 neuronas: sensores de distancia normalizados [0, 1]
        
        Capa Oculta 1: 16 neuronas (ReLU) + Dropout(0.2)
        Capa Oculta 2: 12 neuronas (ReLU) + Dropout(0.2)
        Capa Oculta 3: 8 neuronas (ReLU)
        
        Capa de Salida: 2 neuronas (tanh) → [-1, 1]
          - Neurona 1: steering (dirección)
          - Neurona 2: throttle (aceleración)
        
        Optimizador: Adam (lr=0.001)
        Función de pérdida: MSE (Mean Squared Error)
        """
