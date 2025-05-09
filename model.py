import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import LSTM, TimeDistributed, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Importar desde tu script video_processor.py
print(f"Intentando importar desde 'video_processor.py'...")
try:
    from video_processor import load_data_from_folders, NUM_FRAMES_PER_SEQUENCE, IMG_HEIGHT, IMG_WIDTH, ACTIVITIES
    print("Importación de 'video_processor.py' exitosa!")
except ImportError as e:
    print(f"Error de importación detallado: {e}") # <--- AÑADE ESTA LÍNEA PARA VER EL ERROR EXACTO
    print("Error: Asegúrate de que 'video_processor.py' está en el mismo directorio y no tiene errores.")
    # Definir valores por defecto si la importación falla... (el resto de tu bloque except)
    NUM_FRAMES_PER_SEQUENCE = 20
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    ACTIVITIES = ['welding', 'grinding'] 
    def load_data_from_folders(*args, **kwargs): 
        print("ADVERTENCIA: Usando función dummy load_data_from_folders porque la importación falló.")
        return np.array([]), np.array([])


# Parámetros adicionales para el modelo
CHANNELS = 3 # RGB
NUM_CLASSES = len(ACTIVITIES)


def create_cnn_lstm_model(num_frames, height, width, channels, num_classes):
    """
    Crea un modelo CNN + LSTM para reconocimiento de actividades.
    - CNN procesa cada frame individualmente (usando TimeDistributed).
    - LSTM procesa la secuencia de características extraídas por la CNN.
    """
    model = Sequential()

    # --- Parte CNN (Aplicada a cada frame de la secuencia) ---
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), 
                              input_shape=(num_frames, height, width, channels)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    
    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    
    model.add(TimeDistributed(Flatten()))
    
    # --- Parte LSTM ---
    model.add(LSTM(128, return_sequences=False)) # Aumentado el número de unidades LSTM
    model.add(Dropout(0.5))
    
    # --- Clasificador Final ---
    model.add(Dense(64, activation='relu')) # Capa densa adicional
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), # Ajuste de learning rate
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # --- 1. Cargar Datos ---
    DATASET_BASE_FOLDER = 'dataset' # Asegúrate que esta es tu carpeta de datasets
    print(f"Cargando datos desde: {DATASET_BASE_FOLDER}")
    print(f"Actividades esperadas: {ACTIVITIES}")
    print(f"Parámetros de secuencia: {NUM_FRAMES_PER_SEQUENCE} frames, {IMG_HEIGHT}x{IMG_WIDTH}px")

    X_data, y_data_numeric = load_data_from_folders(DATASET_BASE_FOLDER, ACTIVITIES, 
                                                    NUM_FRAMES_PER_SEQUENCE, IMG_HEIGHT, IMG_WIDTH)

    if X_data.size == 0:
        print("\nNo se cargaron datos. Verifica lo siguiente:")
        print(f"1. Que la carpeta '{DATASET_BASE_FOLDER}' exista en el mismo directorio que este script.")
        print(f"2. Que dentro de '{DATASET_BASE_FOLDER}' existan las carpetas de actividades: {ACTIVITIES}")
        print("3. Que cada carpeta de actividad contenga archivos de video (.mp4, .avi, etc.).")
        print("4. Que 'video_processor.py' funcione correctamente y esté en el mismo directorio.")
        exit()
    
    print(f"\nForma de los datos de secuencias (X_data): {X_data.shape}")
    print(f"Forma de las etiquetas (y_data_numeric): {y_data_numeric.shape}")

    # --- 2. Preparar Etiquetas y Dividir Datos ---
    # Convertir etiquetas a formato categórico (one-hot encoding)
    y_data_categorical = to_categorical(y_data_numeric, num_classes=NUM_CLASSES)
    print(f"Forma de las etiquetas categóricas (y_data_categorical): {y_data_categorical.shape}")

    if X_data.shape[0] <= 1:
        print("\nError: Se necesita más de una muestra para dividir en entrenamiento y prueba.")
        print("Por favor, añade más videos o asegúrate que tus videos son lo suficientemente largos para generar múltiples secuencias.")
        exit()
        
    # Dividir en entrenamiento y prueba (80% entrenamiento, 20% prueba)
    # 'stratify' ayuda a mantener la proporción de clases en ambos conjuntos
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data_categorical, 
            test_size=0.20, 
            random_state=42, # Para reproducibilidad
            stratify=y_data_numeric # importante para clases desbalanceadas
        )
        print(f"\nDatos divididos:")
        print(f"Forma de X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Forma de X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Verificar la distribución de clases en entrenamiento y prueba
        print(f"Distribución de clases en y_train: {np.sum(y_train, axis=0)}")
        print(f"Distribución de clases en y_test: {np.sum(y_test, axis=0)}")

    except ValueError as e:
        print(f"\nError al dividir los datos: {e}")
        print("Esto puede suceder si tienes muy pocas muestras de alguna clase.")
        print(f"Distribución de tus etiquetas originales (y_data_numeric):")
        for i, activity in enumerate(ACTIVITIES):
            print(f" - Clase '{activity}' (etiqueta {i}): {np.sum(y_data_numeric == i)} muestras")
        exit()


    # --- 3. Crear y Compilar el Modelo ---
    print("\nCreando el modelo CNN+LSTM...")
    model = create_cnn_lstm_model(NUM_FRAMES_PER_SEQUENCE, IMG_HEIGHT, IMG_WIDTH, CHANNELS, NUM_CLASSES)
    model.summary() # Imprime la estructura del modelo

    # --- 4. Entrenamiento (¡Esto puede tardar!) ---
    # Considera empezar con pocas épocas y un batch_size pequeño para probar.
    # Necesitarás una cantidad decente de datos para que el modelo aprenda algo útil.
    
    if X_train.shape[0] == 0:
        print("\nNo hay datos de entrenamiento. Revisa los pasos anteriores y la cantidad de videos.")
    else:
        print("\nIniciando entrenamiento...")
        print("Si es tu primera vez, considera usar pocas 'epochs' (ej. 10-20) y un 'batch_size' (ej. 4-8) pequeño.")
        
        # Callbacks (opcional, pero útil)
        # Reduce el learning rate si la validación no mejora
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
        # Detiene el entrenamiento temprano si no hay mejora
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

        history = model.fit(X_train, y_train,
                            epochs=10,  # Número de veces que el modelo verá todo el dataset
                            batch_size=4, # Número de secuencias a procesar antes de actualizar pesos
                            validation_data=(X_test, y_test),
                            callbacks=[reduce_lr, early_stopping])
        
        print("\nEntrenamiento completado.")
        print("history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), callbacks=[reduce_lr, early_stopping])")


    # --- 5. Guardar el Modelo (Opcional, pero recomendado después de un entrenamiento real) ---
    if 'history' in locals(): # Solo si el entrenamiento se ejecutó
        print("\nGuardando el modelo entrenado...")
        model.save('activity_recognition_model.keras') # Nuevo formato recomendado .keras
        print("Modelo guardado como 'activity_recognition_model.keras'")
    else:
        print("\nModelo no entrenado, no se guardará.")
