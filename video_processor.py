import cv2
import numpy as np
import os

# --- Parámetros Configurables ---
NUM_FRAMES_PER_SEQUENCE = 20  # Cuántos frames consecutivos formarán una secuencia
IMG_HEIGHT = 64               # Altura a la que redimensionaremos los frames
IMG_WIDTH = 64                # Ancho al que redimensionaremos los frames
ACTIVITIES = ['welding', 'grinding'] # <--- MUEVE ACTIVITIES AQUÍ ARRIBA
# ---------------------------------

def preprocess_frame(frame, target_height, target_width):
    # ... (tu código existente) ...
    """Redimensiona y normaliza un frame."""
    frame_resized = cv2.resize(frame, (target_width, target_height))
    # Normalizar los píxeles al rango [0, 1]
    frame_normalized = frame_resized / 255.0
    return frame_normalized

def extract_sequences_from_video(video_path, num_frames_per_sequence, img_height, img_width):
    # ... (tu código existente) ...
    """
    Extrae secuencias de frames de un video.
    Devuelve una lista de secuencias, donde cada secuencia es un array de NumPy
    de forma (num_frames_per_sequence, img_height, img_width, 3).
    """
    sequences = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return sequences

    frames_buffer = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break # Fin del video o error

        processed_frame = preprocess_frame(frame, img_height, img_width)
        frames_buffer.append(processed_frame)

        if len(frames_buffer) == num_frames_per_sequence:
            sequences.append(np.array(frames_buffer))
            frames_buffer = [] 

    cap.release()
    return sequences

def load_data_from_folders(base_folder, activity_labels, num_frames_per_sequence, img_height, img_width):
    # ... (tu código existente) ...
    """
    Carga datos de video de carpetas estructuradas por actividad.
    """
    all_sequences = []
    all_labels = []

    for activity_index, activity_name in enumerate(activity_labels): # Usa activity_labels que se pasa como argumento
        activity_path = os.path.join(base_folder, activity_name)
        if not os.path.isdir(activity_path):
            print(f"Advertencia: Carpeta no encontrada para la actividad {activity_name}")
            continue
        
        video_files = [f for f in os.listdir(activity_path) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        print(f"Procesando actividad: {activity_name} ({len(video_files)} videos)")
        for video_file in video_files:
            video_path = os.path.join(activity_path, video_file)
            sequences = extract_sequences_from_video(video_path, num_frames_per_sequence, img_height, img_width)
            
            for seq in sequences:
                all_sequences.append(seq)
                all_labels.append(activity_index)
        # Corrección en el print de secuencias (estaba contando mal antes)
        # Este print es un poco engañoso porque sequences se resetea para cada video.
        # Sería mejor contar el total de secuencias para la actividad después del bucle de videos.
        num_seq_activity = sum(1 for label in all_labels if label == activity_index)
        print(f"Secuencias extraídas para {activity_name}: {num_seq_activity}")


    return np.array(all_sequences), np.array(all_labels)


if __name__ == '__main__':
    # --- Configuración para la carga de datos ---
    # Esta variable DATASET_BASE_FOLDER solo se usa cuando ejecutas video_processor.py directamente
    DATASET_BASE_FOLDER = 'dataset' 
    
    print("Ejecutando video_processor.py directamente...")
    print("Cargando y preprocesando datos de video...")
    
    # Cuando se ejecuta directamente, usa la variable global ACTIVITIES
    X_data, y_data = load_data_from_folders(DATASET_BASE_FOLDER, ACTIVITIES, 
                                            NUM_FRAMES_PER_SEQUENCE, IMG_HEIGHT, IMG_WIDTH)

    if X_data.size == 0:
        print("No se cargaron datos. Asegúrate de que la estructura de carpetas y los videos sean correctos.")
        print(f"Estructura esperada: {DATASET_BASE_FOLDER}/<nombre_actividad>/<video_file.mp4>")
    else:
        print(f"\nForma de los datos de secuencias (X_data): {X_data.shape}")
        print(f"Forma de las etiquetas (y_data): {y_data.shape}")
        if len(y_data) > 0: # Añadida comprobación para evitar error si y_data está vacío
            print(f"Ejemplo de etiqueta: {y_data[0]} (corresponde a '{ACTIVITIES[y_data[0]]}')")
            print(f"Número de secuencias para welding: {np.sum(y_data == ACTIVITIES.index('welding'))}")
            print(f"Número de secuencias para grinding: {np.sum(y_data == ACTIVITIES.index('grinding'))}")

            
