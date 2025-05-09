import cv2
import numpy as np
import tensorflow as tf
from collections import deque # Para el buffer de frames

# Importar constantes y funciones de preprocesamiento
try:
    from video_processor import preprocess_frame, NUM_FRAMES_PER_SEQUENCE, IMG_HEIGHT, IMG_WIDTH, ACTIVITIES
except ImportError as e:
    print(f"Error al importar desde video_processor.py: {e}")
    # ... (código de fallback como antes) ...
    NUM_FRAMES_PER_SEQUENCE = 20
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    ACTIVITIES = ['welding', 'grinding']
    def preprocess_frame(frame, target_height, target_width):
        print("ADVERTENCIA: Usando función dummy preprocess_frame.")
        frame_resized = cv2.resize(frame, (target_width, target_height))
        return frame_resized / 255.0

# --- Parámetros de Inferencia ---
MODEL_PATH = 'activity_recognition_model.keras'
VIDEO_SOURCE = 'video_de_prueba2.mp4' # Reemplaza con la ruta a tu video o 0 para la cámara web
PREDICTION_INTERVAL = 10 
CONFIDENCE_THRESHOLD = 0.6
MIN_CONTOUR_AREA = 500 # Área mínima para dibujar un rectángulo de "movimiento"

# --- Función para detección de movimiento (simplificada) ---
def find_motion_regions(frame1, frame2, min_area):
    """Encuentra regiones de movimiento entre dos frames."""
    rectangles = []
    if frame1 is None or frame2 is None:
        return rectangles

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
    
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
    
    frame_delta = cv2.absdiff(gray1, gray2)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        rectangles.append((x, y, w, h))
        
    return rectangles

def main():
    # 1. Cargar el Modelo Entrenado
    print(f"Cargando modelo desde: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return
    
    print("Modelo cargado exitosamente.")

    # 2. Preparar la Captura de Video
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video o la cámara en '{VIDEO_SOURCE}'")
        return

    frames_queue = deque(maxlen=NUM_FRAMES_PER_SEQUENCE)
    frame_count = 0
    predicted_activity_text = "Analizando..." # Texto para mostrar en pantalla
    previous_original_frame = None # Para la detección de movimiento

    print(f"Procesando video: {VIDEO_SOURCE}. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o error al leer frame.")
            break

        original_frame_for_display = frame.copy() # Usar esta para dibujar y mostrar
        
        processed_frame_for_model = preprocess_frame(frame, IMG_HEIGHT, IMG_WIDTH)
        frames_queue.append(processed_frame_for_model)

        frame_count += 1
        
        current_activity_detected = None # Para saber si se detectó una actividad específica en esta iteración

        if len(frames_queue) == NUM_FRAMES_PER_SEQUENCE and frame_count % PREDICTION_INTERVAL == 0:
            sequence_to_predict = np.expand_dims(np.array(frames_queue), axis=0)
            predictions = model.predict(sequence_to_predict, verbose=0)[0]
            predicted_class_index = np.argmax(predictions)
            confidence = predictions[predicted_class_index]

            if confidence >= CONFIDENCE_THRESHOLD:
                activity_name = ACTIVITIES[predicted_class_index]
                predicted_activity_text = f"{activity_name} ({confidence:.2f})"
                # Solo consideramos "welding" o "grinding" para dibujar rectángulos de movimiento
                if activity_name.lower() in ["welding", "grinding"]:
                    current_activity_detected = activity_name 
            else:
                predicted_activity_text = "Incierto"
        
        # Dibujar rectángulos de movimiento si se detectó una actividad relevante
        if current_activity_detected and previous_original_frame is not None:
            motion_rects = find_motion_regions(previous_original_frame, frame, MIN_CONTOUR_AREA)
            for (x, y, w, h) in motion_rects:
                cv2.rectangle(original_frame_for_display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostrar la actividad predicha (texto)
        cv2.putText(original_frame_for_display, predicted_activity_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) # Color rojo para el texto
        
        cv2.imshow('Reconocimiento de Actividad', original_frame_for_display)
        
        previous_original_frame = frame.copy() # Actualizar el frame anterior para la próxima iteración

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Procesamiento finalizado.")

if __name__ == '__main__':
    main()