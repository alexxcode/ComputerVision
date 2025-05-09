# Reconocimiento de Actividades Industriales en Video

Este proyecto tiene como objetivo desarrollar un sistema capaz de reconocer actividades industriales específicas, como "soldadura" (welding) y "corte con esmeriladora" (grinding), a partir de secuencias de video utilizando técnicas de Computer Vision y Deep Learning.

## Objetivo

El sistema analiza un flujo de video (ya sea un archivo grabado o una transmisión en vivo desde una cámara) para identificar y clasificar en tiempo real las actividades industriales que se están llevando a cabo.

##  Características Principales

* **Entrenamiento de Modelo Personalizado:** Utiliza una arquitectura CNN+LSTM para aprender características espaciales y temporales de los videos.
* **Procesamiento de Video:** Scripts para extraer secuencias de frames de videos y prepararlos para el entrenamiento.
* **Inferencia en Tiempo Real/Video:** Capacidad de cargar un modelo entrenado para predecir actividades en nuevos videos.
* **Visualización:** Muestra el video con la etiqueta de la actividad predicha y resalta las regiones de movimiento asociadas a la actividad detectada.

## 🛠️ Actividades Reconocidas (Ejemplos)

Actualmente, el sistema está enfocado en reconocer las siguientes actividades (pero puede ser extendido):

1.  Soldadura (Welding)
2.  Corte con Esmeriladora (Grinding)

## 💻 Tecnologías Utilizadas

* Python 3.8+
* OpenCV
* TensorFlow (Keras API)
* NumPy
* Scikit-learn


##  Instalación

1.  **Requisitos Previos:**
    * Python 3.8 o superior instalado.
    * `pip` (el gestor de paquetes de Python) instalado.

2.  **Clonar el Repositorio (Opcional):**


3.  **Crear un Entorno Virtual (Recomendado):**

      

4.  **Instalar Dependencias:**
    Asegúrate de tener un archivo `requirements.txt` con el siguiente contenido:
    ```txt
    opencv-python
    tensorflow
    numpy
    scikit-learn
    ```
    Luego, instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

##  Preparación de Datos para Entrenamiento

1.  Crea una carpeta llamada `dataset` en la raíz del proyecto.
2.  Dentro de `dataset`, crea subcarpetas para cada actividad que quieras reconocer. Los nombres de estas subcarpetas deben coincidir con los definidos en la lista `ACTIVITIES` en los scripts `video_processor.py` y `model.py` (ej. `welding`, `grinding`).
3.  Coloca tus archivos de video (ej. `.mp4`, `.avi`) en las carpetas de actividad correspondientes

##  Uso del Proyecto

El flujo de trabajo consta de tres pasos principales:

**Paso 1: Procesamiento de Datos para Entrenamiento**

Este paso lee tus videos, los divide en secuencias de frames y los prepara para ser alimentados al modelo.

* Abre tu terminal, navega a la carpeta del proyecto y (con el entorno virtual activado) ejecuta:
    ```bash
    python video_processor.py
    ```
* Este script procesará los videos de la carpeta `dataset/` e imprimirá información sobre las secuencias y etiquetas generadas. Verifica que no haya errores y que se detecten tus videos.

**Paso 2: Entrenamiento del Modelo**

Este paso define la arquitectura de la red neuronal (CNN+LSTM), carga los datos procesados y entrena el modelo.

* Abre el archivo `model.py` en un editor de texto.
* Localiza la sección de entrenamiento (busca `model.fit(...)`). Si está comentada, descoméntala para habilitar el entrenamiento. Puedes ajustar parámetros como `epochs` y `batch_size`.
    ```python
    # history = model.fit(X_train, y_train,
    #                     epochs=50,  # Ajusta según sea necesario
    #                     batch_size=8, # Ajusta según tu memoria
    #                     validation_data=(X_test, y_test),
    #                     callbacks=[reduce_lr, early_stopping]) # Descomentar para entrenar
    ```
* Guarda los cambios en `model.py`.
* Ejecuta el script desde tu terminal:
    ```bash
    python model.py
    ```
* El entrenamiento puede tardar considerablemente dependiendo de la cantidad de datos, la complejidad del modelo y tu hardware.
* Una vez finalizado (o si descomentaste la sección de guardado), se generará un archivo llamado `activity_recognition_model.keras` (o el nombre que hayas configurado), que contiene tu modelo entrenado.

**Paso 3: Inferencia y Predicción en Nuevos Videos**

Este paso carga el modelo entrenado y lo utiliza para predecir actividades en un video nuevo (o desde una cámara).

* Abre el archivo `predict_activity.py`.
* Modifica la variable `VIDEO_SOURCE` para que apunte a la ruta de tu video de prueba o usa `0` para la cámara web:
    ```python
    VIDEO_SOURCE = 'ruta/a/tu/video_de_prueba.mp4' # O VIDEO_SOURCE = 0
    ```
* También puedes ajustar `PREDICTION_INTERVAL` y `CONFIDENCE_THRESHOLD` si es necesario.
* Guarda los cambios.
* Ejecuta el script desde tu terminal:
    ```bash
    python predict_activity.py
    ```
* Se abrirá una ventana mostrando el video con la actividad predicha y (si aplica) rectángulos verdes alrededor de las zonas de movimiento detectadas durante la actividad. Presiona 'q' para cerrar la ventana.

##  Personalización y Parámetros Clave

Puedes ajustar varios parámetros en la parte superior de los scripts (`video_processor.py`, `model.py`, `predict_activity.py`):

* `NUM_FRAMES_PER_SEQUENCE`: Número de frames por secuencia para el modelo.
* `IMG_HEIGHT`, `IMG_WIDTH`: Dimensiones a las que se redimensionan los frames.
* `ACTIVITIES`: Lista de nombres de las actividades (debe coincidir con los nombres de las carpetas en `dataset/`).
* En `model.py`: Arquitectura de la red, optimizador, tasa de aprendizaje, épocas, tamaño del batch.
* En `predict_activity.py`: `PREDICTION_INTERVAL`, `CONFIDENCE_THRESHOLD`, `MIN_CONTOUR_AREA`.

##  Posibles Mejoras y Trabajo Futuro

* **Aumentar el Conjunto de Datos:** Recopilar más videos y más variados para cada actividad para mejorar la robustez y precisión del modelo.
* **Data Augmentation:** Aplicar técnicas de aumento de datos a los frames de video.
* **Ajuste Fino de Hiperparámetros:** Experimentar sistemáticamente con diferentes arquitecturas de modelo y parámetros de entrenamiento.
* **Transfer Learning:** Utilizar modelos CNN pre-entrenados (ej. MobileNetV2, ResNet) como extractores de características.
* **Localización Espacial Precisa:** Implementar modelos de detección de objetos o localización de acciones espacio-temporales para obtener bounding boxes más precisos alrededor de los actores o herramientas.
* **Suavizado de Predicciones:** Aplicar filtros temporales a las predicciones para una salida más estable.
* **Interfaz Gráfica de Usuario (GUI):** Desarrollar una interfaz más amigable.


##  Contacto

Alexis - alexisrrm12@gmail.com 
