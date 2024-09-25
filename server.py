# -*- coding: utf-8 -*-

import os
import time
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import mediapipe as mp
import cv2
import numpy as np
import base64
import keyboard
import string
import ssl
import logging
import sys
from datetime import datetime
from io import BytesIO
from PIL import Image
import language_tool_python
from my_functions_x import image_process, draw_landmarks_with_lines, keypoint_extraction
from tensorflow.keras.models import load_model
import tensorflow as t
from turbojpeg import TurboJPEG
from collections import deque, defaultdict
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler



unidades = {
    "familiares":["hermana", "hijo", "mama", "papa"],
    "colores": ["amarillo", "negro", "rojo", "verde"],
    "pronombres": ["el-ella", "nosotros", "ustedes", "vos", "yo"]
} 

#---------VARIABLES-------------------
expected_phrase = []
current_word_index = 0
palabra_detectada = None
activo = True
correcto = False

# Define constants for the gesture recognition
PUNTOS_MINIMOS_MANOS = 5
# Define the confidence threshold
CONFIDENCE_THRESHOLD = 0.96  # Adjust this value to make the detection stricter
# Initialize detection counters
detection_count = defaultdict(int)
DETECTION_THRESHOLD = 10  # Number of times a prediction must be detected before being added to the sentence
# Initialize the lists and queue
sentence, keypoints, last_prediction, grammar, grammar_result = [], deque(maxlen=4), [], [], []
activo = False
palabra_detectada =  None
correcto =False
# Set the path to the data directory
PATH = os.path.join('data_remote')
# Create an array of action labels by listing the contents of the data directory
actions = unidades['familiares']
#---------FIN VARIABLES-------------------
# Configurar el logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def handle_exception(exc_type, exc_value, exc_traceback):
    # Verificar si la excepción es un SSLError con "wrong version number"
    if issubclass(exc_type, ssl.SSLError) and "wrong version number" in str(exc_value):
        logging.warning(f"SSL error: {exc_value}")
    else:
        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

# Establecer el manejador de excepciones global
sys.excepthook = handle_exception
app = Flask(__name__)
socketio = SocketIO(app, async_mode='gevent', cors_allowed_origins="*", ping_timeout=20, ping_interval=5)
jpeg = TurboJPEG("C:/Facultad/Proyecto/ensenas-core/venv/Lib/site-packages/PyTurboJPEG-1.7.6.dist-info/libjpeg-turbo-gcc64/bin/libturbojpeg.dll")
# Create an instance of the grammar correction tool for Spanish (Argentina)
tool = language_tool_python.LanguageToolPublicAPI('es-AR')
# Initialize the confidence score
confidence_score = 0.0

# Verifica que TensorFlow reconozca la GPU
print("Num GPUs Available: ", len(t.config.experimental.list_physical_devices('GPU')))

# Cargar el modelo previamente entrenado
with t.device('/GPU:0'):  # Forzar el uso de la GPU al cargar el modelo
    model = load_model('models/familiares_model.h5')

# Inicializa Mediapipe y el modelo de procesamiento
mp_holistic = mp.solutions.holistic.Holistic(
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    refine_face_landmarks=True,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# Variables para el cálculo de FPS
frame_count = 0
start_time = None
transmission_active = False


@app.route('/')
def index():
    # Obtener la fecha y hora actual
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    # Obtener la IP del cliente
    client_ip = request.remote_addr
    # Imprimir la fecha, hora y la IP del cliente
    print()
    print(f"Connected at {current_time} from IP: {client_ip} processing a new connection...")
    return render_template('index_new.html')

@socketio.on('video_frame')
def handle_video_stream(data):
    global frame_count, start_time, transmission_active, jpeg,sentence, keypoints, last_prediction, grammar, grammar_result, confidence_score
    start_time_total = time.perf_counter()
    detection_status = "processing"

    # Detectar si es una nueva transmisión (si no hay frames procesados, es nueva)
    if not transmission_active:
        detection_status = "processing"
        transmission_active = True
        frame_count = 0
        confidence_score = 0.0
        sentence, keypoints, last_prediction, grammar, grammar_result = [], deque(maxlen=4), [], [], []
        start_time = time.perf_counter()

    # Decodificar la imagen recibida del cliente directamente con OpenCV
    img_data = data.get('image')  # Obtén la cadena base64 del diccionario
    if not img_data:
        print("No se recibió ninguna imagen en el evento 'video_frame'.")
        return  # Salir de la función si no se recibió ninguna imagen

    # Procesar la imagen como de costumbre...
    img_bytes = base64.b64decode(img_data.split(',')[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    # np_arr debe contener los datos binarios de la imagen con turbojpeg
    try:
        img = jpeg.decode(np_arr)
    except Exception as e:
        print(f"Error al decodificar con TurboJPEG: {str(e)}")
        return  # Salir de la función si la decodificación falla

    # Incrementar el contador de frames
    frame_count += 1

    # Procesar la imagen usando la GPU si está disponible
    with t.device('/GPU:0'):  # Forzar el uso de la GPU
        results, img = image_process(img, mp_holistic)
        draw_landmarks_with_lines(img, results)


    #------------
    # Deteccion de palabras

    # Extract keypoints and check if there are valid hand landmarks
    keypoints_extracted, valid_hand_landmarks = keypoint_extraction(results, min_hand_landmarks=PUNTOS_MINIMOS_MANOS)  # Ajusta el valor aquí

    if valid_hand_landmarks:
        keypoints.append(keypoints_extracted)

    # Check if 4 frames have been accumulated
    if len(keypoints) == 4:
        # continue  # Si estás dentro de un bucle y quieres continuar al siguiente ciclo
        # Convert keypoints deque to a numpy array
        keypoints_array = np.array(keypoints)
        # Make a prediction on the keypoints using the loaded model
        with t.device('/GPU:0'):  # Asegurarse de que la GPU se usa para la predicción
            prediction = model.predict(keypoints_array[np.newaxis, :, :],verbose=0)
        confidence_score = np.amax(prediction)
        # Check if the maximum prediction value is above the confidence threshold
        if confidence_score > CONFIDENCE_THRESHOLD:
            predicted_action = actions[np.argmax(prediction)]
            detection_count[predicted_action] += 1

            # Check if the predicted sign is detected enough times
            if detection_count[predicted_action] >= DETECTION_THRESHOLD:
                # Check if the predicted sign is different from the previously predicted sign
                if last_prediction != predicted_action:
                    # Append the predicted sign to the sentence list
                    sentence.append(predicted_action)
                    # Record a new prediction to use it on the next cycle
                    last_prediction = predicted_action
                    # Reset the counter for this action
                    detection_count[predicted_action] = 0
                    # Status
                    detection_status = "passed"
    else:
        detection_status = "failed" 

    # Limit the sentence length to 7 elements to make sure it fits on the screen
    if len(sentence) > 7:
        sentence = sentence[-7:]

    # Reset if the "Spacebar" is pressed
    if keyboard.is_pressed(' ') or data.get('reset'):
        sentence, keypoints, last_prediction, grammar, grammar_result = [], deque(maxlen=4), [], [], []
        detection_count.clear()

    # Check if the list is not empty
    if sentence:
        # Capitalize the first word of the sentence
        sentence[0] = sentence[0].capitalize()

    # Check if the sentence has at least two elements
    if len(sentence) >= 2:
        # Check if the last element of the sentence belongs to the alphabet (lower or upper cases)
        if sentence[-1] in string.ascii_lowercase or sentence[-1] in string.ascii_uppercase:
            # Check if the second last element of sentence belongs to the alphabet or is a new word
            if sentence[-2] in string.ascii_lowercase or sentence[-2] in string.ascii_uppercase or (sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                # Combine last two elements
                sentence[-1] = sentence[-2] + sentence[-1]
                sentence.pop(len(sentence) - 2)
                sentence[-1] = sentence[-1].capitalize()

    # Perform grammar check if "Enter" is pressed
    if keyboard.is_pressed('enter'):
        # Record the words in the sentence list into a single string
        text = ' '.join(sentence)
        # Apply grammar correction tool and extract the corrected result
        grammar_result = tool.correct(text)

    # Draw the sentence on the image
    if grammar_result:
        # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
        textsize = cv2.getTextSize(grammar_result, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_X_coord = (img.shape[1] - textsize[0]) // 2

        # Draw the sentence on the image
        cv2.putText(img, grammar_result, (text_X_coord, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
        textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_X_coord = (img.shape[1] - textsize[0]) // 2

        # Draw the sentence on the image
        cv2.putText(img, ' '.join(sentence), (text_X_coord, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


    # Mostrar la puntuación de confianza en la imagen
    cv2.putText(img, f'Confidence: {confidence_score:.2f}',
                (img.shape[1] - 250, img.shape[0] - 20),  # Coordenadas para la posición del texto
                cv2.FONT_HERSHEY_SIMPLEX,  # Tipo de fuente
                0.7,  # Escala de la fuente (tamaño)
                (255, 255, 255),  # Color del texto en formato BGR (blanco)
                2,  # Grosor del texto
                cv2.LINE_AA)  # Tipo de línea para el texto (antialiasing)

    # Funcionalidad "Deteccion"
    # Agregar el texto "Detección" en el medio y arriba de la imagen
    text = "Detectar"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)  # Blanco
    thickness = 2
    # Calcula las coordenadas para centrar el texto
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = text_size[1] + 10  # Un poco por debajo del borde superior
    # Inserta el texto en la imagen
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)


    #------------
    # Calcular el tiempo transcurrido desde que se recibió el primer frame
    elapsed_time = time.perf_counter() - start_time
    # Calcular FPS
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
    # Generar la cadena de texto que deseas imprimir
    status_line = f"Elapsed time: {elapsed_time:.2f} seconds | Frames processed: {frame_count} | FPS reales: {fps:.2f}"
    # Imprimir en la misma línea sobrescribiendo la anterior
    print(status_line.ljust(80), end='\r', flush=True)


    # Redimensionar la imagen a 720p antes de enviarla de vuelta al cliente
    img_resized = cv2.resize(img, (1280, 720))  # Redimensionar a 1280x720
    # Codificar la imagen procesada para enviarla de vuelta al cliente
    # Codificación con OPENCV  VIEJO
    #_, buffer = cv2.imencode('.jpg', img_resized,[int(cv2.IMWRITE_JPEG_QUALITY), 50])
    # Codificación con TurboJPEG
    buffer = jpeg.encode(img_resized, quality=50)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    # Crear un diccionario con la imagen y el tiempo
    end_time_total = time.perf_counter()
    # Calcular el tiempo transcurrido
    total_time_time = end_time_total - start_time_total


    data_to_send = {
        'image': 'data:image/jpeg;base64,' + img_base64,
        'total_time_time': total_time_time,
        'detection_status': detection_status,  
        'confidence_score': confidence_score   
    }


    # Emitir la imagen y el tiempo
    emit('processed_frame', data_to_send)
# Agrega estas variables globales al inicio del script
expected_phrase = []
current_word_index = 0

@socketio.on('corregir_video_stream')
def corregir_video_stream(data):
    global frame_count, start_time, transmission_active, jpeg, sentence, keypoints, last_prediction, grammar, grammar_result
    global confidence_score, activo, palabra_detectada, correcto, expected_phrase, current_word_index

    start_time_total = time.perf_counter()

    # Detectar si es una nueva transmisión
    if not transmission_active:
        transmission_active = True
        frame_count = 0
        confidence_score = 0.0
        sentence, keypoints, last_prediction, grammar, grammar_result = [], deque(maxlen=4), [], [], []
        start_time = time.perf_counter()
        # Resetear variables adicionales
        activo = True
        palabra_detectada = None
        correcto = False
        expected_phrase = []
        current_word_index = 0

    # Obtener la frase del cliente y dividirla en palabras si aún no se ha hecho
    frase = data.get('frase')
    if frase and not expected_phrase:
        expected_phrase = frase.split()
        current_word_index = 0
        activo = True
        palabra_detectada = None
        correcto = False
        print(f"Frase a corregir: {expected_phrase}")

    # Decodificar la imagen recibida del cliente
    img_data = data.get('image')
    if img_data:
        img_bytes = base64.b64decode(img_data.split(',')[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        try:
            img = jpeg.decode(np_arr)
        except Exception as e:
            print(f"Error al decodificar con TurboJPEG: {str(e)}")
            return
    else:
        print("No se recibió ninguna imagen en la transmisión.")
        return

    frame_count += 1

    # Procesar la imagen y extraer keypoints
    with t.device('/GPU:0'):
        results, img = image_process(img, mp_holistic)
        draw_landmarks_with_lines(img, results)

    keypoints_extracted, valid_hand_landmarks = keypoint_extraction(results, min_hand_landmarks=PUNTOS_MINIMOS_MANOS)

    if valid_hand_landmarks:
        keypoints.append(keypoints_extracted)

    # Realizar predicción cuando se acumulen suficientes keypoints
    if len(keypoints) == 4:
        keypoints_array = np.array(keypoints)
        with t.device('/GPU:0'):
            prediction = model.predict(keypoints_array[np.newaxis, :, :], verbose=0)
        confidence_score = np.amax(prediction)
        if confidence_score > CONFIDENCE_THRESHOLD:
            predicted_action = actions[np.argmax(prediction)]
            detection_count[predicted_action] += 1
            if detection_count[predicted_action] >= DETECTION_THRESHOLD:
                if last_prediction != predicted_action:
                    palabra_detectada = predicted_action
                    print(f"Detectado: {predicted_action}")
                    last_prediction = predicted_action
                    detection_count[predicted_action] = 0
                    keypoints.clear()
                    activo = False  # Pausar detección hasta que el usuario presione la barra espaciadora
        else:
            print("Gesto no reconocido")

    # Verificar si la palabra detectada es igual a la palabra esperada
    if palabra_detectada is not None and not activo:
        if expected_phrase and current_word_index < len(expected_phrase):
            palabra_esperada = expected_phrase[current_word_index]
            if palabra_detectada == palabra_esperada:
                correcto = True
                current_word_index += 1
                print(f"Palabra correcta: {palabra_detectada}")
            else:
                correcto = False
                print(f"Palabra incorrecta: {palabra_detectada} (esperada: {palabra_esperada})")
        palabra_detectada = None  # Reiniciar la palabra detectada

    # Mostrar mensajes de retroalimentación
    if not activo:
        if correcto:
            cv2.putText(img, 'Correcto', (10, 140), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, 'Presione barra espaciadora para continuar', (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, 'Incorrecto', (10, 140), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, 'Presione barra espaciadora para reintentar', (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # Mostrar la frase y el progreso
    frase_mostrada = ''
    for i, palabra in enumerate(expected_phrase):
        if i < current_word_index:
            # Palabras ya reconocidas
            frase_mostrada += palabra + ' '
        elif i == current_word_index:
            # Palabra actual a reconocer
            frase_mostrada += '_ '
        else:
            # Palabras pendientes
            frase_mostrada += '_ '

    cv2.putText(img, frase_mostrada.strip(), (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Mostrar la puntuación de confianza en la imagen
    cv2.putText(img, f'Confidence: {confidence_score:.2f}',
                (img.shape[1] - 250, img.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Mostrar la palabra actual a corregir
    if expected_phrase and current_word_index < len(expected_phrase):
        text = "Corregir: " + expected_phrase[current_word_index]
    else:
        text = "Frase completada"

    # Agregar el texto a la imagen
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = text_size[1] + 10
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    # Permitir reintento o continuar al presionar la barra espaciadora
    if keyboard.is_pressed(' ') or data.get('reset'):
        activo = True
        correcto = False
        palabra_detectada = None
        detection_count.clear()
        keypoints.clear()
        last_prediction = None

    # Enviar la imagen procesada al cliente
    img_resized = cv2.resize(img, (1280, 720))
    buffer = jpeg.encode(img_resized, quality=50)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    end_time_total = time.perf_counter()
    total_time_time = end_time_total - start_time_total
    data_to_send = {
        'image': 'data:image/jpeg;base64,' + img_base64,
        'total_time_time': total_time_time
    }
    emit('processed_frame', data_to_send)


@socketio.on('reset_text')
def handle_reset_text():
    global sentence, keypoints, last_prediction, grammar, grammar_result, detection_count, activo, palabra_actual_detectada, correcto, expected_phrase, current_word_index
    print("Reseteando textos y predicciones en el servidor.")
    sentence, keypoints, last_prediction, grammar, grammar_result = [], deque(maxlen=4), [], [], []
    detection_count.clear()
    activo = True
    palabra_actual_detectada = None
    correcto = False
    expected_phrase = []
    current_word_index = 0

@socketio.on('disconnect')
def handle_disconnect():
    global transmission_active
    transmission_active = False
    # Obtener la fecha y hora actual
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print()
    print(f"Transmission ended at {current_time}. Waiting for a new connection...")

@socketio.on('unit_selected')
def handle_unit_selected(data):
    global model, actions  # Asegúrate de que estas variables son globales para poder acceder a ellas
    unidad = data.get('unidad')
    
    if unidad:
        model_path = f'models/{unidad}_model.h5'
        actions = unidades[unidad]   # Asegúrate de tener los archivos de acciones predefinidos
        
        # Cargar el modelo correspondiente
        with t.device('/GPU:0'):  # Usar la GPU si está disponible
            model = load_model(model_path)
        
        print(f"Modelo para {unidad} cargado desde {model_path} con acciones {actions}")
    else:
        print("No se recibió una unidad válida.")

@socketio.on('verificar_frase')
def verificar_frase(data):
    try:
        # Obtener la frase recibida del cliente
        frase_recibida = data.get('frase')

        # Ejemplo de frase correcta (puedes modificarlo según tu lógica)
        frase_correcta = "mi mama cocina"

        # Verificación de la frase
        if frase_recibida.lower() == frase_correcta.lower():
            emit('resultado_frase', {"resultado": "correcto"})
        else:
            emit('resultado_frase', {"resultado": "incorrecto"})
    except Exception as e:
        logging.error(f"Error al verificar la frase: {str(e)}")
        emit('resultado_frase', {"error": "Error interno del servidor"})


if __name__ == '__main__':
    # Configuración del servidor SSL con gevent
    server = pywsgi.WSGIServer(
        ('0.0.0.0', 3051),
        app, 
        handler_class=WebSocketHandler,
        keyfile='privkey.pem',  # Ruta completa al archivo privkey.pem
        certfile='fullchain.pem',  # Ruta completa al archivo fullchain.pem
        ssl_version=ssl.PROTOCOL_TLS  # Usar TLS en lugar de SSLv3
        #keyfile='key.pem',
        #certfile='cert.pem',
        #ssl_version=ssl.PROTOCOL_TLS  # Usar TLS en lugar de SSLv3
    )
    server.serve_forever()
