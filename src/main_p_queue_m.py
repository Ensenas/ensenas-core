import numpy as np
import os
import string
import mediapipe as mp
import cv2
from my_functions_x import image_process, draw_landmarks_with_lines, keypoint_extraction
import keyboard
from tensorflow.keras.models import load_model
import language_tool_python
import tensorflow as tf
from collections import deque, defaultdict

# Verificar si TensorFlow está utilizando la GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Listar los dispositivos físicos disponibles
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("TensorFlow is using the following GPU devices:")
    for device in physical_devices:
        print(device)
else:
    print("No GPU devices found.")

# Set the path to the data directory
PATH = os.path.join('data_m')

# Define constants for the gesture recognition
PUNTOS_MINIMOS_MANOS = 5
# Create an array of action labels by listing the contents of the data directory
actions = np.array(os.listdir(PATH))

# Load the trained model
model = load_model('my_model_m.h5')

# Create an instance of the grammar correction tool for Spanish (Argentina)
tool = language_tool_python.LanguageToolPublicAPI('es-AR')

# Initialize the lists and queue
sentence, keypoints, last_prediction, grammar, grammar_result = [], deque(maxlen=15), [], [], []

# Initialize detection counters
detection_count = defaultdict(int)
DETECTION_THRESHOLD = 8  # Number of times a prediction must be detected before being added to the sentence

# Define the confidence threshold
CONFIDENCE_THRESHOLD = 0.96  # Adjust this value to make the detection stricter

# Initialize the confidence score
confidence_score = 0.0

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
# Set the resolution to 1080p and frame rate to 15 fps
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Create a holistic object for sign prediction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    # Create a named window and set it to fullscreen
    cv2.namedWindow('Camera', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Run the loop while the camera is open
    while cap.isOpened():
        # Read a frame from the camera
        _, image = cap.read()
        # Process the image and obtain sign landmarks using image_process function from my_functions_x.py
        results, image = image_process(image, holistic)
        # Draw the sign landmarks on the image using draw_landmarks_with_lines function from my_functions_x.py
        draw_landmarks_with_lines(image, results)
        # Extract keypoints and check if there are valid hand landmarks
        keypoints_extracted, valid_hand_landmarks = keypoint_extraction(results, min_hand_landmarks=PUNTOS_MINIMOS_MANOS)  # Ajusta el valor aquí
        
        if valid_hand_landmarks:
            keypoints.append(keypoints_extracted)

        # Check if 15 frames have been accumulated
        if len(keypoints) == 15:
            # Convert keypoints deque to a numpy array
            keypoints_array = np.array(keypoints)
            # Make a prediction on the keypoints using the loaded model
            with tf.device('/GPU:0'):  # Asegurarse de que la GPU se usa para la predicción
                prediction = model.predict(keypoints_array[np.newaxis, :, :])
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

        # Limit the sentence length to 7 elements to make sure it fits on the screen
        if len(sentence) > 7:
            sentence = sentence[-7:]

        # Reset if the "Spacebar" is pressed
        if keyboard.is_pressed(' '):
            sentence, keypoints, last_prediction, grammar, grammar_result = [], deque(maxlen=15), [], [], []
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

        if grammar_result:
            # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
            textsize = cv2.getTextSize(grammar_result, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2

            # Draw the sentence on the image
            cv2.putText(image, grammar_result, (text_X_coord, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
            textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2

            # Draw the sentence on the image
            cv2.putText(image, ' '.join(sentence), (text_X_coord, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the confidence score on the image
        cv2.putText(image, f'Confidence: {confidence_score:.2f}', (image.shape[1] - 250, image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the image on the display
        cv2.imshow('Camera', image)

        # Break the loop if the "Esc" key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # Check if the 'Camera' window was closed and break the loop
        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Shut off the server
    tool.close()

