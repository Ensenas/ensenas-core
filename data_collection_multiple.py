import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from my_functions_x import image_process, draw_landmarks_with_lines, keypoint_extraction
import keyboard

# Define the actions (signs) that will be recorded and stored in the dataset
actions = np.array(['rojo'])
unidad = "colores"
# Define constants for the gesture recognition
PUNTOS_MINIMOS_MANOS = 5

# Define the number of sequences and frames to be recorded for each action
sequences = 30
FPS_CAPTURE_MIN = 4  # Minimum FPS to save in disk
FPS_CAPTURE_MAX = 20 # FPS of camera capture
frames = FPS_CAPTURE_MAX  # Record 1 second per sequence

# Set the path where the dataset will be stored
PATH = os.path.join(f'data_multiple/{unidad}')

# Create directories for each action, sequence, and frame in the dataset
for action, sequence in product(actions, range(sequences)):
    for fps in range(FPS_CAPTURE_MIN, FPS_CAPTURE_MAX + 1):  # Creating directories for each FPS (MIN to MAX)
        fps_path = os.path.join(PATH, action, str(sequence), str(fps))
        os.makedirs(fps_path, exist_ok=True)

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
# Set the resolution to 1080p and frame rate to 15 fps                                                                           
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, FPS_CAPTURE_MAX) # FPS of camera capture

if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Create a MediaPipe Holistic object for hand tracking and landmark extraction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    # Loop through each action and sequence to record data
    for action, sequence in product(actions, range(sequences)):
        recorded_frames = 0
        all_frames = []  # To store all frames captured at 20 FPS
        while recorded_frames < frames:
            # Wait for the spacebar key press to start recording
            while True:
                _, image = cap.read()

                results, image = image_process(image, holistic)
                draw_landmarks_with_lines(image, results)

                cv2.putText(image, 'Recording data for the "{}". Sequence number {}.'.format(action, sequence),
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(image, 'Pause.', (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Press "Space" when you are ready.', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Camera', image)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

                if keyboard.is_pressed(' '):
                    break

                if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                    break

            # Wait until enough hand landmarks are detected
            while True:
                _, image = cap.read()
                results, image = image_process(image, holistic)
                draw_landmarks_with_lines(image, results)

                # Extract the landmarks from both hands and face
                keypoints, valid_hand_landmarks = keypoint_extraction(results, min_hand_landmarks=PUNTOS_MINIMOS_MANOS)

                if valid_hand_landmarks:
                    break

                cv2.putText(image, 'Waiting for hand landmarks...', (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Camera', image)

                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            # Start recording frames
            while recorded_frames < frames:
                _, image = cap.read()
                results, image = image_process(image, holistic)
                draw_landmarks_with_lines(image, results)

                # Extract the landmarks from both hands and face and save them in arrays
                keypoints, valid_hand_landmarks = keypoint_extraction(results, min_hand_landmarks=PUNTOS_MINIMOS_MANOS)
                if valid_hand_landmarks:  # Solo grabar el frame si se cumplen los puntos mínimos
                    all_frames.append(keypoints)
                    cv2.putText(image, f'Recording {len(all_frames)} / {FPS_CAPTURE_MAX} frames at {FPS_CAPTURE_MAX} FPS',
                                (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)                    
                    recorded_frames += 1
                else:
                    cv2.putText(image, "No se detectaron suficientes puntos clave",
                                (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA) 
                # Display text on the image indicating the action and sequence number being recorded
                cv2.putText(image, 'Recording data for the "{}". Sequence number {}.'.format(action, sequence),
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('Camera', image)

                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

        # Save the captured frames for different FPS (4 to 20)
        for fps in range(4, 21):
            for i in range(fps):
                frame_index = int(i * (20 / fps))
                frame_path = os.path.join(PATH, action, str(sequence), str(fps), f'{i}.npy')
                np.save(frame_path, all_frames[frame_index])

        recorded_frames = 0  # Reset the recorded frames for the next sequence
        cv2.putText(image, f'Frames saved for {fps} FPS', 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('Camera', image)
        cv2.waitKey(1000)  # Wait 1 second before starting the next sequence

    # Release the camera and close any remaining windows
    cap.release()
    cv2.destroyAllWindows()