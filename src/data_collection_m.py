import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from my_functions_m import image_process, draw_landmarks_with_lines, keypoint_extraction
import keyboard

# Define the actions (signs) that will be recorded and stored in the dataset
actions = np.array(['hermano', 'hijo', 'mama', 'papa'])

# Define constants for the gesture recognition
PUNTOS_MINIMOS_MANOS = 5

# Define the number of sequences and frames to be recorded for each action
sequences = 30

# Set the frames per second (fps) to match the camera's fps for 1 second recording
fps = 30
frames = 15  # Record 1 second per sequence

# Set the path where the dataset will be stored
PATH = os.path.join('data_m')

# Create directories for each action, sequence, and frame in the dataset
for action, sequence in product(actions, range(sequences)):
    try:
        os.makedirs(os.path.join(PATH, action, str(sequence)))
    except:
        pass

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
# Set the resolution to 1080p and frame rate to 15 fps
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, fps)

if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Create a MediaPipe Holistic object for hand tracking and landmark extraction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    # Loop through each action and sequence to record data
    for action, sequence in product(actions, range(sequences)):
        recorded_frames = 0
        while recorded_frames < frames:
            # Wait for the spacebar key press to start recording
            while True:
                _, image = cap.read()

                results, image = image_process(image, holistic)
                draw_landmarks_with_lines(image, results)

                cv2.putText(image, 'Recording data for the "{}". Sequence number {}.'.format(action, sequence),
                            (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
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
                if valid_hand_landmarks:
                    frame_path = os.path.join(PATH, action, str(sequence), str(recorded_frames))
                    np.save(frame_path, keypoints)
                    recorded_frames += 1

                # Display text on the image indicating the action and sequence number being recorded
                cv2.putText(image, 'Recording data for the "{}". Sequence number {}.'.format(action, sequence),
                            (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('Camera', image)

                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

        recorded_frames = 0  # Reset the recorded frames for the next sequence

    # Release the camera and close any remaining windows
    cap.release()
    cv2.destroyAllWindows()
