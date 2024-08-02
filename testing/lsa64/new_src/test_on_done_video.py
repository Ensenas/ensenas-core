import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
from collections import deque

# Define actions based on the provided list
ACTIONS = ['Opaco', 'Rojo', 'Verde']

# File paths
MODEL_PATH = 'my_model.keras'
LABEL_ENCODER_PATH = './label_encoder.pkl'
VIDEO_PATH = 'all_cut/002_002_001.mp4'  # Replace with the path to your video file

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Check if the label encoder file exists
if not os.path.exists(LABEL_ENCODER_PATH):
    raise FileNotFoundError(f"Label encoder file not found: {LABEL_ENCODER_PATH}")

# Load the trained model
model = load_model(MODEL_PATH)

# Load the label encoder
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.75)


# Define a function to extract keypoints from results
def keypoint_extraction(results):
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    pose = np.array([[res.x, res.y, res.z] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(99)
    return np.concatenate([lh, rh, pose])


# Function to preprocess keypoints for prediction
def preprocess_keypoints(buffer, max_seq_length):
    if len(buffer) < max_seq_length:
        padding = np.zeros((max_seq_length - len(buffer), buffer[0].shape[0]))
        buffer = np.vstack((buffer, padding))
    else:
        buffer = np.array(buffer[-max_seq_length:])
    return np.expand_dims(buffer, axis=0)


# Get the maximum sequence length from training
max_seq_length = model.input_shape[1]

# Buffer to store frames
frame_buffer = deque(maxlen=max_seq_length)

# Process the specified video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Error opening video file: {VIDEO_PATH}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop when the video ends

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)
    keypoints = keypoint_extraction(results)

    if keypoints.sum() != 0:  # If any keypoints were detected
        frame_buffer.append(keypoints)
        print(f"Frame buffer length: {len(frame_buffer)}")  # Debug statement
        if len(frame_buffer) == max_seq_length:
            keypoints = preprocess_keypoints(list(frame_buffer), max_seq_length)
            print(f"Keypoints shape: {keypoints.shape}")  # Debug statement
            prediction = model.predict(keypoints)
            print(f"Prediction: {prediction}")  # Debug statement
            predicted_action = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            cv2.putText(frame, predicted_action, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print(f"Predicted action: {predicted_action}")  # Debug statement

    # Draw the landmarks on the frame
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    cv2.imshow('Action Recognition from Video', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()
