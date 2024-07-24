import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
from collections import deque

# Define actions based on the provided list
actions = ['Opaco', 'Rojo', 'Verde']

# File paths
model_path = 'my_model.keras'
label_encoder_path = './label_encoder.pkl'

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
    exit(1)

# Check if the label encoder file exists
if not os.path.exists(label_encoder_path):
    print(f"Label encoder file not found: {label_encoder_path}")
    exit(1)

# Load the trained model
model = load_model(model_path)

# Load the label encoder
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)

# Initialize OpenCV VideoCapture
cap = cv2.VideoCapture(0)

# Define a function to extract keypoints from results
def keypoint_extraction(results):
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    pose = np.array([[res.x, res.y, res.z] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(99)
    keypoints = np.concatenate([lh, rh, pose])
    return keypoints

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
max_seq_length = 400  # Ensure this matches your training sequence length
# Buffer to store frames
frame_buffer = deque(maxlen=max_seq_length)

# Variable to track recognition status
recognition_active = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)
    keypoints = keypoint_extraction(results)

    # Toggle recognition status on space bar press
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        recognition_active = not recognition_active
    elif key == ord('q'):
        break

    if recognition_active and keypoints.sum() != 0:  # If any keypoints were detected and recognition is active
        frame_buffer.append(keypoints)
        if len(frame_buffer) == max_seq_length:
            keypoints = preprocess_keypoints(list(frame_buffer), max_seq_length)
            prediction = model.predict(keypoints)
            predicted_action = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            cv2.putText(frame, predicted_action, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw the landmarks on the frame
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    cv2.imshow('Real-Time Action Recognition', frame)

cap.release()
cv2.destroyAllWindows()
holistic.close()
