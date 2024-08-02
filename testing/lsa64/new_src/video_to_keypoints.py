import os
import numpy as np
import mediapipe as mp
import cv2
import shutil

# Define actions based on the provided list
actions = [
    'Opaco', 'Rojo', 'Verde', 'Amarillo', 'Brillante', 'Celeste', 'Colores', 'Rosa', 'Mujeres', 'Enemigo', 'Hijo',
    'Hombre', 'Lejos', 'Cajón', 'Nacer', 'Aprender', 'Llamar', 'Espumadera', 'Amargo', 'Leche dulce', 'Leche', 'Agua',
    'Comida', 'Argentina', 'Uruguay', 'País', 'Apellido', 'Dónde', 'Burlar', 'Cumpleaños', 'Desayuno', 'Foto', 'Hambre',
    'Mapa', 'Moneda', 'Música', 'Barco', 'Ninguno', 'Nombre', 'Paciencia', 'Perfume', 'Sordo', 'Trampa', 'Arroz',
    'Asado', 'Dulce', 'Chicle', 'Espaguetis', 'Yogur', 'Aceptar', 'Gracias', 'Apagar', 'Aparecer', 'Aterrizar',
    'Atrapar', 'Ayudar', 'Bailar', 'Bañarse', 'Comprar', 'Copiar', 'Correr', 'Darse cuenta', 'Dar', 'Encontrar'
]

# Create a label map to map each action label to a numeric value
label_map = {num + 1: label for num, label in enumerate(actions)}  # Note: IDs start from 1

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define a function to extract keypoints from results (excluding face keypoints)
def keypoint_extraction(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(99)
    keypoints = np.concatenate([lh, rh, pose])
    return keypoints

# Define a function to process each video and extract keypoints
def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        keypoints.append(keypoint_extraction(results))
    cap.release()
    return keypoints

# Paths to your directories
clean_videos_dir = './clean_videos'
output_dir = 'clean_dataset_v2'

# Create directories if they do not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over the clean videos
for video_file in os.listdir(clean_videos_dir):
    if video_file.endswith('.mp4'):
        print(f'Processing video: {video_file}')
        video_id, interpreter_id, take_id = map(int, video_file[:-4].split('_'))
        action = label_map.get(video_id)
        if action:
            action_dir = os.path.join(output_dir, action)
            if not os.path.exists(action_dir):
                os.makedirs(action_dir)

            video_path = os.path.join(clean_videos_dir, video_file)
            keypoints = extract_keypoints_from_video(video_path)
            print(f'Extracted keypoints shape: {np.array(keypoints).shape}')  # Debug statement

            # Save keypoints as .npy file
            output_file = f'{interpreter_id}_{take_id}.npy'
            np.save(os.path.join(action_dir, output_file), keypoints)
            print(f'Saved keypoints for video: {video_file}')

# Close the holistic model
holistic.close()
