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

last_checkpoint = len(os.listdir("clean_dataset"))

# Create a label map to map each action label to a numeric value
label_map = {num + 1: label for num, label in enumerate(actions)}  # Note: IDs start from 1

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.25, min_tracking_confidence=0.25)

# Define a function to process each video and extract keypoints
def extract_keypoints_from_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        keypoints.append(keypoint_extraction(results))

        # Visualization of the frame with keypoints
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        draw_landmarks(frame, results)
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return keypoints

# Define a function to extract keypoints from results (excluding face keypoints)
def keypoint_extraction(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(99)
    keypoints = np.concatenate([lh, rh, pose])
    return keypoints

# Define a function to draw landmarks on the frame
def draw_landmarks(frame, results):
    mp_drawing = mp.solutions.drawing_utils
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

# Path to your videos
video_dir = 'all_cut'
output_dir = 'clean_dataset'
clean_videos_dir = './clean_videos'

# Create directories if they do not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(clean_videos_dir):
    os.makedirs(clean_videos_dir)

# Iterate over the videos
for video_file in os.listdir(video_dir)[(last_checkpoint - 1) * 50:]:
    if video_file.endswith('.mp4'):
        print(f'Processing video: {video_file}')
        video_id, interpreter_id, take_id = map(int, video_file[:-4].split('_'))
        action = label_map.get(video_id)
        if action:
            action_dir = os.path.join(output_dir, action)
            if not os.path.exists(action_dir):
                os.makedirs(action_dir)

            video_path = os.path.join(video_dir, video_file)
            include_video = False
            while True:
                keypoints = extract_keypoints_from_video(video_path, action)
                print(f'Extracted keypoints shape: {np.array(keypoints).shape}')  # Debug statement
                cv2.destroyAllWindows()
                while True:
                    key = input("Press 'y' to include, 'n' to skip, 'r' to review again: ").strip().lower()
                    if key in ['y', 'n', 'r']:
                        break
                    print("Invalid input. Please press 'y', 'n', or 'r'.")

                if key == 'y':
                    include_video = True
                    break
                elif key == 'n':
                    include_video = False
                    break
                elif key == 'r':
                    continue

            if include_video:

                # Copy the video file to the clean_videos_dir
                shutil.copy(video_path, clean_videos_dir)
                print(f"Video {video_file} copied to: {clean_videos_dir}")

# Close the holistic model
holistic.close()
