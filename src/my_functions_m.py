import mediapipe as mp
import cv2
import numpy as np

# Definir los índices de los puntos de interés en la cara y el cuerpo
FACIAL_LANDMARKS = [33, 263, 1, 234, 454, 13, 14, 152, 10]
SHOULDER_LANDMARKS = [11, 12]  # 11: Left shoulder, 12: Right shoulder
ELBOW_LANDMARKS = [13, 14]  # 13: Left elbow, 14: Right elbow

def draw_landmarks_with_lines(image, results):
    """
    Draw the landmarks on the image and lines for shoulders, elbows, and a vertical midline.

    Args:
        image (numpy.ndarray): The input image.
        results: The landmarks detected by Mediapipe.

    Returns:
        None
    """
    # Dibujar los puntos específicos de la cara
    if results.face_landmarks:
        for idx in FACIAL_LANDMARKS:
            x = int(results.face_landmarks.landmark[idx].x * image.shape[1])
            y = int(results.face_landmarks.landmark[idx].y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Puntos verdes
    
    # Dibujar los puntos de las manos si están presentes
    if results.left_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    
    # Dibujar los puntos de los hombros y los codos
    if results.pose_landmarks:
        for idx in SHOULDER_LANDMARKS + ELBOW_LANDMARKS:
            x = int(results.pose_landmarks.landmark[idx].x * image.shape[1])
            y = int(results.pose_landmarks.landmark[idx].y * image.shape[0])
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)  # Puntos azules
        
        # Dibujar la línea entre los hombros
        left_shoulder = results.pose_landmarks.landmark[SHOULDER_LANDMARKS[0]]
        right_shoulder = results.pose_landmarks.landmark[SHOULDER_LANDMARKS[1]]
        cv2.line(image, 
                 (int(left_shoulder.x * image.shape[1]), int(left_shoulder.y * image.shape[0])), 
                 (int(right_shoulder.x * image.shape[1]), int(right_shoulder.y * image.shape[0])), 
                 (255, 0, 0), 2)

        # Dibujar la línea entre los codos
        left_elbow = results.pose_landmarks.landmark[ELBOW_LANDMARKS[0]]
        right_elbow = results.pose_landmarks.landmark[ELBOW_LANDMARKS[1]]
        cv2.line(image, 
                 (int(left_elbow.x * image.shape[1]), int(left_elbow.y * image.shape[0])), 
                 (int(right_elbow.x * image.shape[1]), int(right_elbow.y * image.shape[0])), 
                 (255, 0, 0), 2)

        # Dibujar la línea vertical media del cuerpo
        mid_x = (left_shoulder.x + right_shoulder.x) / 2
        mid_x = int(mid_x * image.shape[1])
        cv2.line(image, 
                 (mid_x, 0), 
                 (mid_x, image.shape[0]), 
                 (0, 255, 255), 2)  # Línea amarilla

def image_process(image, model):
    """
    Process the image and obtain sign landmarks.

    Args:
        image (numpy.ndarray): The input image.
        model: The Mediapipe holistic object.

    Returns:
        results: The processed results containing sign landmarks.
        image: The image set back to writable mode.
    """
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image.flags.writeable = True
    return results, image

def keypoint_extraction(results, min_hand_landmarks=10):
    """
    Extract the keypoints from the sign landmarks.

    Args:
        results: The processed results containing sign landmarks.
        min_hand_landmarks (int): Minimum number of hand landmarks required to consider the detection valid.

    Returns:
        keypoints (numpy.ndarray): The extracted keypoints.
        valid_hand_landmarks (bool): True if sufficient hand landmarks are detected.
    """
    # Extract the keypoints for the left hand if present, otherwise set to zeros
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    # Extract the keypoints for the right hand if present, otherwise set to zeros
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    # Extract the keypoints for the face if present, otherwise set to zeros
    if results.face_landmarks:
        face_landmarks = results.face_landmarks.landmark
        face = np.array([[face_landmarks[i].x, face_landmarks[i].y, face_landmarks[i].z] for i in FACIAL_LANDMARKS]).flatten()
    else:
        face = np.zeros(len(FACIAL_LANDMARKS) * 3)
    
    # Extract the keypoints for the shoulders if present, otherwise set to zeros
    shoulders = np.array([[res.x, res.y, res.z] for idx, res in enumerate(results.pose_landmarks.landmark) if idx in SHOULDER_LANDMARKS]).flatten() if results.pose_landmarks else np.zeros(len(SHOULDER_LANDMARKS) * 3)
    # Extract the keypoints for the elbows if present, otherwise set to zeros
    elbows = np.array([[res.x, res.y, res.z] for idx, res in enumerate(results.pose_landmarks.landmark) if idx in ELBOW_LANDMARKS]).flatten() if results.pose_landmarks else np.zeros(len(ELBOW_LANDMARKS) * 3)
    
    # Concatenate the keypoints for both hands, face, shoulders, and elbows
    keypoints = np.concatenate([lh, rh, face, shoulders, elbows])
    
    # Check if there are enough landmarks detected in at least one hand
    valid_hand_landmarks = (results.left_hand_landmarks is not None and len(results.left_hand_landmarks.landmark) >= min_hand_landmarks) or \
                           (results.right_hand_landmarks is not None and len(results.right_hand_landmarks.landmark) >= min_hand_landmarks)
    
    return keypoints, valid_hand_landmarks
