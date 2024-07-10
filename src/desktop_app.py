"""Maneja la interfaz gráfica, capturando video y mostrando resultados"""
import os
import cv2
import numpy as np
from modelo_ia import SignLanguageModel
from image_processing import process_frame_roi

# Inicializa el modelo de IA
model_path = './model.keras'
sign_model = SignLanguageModel(model_path)

if __name__ == "__main__":
    # get the reference to the webcam
    camera = cv2.VideoCapture(0)
    # region of interest (ROI) coordinates
    top, right, bottom, left = 50, 50, 200, 200
    # initialize num of frames
    num_frames = 0

    lista = os.listdir('Signos_ASL')

    while True:
        # get the current frame
        (grabbed, frame) = camera.read()
        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        
        # Frame Processing
        processed_frame, letra = process_frame_roi(clone, top, right, bottom, left, sign_model)
        
        num_frames += 1
        cv2.imshow("Video Feed", processed_frame)

        keypress2 = cv2.waitKey(1)
        if keypress2 == ord(" "):
            letrica = lista[np.random.randint(24)]
            letraimagen = cv2.imread('Signos_ASL/' + letrica)
            letraimagen = cv2.putText(letraimagen, str(letrica[0]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (226, 43, 138), 2, cv2.LINE_AA)
            cv2.imshow("Letra", letraimagen)

        keypress = cv2.waitKey(1)
        if keypress == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
