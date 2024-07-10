""" Contiene funciones para procesar imágenes y manejar la lógica de segmentación."""
import cv2
import numpy as np

bg = None


def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def process_frame_roi(frame, top, right, bottom, left, sign_model):
    roi = frame[top:bottom, right:left]
    pred = sign_model.predict(roi)
    letra = sign_model.get_top_prediction(pred)
    top3 = sign_model.get_top_predictions(pred)
    
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(frame, letra, (left - 90, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, top3[1], (left - 150, top + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, top3[2], (left - 10, top + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    return frame, letra


def process_frame_full_image(frame, sign_model):
    pred = sign_model.predict(frame)
    top = sign_model.get_top_predictions(pred=pred, top_n=4)
    letra = top.pop(0)
    
    cv2.putText(frame, letra, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, top[1], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, top[2], (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame, letra, top
