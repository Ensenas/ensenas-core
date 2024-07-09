"""Se encarga de la lógica de predicción y procesamiento de la imagen."""
from tensorflow.keras.models import load_model
import numpy as np
import cv2

class SignLanguageModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.abc = 'ABCDEFGHIKLMNOPQRSTUVWXY'

    def predict(self, roi):
        k = 2
        resized = cv2.resize(roi, (28 * k, 28 * k), interpolation=cv2.INTER_AREA) / 255
        pred = self.model.predict(resized.reshape(-1, 28 * k, 28 * k, 3))
        return pred

    def get_top_predictions(self, pred, top_n=3):
        index = np.argsort(pred[0])
        top_indices = index[-top_n:][::-1]
        return [self.abc[i] for i in top_indices]

    def get_top_prediction(self, pred):
        return self.abc[np.argmax(pred)]
