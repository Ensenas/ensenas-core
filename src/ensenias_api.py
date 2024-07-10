import os
from PIL import Image
from flask import Flask, request
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from modelo_ia import SignLanguageModel
from image_processing import process_frame_full_image


app = Flask(__name__)

# Init IA Model
model_path = './model.keras'
static_sign_model = SignLanguageModel(model_path)

# Videos Folder
UPLOAD_FOLDER = "uploads/"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB max file size

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/image", methods=["POST"])
def upload_image():
    if "image" not in request.files or request.files["image"].filename == "":
        return {"error": "No image part in the request"}, 400

    file = request.files["image"]
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    processed_frame, letra, top3 = process_frame_full_image(image, static_sign_model)
    
    # convert processed frame to jpeg
    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_image = buffer.tobytes()
    
    response = {
        "letra": letra,
        "top3": top3,
        "processed_image": processed_image.decode('latin1')
    }
    return response


@app.route("/video", methods=["POST"])
def upload_video():
    if (
        "video" not in request.files
        or request.files["video"].filename == ""
        or not request.files["video"]
    ):
        return {"error": "No video in the request"}, 400

    file = request.files["video"]

    if not allowed_file(file.filename):
        return {"error": "File type not allowed"}, 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Procesar el video

    response_dict = {"filename": filename, "size": os.path.getsize(file_path)}
    return response_dict, 200


if __name__ == "__main__":
    app.run(debug=True)
