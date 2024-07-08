import os
from PIL import Image
from flask import Flask, request
from werkzeug.utils import secure_filename


app = Flask(__name__)


UPLOAD_FOLDER = "uploads/"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB máximo tamaño de archivo

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/image", methods=["POST"])
def upload_image():
    if "image" not in request.files or request.files["image"].filename == "":
        return {"error": "No image part in the request"}, 400

    file = request.files["image"]

    try:
        image = Image.open(file.stream)

        # Procesar Imagen

        width, height = image.size
        response_dict = {"width": width, "height": height, "format": image.format}

        return response_dict, 200

    except Exception as e:
        return {"error": str(e)}, 500


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
