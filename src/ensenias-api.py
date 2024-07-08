from PIL import Image
from flask import Flask, request


app = Flask(__name__)


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


if __name__ == "__main__":
    app.run(debug=True)
