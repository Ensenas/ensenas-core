import unittest
from io import BytesIO
from PIL import Image
from src.ensenias_api import app


class FlaskTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def create_test_image(self):
        img = Image.new("RGB", (100, 100), color="red")
        img_io = BytesIO()
        img.save(img_io, "JPEG")
        img_io.seek(0)
        return img_io

    def test_upload_image_success(self):
        img = self.create_test_image()

        response = self.app.post("/image", data={"image": (img, "test.jpg")})

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn("width", data)
        self.assertIn("height", data)
        self.assertEqual(data["width"], 100)
        self.assertEqual(data["height"], 100)

    def test_no_image_part(self):
        response = self.app.post("/image", data={})

        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)
        self.assertEqual(data["error"], "No image part in the request")


if __name__ == "__main__":
    unittest.main()
