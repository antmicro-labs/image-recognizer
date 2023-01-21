import unittest
import os
import re
from app.image_recognizer import ImageRecognizer


class TestImageRecognizer(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.DATA_DIR = "tests/test_data"
        self.REGEX = r".+_.+_.+_(\d+)x(\d+).*"

    def template(self, color_format: str):
        DIR = self.DATA_DIR + f"/{color_format}"
        image_recognizer = ImageRecognizer()
        for im in os.listdir(DIR):
            exp_width = int(int(re.search(self.REGEX, im).group(1)))
            exp_color_format = color_format
            img_path = f"{DIR}/{im}"
            print(img_path)
            res = image_recognizer.recognize(img_path)
            print(f"Got: {res.img_width}, {res.color_format} | Expected: {exp_width}, {exp_color_format}")
            with self.subTest():
                self.assertTupleEqual((exp_width, exp_color_format), (res.img_width, res.color_format))

    def test_abgr444(self):
        self.template("ABGR444")

    def test_abgr555(self):
        self.template("ABGR555")

    def test_gray(self):
        self.template("GRAY")

    def test_rgb(self):
        self.template("RGB24")

    def test_rgb332(self):
        self.template("RGB332")

    def test_rgb565(self):
        self.template("RGB565")

    def test_rgba32(self):
        self.template("RGBA32")

    def test_uyvy(self):
        self.template("UYVY")
