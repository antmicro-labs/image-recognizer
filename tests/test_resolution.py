import unittest
import os
import re
from app.resolution import ResolutionFinder


class TestResolution(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.DATA_DIR = "tests/test_data"
        self.REGEX = r".+_.+_.+_(\d+)x(\d+).*"
        self.delta = 20

    def template(self, format: str, mul: float):
        DIR = self.DATA_DIR + f"/{format}"
        for im in os.listdir(DIR):
            expected = int(int(re.search(self.REGEX, im).group(1)) * mul)
            results = ResolutionFinder().find_resolution(f"{DIR}/{im}")
            best = min(results, key=lambda x: abs(x.width - expected))
            print(f"Got: {best.width} Expected: {expected}")
            with self.subTest():
                self.assertAlmostEqual(expected, best.width, delta=self.delta)

    def test_abgr444(self):
        self.template("abgr444", 12 / 8)

    def test_abgr555(self):
        self.template("abgr555", 15 / 8)

    def test_gray(self):
        self.template("gray", 1)

    def test_rgb(self):
        self.template("rgb", 3)

    def test_rgb332(self):
        self.template("rgb332", 1)

    def test_rgb565(self):
        self.template("rgb565", 2)

    def test_rgba32(self):
        self.template("rgba32", 4)

    def test_uyvy(self):
        self.template("uyvy", 1)
