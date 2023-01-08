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

    def template(self, format: str):
        DIR = self.DATA_DIR + f"/{format}"
        for im in os.listdir(DIR):
            expected = int(re.search(self.REGEX, im).group(1))
            results = ResolutionFinder().find_resolution(f"{DIR}/{im}")
            best = min(results, key=lambda x: abs(x.width - expected))
            self.assertAlmostEqual(expected, best.width, delta=self.delta)

    def test_abgr444(self):
        self.template("abgr444")

    def test_abgr555(self):
        self.template("abgr555")

    def test_gray(self):
        self.template("gray")

    def test_rgb(self):
        self.template("rgb")

    def test_rgb332(self):
        self.template("rgb332")

    def test_rgb565(self):
        self.template("rgb565")

    def test_rgba32(self):
        self.template("rgba32")

    def test_uyvy(self):
        self.template("uyvy")
