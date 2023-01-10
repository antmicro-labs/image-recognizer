import unittest
import os
import re
from app.resolution import ResolutionFinder
from parameterized import parameterized

DATA_DIR = "tests/test_data"
FILENAMES = [
    f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))
]


class TestResolution(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.REGEX = r".+_.+_.+_(\d+)x(\d+)_*([a-zA-Z]+)*(\d+)*"
        self.delta = 20
        self.DATA_DIR = "tests/test_data"
        self.FILENAMES = [
            f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))
        ]

    @parameterized.expand([[filename] for filename in FILENAMES])
    def test_resolutions(self, filename: str):
        img_data = re.search(self.REGEX, filename)
        format = img_data.group(3)
        if format is None:
            # No format in name means rgb
            mul = 3
        else:
            bits = img_data.group(4)
            if bits is None:
                mul = 1
            else:
                mul = sum([int(x) for x in bits]) / 8
        expected = int(int(img_data.group(1)) * mul)
        results = ResolutionFinder().find_resolution(f"{DATA_DIR}/{filename}")
        best = min(results, key=lambda x: abs(x.width - expected))
        self.assertAlmostEqual(expected, best.width, delta=self.delta)
