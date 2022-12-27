import math
import tensorflow as tf
from tensorflow import keras  # noqa
from keras import models
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt  # noqa
from typing import Tuple


class ResolutionFinder:
    def __init__(
        self,
        model_path="models/default_resolution_model.h5",
    ):
        self.model: models.Model = models.load_model(model_path)

    class ImageTooSmall(Exception):
        pass

    def single_prediction_tour(
        self,
        raw: bytes,
        min_width: int = 256,
        max_width: int = 0,
        step: int = 50,
        min_height: int = 256,
        max_wh_ratio: int = 6,
    ) -> int:
        """Predict resolution of image from raw bytes,
        by checking some possible resolutions based on passed argumetns

        Args:
            raw (bytes): Raw bytes of image.
            min_width (int, optional): Min width to check. Defaults to 256.
            max_width (int, optional): Max width to check. Defaults to 0.
            step (int, optional): Width change step. Defaults to 50.
            min_height (int, optional): Min height to check. Defaults to 256.
            max_wh_ratio (int, optional): Max ratio beetwen width and height. Defaults to 6.

        Raises:
            self.ImageTooSmall: Given image is too small to check.

        Returns:
            int: Predicted image line width.
        """
        if len(raw) < min_width * min_height:
            raise self.ImageTooSmall(
                f"Image is too small. Required size: {min_width}*{min_height}\nActual size: {len(raw)}"
            )
        if not max_width:
            max_width = int(math.sqrt(max_wh_ratio * len(raw)))
        batches = []
        for batch_width in range(min_width, max_width, step):
            batch_height = len(raw) // batch_width
            img = Image.frombytes("L", (batch_width, batch_height), raw)
            img = img.resize((256, 256))
            img = np.asarray(img)
            img = np.expand_dims(img, axis=-1)
            batches.append(img / 255 - 0.5)
        batches = tf.stack(batches)
        predictions = self.model.predict(batches)
        resolution = int((predictions.argmax()) * step + min_width)
        return resolution

    def find_resolution(self, path: str) -> Tuple[int, int]:
        """Find resolution of image, from given path

        Args:
            path (str): Path to image.

        Returns:
            Tuple(int, int): width, height
        """
        with open(path, "rb") as f:
            raw = f.read()
            w1 = self.single_prediction_tour(raw)
            w2 = self.single_prediction_tour(raw, w1 - 25, w1 + 25, 1)
            h = len(raw) // w2
            # plt.imshow(np.asarray(Image.frombytes("L", (w2, h), raw)))
            # plt.show()
            return w2, h


w, h = ResolutionFinder().find_resolution("test_data/GRAY_1000_750")
# TEST_DATA_FOLDER = "test_data"
# for im in os.listdir(TEST_DATA_FOLDER):
#     width, height = ResolutionFinder().find_resolution(f"{TEST_DATA_FOLDER}/{im}")
#     print(f"Img: {im.split('_')[0]} width: {width}")
