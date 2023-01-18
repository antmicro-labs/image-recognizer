import math
import os
import logging
import json
import tensorflow as tf
from tensorflow import keras  # noqa
from keras import models
import numpy as np
from PIL import Image
import gdown
from typing import List
from numpy.typing import NDArray
from dataclasses import dataclass
from pathlib import Path


class ResolutionFinder:
    with open("models_links.json") as f:
        DEFAULT_MODEL_URL = json.load(f).get("default_resolution_model", None)
    DEFAULT_MODEL_PATH = "models/default_resolution_model.h5"
    DEFAULT_MIN_WIDTH = 256
    DEFAULT_MIN_HEIGHT = 256
    DEFAULT_WH_RATIO = 8
    DEFAULT_STEP = 20
    RESOLUTION_HEIGHTS = [
        120,
        144,
        160,
        200,
        240,
        300,
        320,
        360,
        400,
        480,
        500,
        540,
        600,
        640,
        700,
        720,
        768,
        800,
        864,
        900,
        960,
        1000,
        1024,
        1050,
        1080,
        1152,
        1200,
        1440,
        1536,
        1600,
        1620,
        1800,
        1920,
        2000,
        2048,
        2160,
        2400,
        2560,
        3000,
        3072,
        4000,
        4320,
    ]

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        ovveride_model=False,
    ):
        """Create object and download model if needed

        Args:
            model_path (_type_, optional): Path to tensorflow model. Defaults to DEFAULT_MODEL_PATH.
            ovveride_model (bool, optional): Tells if model existing model should be redownloaded. Defaults to False.

        Raises:
            InvalidModel: Invalid model format
        """
        if (not os.path.exists(model_path)) and (self.DEFAULT_MODEL_URL is None):
            logging.error("No default model url")
            raise self.InvalidModel("No default model url")
        if not os.path.exists(model_path) or ovveride_model:
            if model_path == self.DEFAULT_MODEL_PATH:
                logging.info("Downloading default model")
            else:
                logging.info("Downloading model")
            gdown.download(self.DEFAULT_MODEL_URL, model_path, quiet=False)
        try:
            Path(*Path(model_path).parts[:-1]).mkdir(parents=True, exist_ok=True)
            self.model: models.Model = models.load_model(model_path)
        except OSError:
            logging.error("Given model is not valid")
            raise self.InvalidModel("Given model is not valid")

    class ImageTooSmall(Exception):
        pass

    class InvalidModel(Exception):
        pass

    @dataclass
    class FoundedResolution:
        width: int
        height: int
        confidence: float

    def single_prediction_tour(
        self,
        raw: bytes,
        min_width: int = DEFAULT_MIN_WIDTH,
        max_width: int = 0,
        step: int = DEFAULT_STEP,
        min_height: int = DEFAULT_MIN_HEIGHT,
        max_wh_ratio: int = DEFAULT_WH_RATIO,
    ) -> List[FoundedResolution]:
        """Predict line width of image from raw bytes,
        by checking some possible resolutions based on passed argumetns

        Args:
            raw (bytes): Raw bytes of image.
            min_width (int, optional): Min width to check. Defaults to 256.
            max_width (int, optional): Max width to check. By default calculated based on max_wh_ratio.
            step (int, optional): Width change step. Defaults to 50.
            min_height (int, optional): Min height to check. Defaults to 256.
            max_wh_ratio (int, optional): Max ratio beetwen width and height. Defaults to 6.

        Raises:
            self.ImageTooSmall: Given image is too small to check.

        Returns:
            List(FoundedResolution): List of founded resolutions, sorted descending by confidence.
        """
        if len(raw) < self.DEFAULT_MIN_WIDTH * self.DEFAULT_MIN_HEIGHT:
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
            batches.append(tf.convert_to_tensor(img / 255 - 0.5))
        batches = tf.stack(batches)
        predictions: NDArray = np.squeeze(self.model.predict(batches), axis=-1)
        result = {}
        for i, prediction in enumerate(predictions):
            result[i * step + min_width] = prediction
        return [
            self.FoundedResolution(k, len(raw) // k, v)
            for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)
        ]

    def find_resolution(
        self, path: str, best_results: int = 3
    ) -> List[FoundedResolution]:
        """Find resolution of image, from given path

        Args:
            path (str): Path to image.
            best_results (int): Number of best results to return.

        Returns:
            List(FoundedResolution): List of founded resolutions, sorted descending by confidence.
        """
        with open(path, "rb") as f:
            result = []
            raw = f.read()
            predictions = self.single_prediction_tour(raw)
            for i, res in enumerate(predictions):
                if i == best_results:
                    break
                r = self.single_prediction_tour(
                    raw,
                    res.width - self.DEFAULT_STEP,
                    res.width + self.DEFAULT_STEP,
                    1,
                )[0]
                r.height = min(self.RESOLUTION_HEIGHTS, key=lambda x: abs(x - r.height))
                r.width = len(raw) // r.height
                result.append(r)
            result = sorted(result, key=lambda x: x.confidence, reverse=True)
            return result
