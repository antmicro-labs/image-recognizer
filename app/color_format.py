import logging
import numpy as np
import tensorflow as tf

from keras import models
from PIL import Image

from app.raw_image_data_previewer.app.core import load_image, get_displayable

COLOR_FORMATS = ["RGB24", "RGB332", "RGB565", "RGBA32", "ABGR444", "ABGR555", "UYVY", "GRAY"]


class ColorFormatFinder:
    IMG_WIDTH = 256
    IMG_HEIGHT = 256

    class InvalidModel(Exception):
        pass

    def __init__(self, model_path: str):
        """Create object and set model

        Args:
            model_path (str): Path to color format model

        Raises:
            InvalidModel: Invalid model format
        """
        try:
            self.model: models.Model = models.load_model(model_path)
        except OSError:
            logging.error("Given model is not valid")
            raise self.InvalidModel("Given model is not valid")

    def generate_color_formats_tensor(self, img_path: str, img_width: int) -> tf.Tensor:
        """Generate a tensor with representations of all color formats for a given image
        Args:
            img_path (str): Path to image
            img_width (int): Image width

        Returns:
            tf.Tensor: Tensor with representations of all color formats
        """
        color_formats_n = len(COLOR_FORMATS)
        imgs = np.empty(shape=(color_formats_n, self.IMG_WIDTH, self.IMG_HEIGHT, 1), dtype=np.float32)

        for i in range(color_formats_n):
            img_data = load_image(img_path, COLOR_FORMATS[i], img_width)
            img = get_displayable(img_data)
            img = Image.fromarray(img, mode="RGB")
            img = img.convert("L")
            img = img.resize((self.IMG_WIDTH, self.IMG_HEIGHT))
            img = np.array(img, dtype=np.float32)
            img = np.expand_dims(img, axis=-1)
            img = (img / 255) - 0.5
            imgs[i] = img

        return tf.convert_to_tensor(imgs)

    def find_color_format(self, img_path: str, img_width: int) -> dict[str, float]:
        """Find color format of the image

        Args:
            img_path (str): Path to image
            img_width (int): Image width

        Returns:
            dict: Dictionary with confidences for the given color format
        """
        color_formats_tensor = self.generate_color_formats_tensor(img_path, img_width)
        color_formats_confidences = {}
        results_tensor = self.model(color_formats_tensor)
        results_list = list(np.squeeze(results_tensor, axis=-1))
        for i in range(len(COLOR_FORMATS)):
            color_formats_confidences[COLOR_FORMATS[i]] = results_list[i]
        return color_formats_confidences
