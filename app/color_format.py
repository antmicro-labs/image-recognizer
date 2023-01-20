import logging
import cv2
import numpy as np
import tensorflow as tf

from keras import models

from app.raw_image_data_previewer.app.core import load_image, get_displayable

COLOR_FORMATS_RATIOS = {
    "RGB24": 3,
    "RGB332": 1,
    "RGB565": 2,
    "RGBA32": 4,
    "ABGR444": 12 / 8,
    "ABGR555": 15 / 8,
    "UYVY": 1,
    "GRAY": 1,
}


class ColorFormatFinder:
    class InvalidModel(Exception):
        pass

    def __init__(self, model_path: str, model_img_width: int, model_img_height: int):
        """Create class object and set keras model

        Args:
            model_path (str): Path to color format model
            model_img_width (int): Model image width
            model_img_height (int): Model image height

        Raises:
            InvalidModel: Invalid model format
        """
        self.model_img_width = model_img_width
        self.model_img_height = model_img_height
        try:
            self.model: models.Model = models.load_model(model_path)
        except OSError:
            logging.error("Given color format model is not valid")
            raise self.InvalidModel("Given color format model is not valid")

    def generate_color_formats_tensor(self, img_path: str, img_width: int) -> tf.Tensor:
        """Generate a tensor with representations of all color formats for a given image
        Args:
            img_path (str): Path to image
            img_width (int): Image width

        Returns:
            tf.Tensor: Tensor with representations of all color formats
        """
        imgs = np.empty(
            shape=(len(COLOR_FORMATS_RATIOS), self.model_img_width, self.model_img_height, 1), dtype=np.float32
        )

        i = 0
        for color_format, resolution_ratio in COLOR_FORMATS_RATIOS.items():
            img_data = load_image(img_path, color_format, int(img_width / resolution_ratio))
            img = get_displayable(img_data)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.model_img_width, self.model_img_height), interpolation=cv2.INTER_AREA)
            img = np.expand_dims(img, axis=-1)
            img = (img / 255) - 0.5
            imgs[i] = img
            i += 1

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
        color_formats = list(COLOR_FORMATS_RATIOS.keys())
        for i in range(len(color_formats)):
            color_formats_confidences[color_formats[i]] = results_list[i]
        return color_formats_confidences
