import logging
import gdown
import json

from os.path import join, exists

from app.resolution import ResolutionFinder
from app.color_format import ColorFormatFinder
from dataclasses import dataclass


class ImageRecognizer:
    MODELS_FOLDER = "models"
    MODELS_LINKS_FILE_PATH = "models_links.json"
    MODELS_LINKS_RESOLUTION_KEY = "default_resolution_model"
    MODELS_LINKS_COLOR_FORMAT_KEY = "default_color_format_model"
    DEFAULT_RESOLUTION_MODEL_NAME = "default_resolution_model.h5"
    DEFAULT_COLOR_FORMAT_MODEL_NAME = "default_color_format_model.h5"
    DEFAULT_RESOLUTION_MODEL_IMG_WIDTH = 256
    DEFAULT_RESOLUTION_MODEL_IMG_HEIGHT = 256
    DEFAULT_COLOR_FORMAT_MODEL_IMG_WIDTH = 256
    DEFAULT_COLOR_FORMAT_MODEL_IMG_HEIGHT = 256
    RESOLUTION_RESULTS_N = 5

    DEFAULT_RESOLUTION_MODEL_PATH = join(MODELS_FOLDER, DEFAULT_RESOLUTION_MODEL_NAME)
    DEFAULT_COLOR_FORMAT_MODEL_PATH = join(MODELS_FOLDER, DEFAULT_COLOR_FORMAT_MODEL_NAME)

    class MissingModelsLinksFile(Exception):
        pass

    class InvalidModelsLinksFile(Exception):
        pass

    class InvalidModelLink(Exception):
        pass

    class CustomModelNotFound(Exception):
        pass

    @dataclass
    class Result:
        color_format: str = ""
        img_width: int = 0
        img_height: int = 0
        color_format_confidence: float = 0.0
        resolution_confidence: float = 0.0

    def __init__(
        self,
        resolution_model_path: str = DEFAULT_RESOLUTION_MODEL_PATH,
        color_format_model_path: str = DEFAULT_COLOR_FORMAT_MODEL_PATH,
        resolution_model_img_width: int = DEFAULT_RESOLUTION_MODEL_IMG_WIDTH,
        resolution_model_img_height: int = DEFAULT_RESOLUTION_MODEL_IMG_HEIGHT,
        color_format_model_img_width: int = DEFAULT_COLOR_FORMAT_MODEL_IMG_WIDTH,
        color_format_model_img_height: int = DEFAULT_COLOR_FORMAT_MODEL_IMG_HEIGHT,
    ):
        """Create class object, set all instance variables, download defaults keras models if needed,
         create ResolutionFinder and ColorFormatFinder objects

        Args:
            resolution_model_path (str, optional): Path to resolution keras model file.
             Defaults to DEFAULT_RESOLUTION_MODEL_PATH.
            color_format_model_path (str, optional): Path to color format keras model file.
             Defaults to DEFAULT_COLOR_FORMAT_MODEL_PATH.
            resolution_model_img_width (int, optional): Resolution keras model image width.
             Defaults to DEFAULT_RESOLUTION_MODEL_IMG_WIDTH.
            resolution_model_img_height (int, optional): Resolution keras model image height.
             Defaults to DEFAULT_RESOLUTION_MODEL_IMG_HEIGHT.
            color_format_model_img_width (int, optional): Color format keras model image width.
             Defaults to DEFAULT_COLOR_FORMAT_MODEL_IMG_WIDTH.
            color_format_model_img_height (int, optional): Color format keras model image height.
             Defaults to DEFAULT_COLOR_FORMAT_MODEL_IMG_HEIGHT.

        Raises:
            self.CustomModelNotFound: Custom model not found
        """
        self.resolution_model_path = resolution_model_path
        self.color_format_model_path = color_format_model_path
        self.resolution_model_img_width = resolution_model_img_width
        self.resolution_model_img_height = resolution_model_img_height
        self.color_format_model_img_width = color_format_model_img_width
        self.color_format_model_img_height = color_format_model_img_height

        if not exists(self.resolution_model_path):
            if self.resolution_model_path == self.DEFAULT_RESOLUTION_MODEL_PATH:
                self.download_default_model(self.MODELS_LINKS_RESOLUTION_KEY, self.DEFAULT_RESOLUTION_MODEL_PATH)
            else:
                logging.error("Custom resolution model not found")
                raise self.CustomModelNotFound("Custom resolution model not found")

        if not exists(self.color_format_model_path):
            if self.color_format_model_path == self.DEFAULT_COLOR_FORMAT_MODEL_PATH:
                self.download_default_model(self.MODELS_LINKS_COLOR_FORMAT_KEY, self.DEFAULT_COLOR_FORMAT_MODEL_PATH)
            else:
                logging.error("Custom resolution model not found")
                raise self.CustomModelNotFound("Custom resolution model not found")

        self.resolution_finder = ResolutionFinder(
            self.resolution_model_path, self.resolution_model_img_width, self.resolution_model_img_height
        )

        self.color_format_finder = ColorFormatFinder(
            self.color_format_model_path, self.color_format_model_img_width, self.color_format_model_img_height
        )

    def download_default_model(self, model_key: str, model_path: str) -> None:
        """Downloads defaults keras model

        Args:
            model_key (str): Model key in JSON file with models links
            model_path (str): Path to location where downloaded model should be saved

        Raises:
            self.MissingModelsLinksFile: Models links file not found
            self.InvalidModelsLinksFile: Models links file is invalid
            self.InvalidModelLink: Given model link is invalid
        """
        try:
            with open(self.MODELS_LINKS_FILE_PATH, "r", encoding="utf-8") as file:
                link = json.load(file)[model_key]
        except FileNotFoundError:
            logging.error("Models links file not found")
            raise self.MissingModelsLinksFile("Models links file not found")
        except ValueError:
            logging.error("Models links file is invalid")
            raise self.InvalidModelsLinksFile("Models links file is invalid")

        if gdown.download(link, model_path) is None:
            logging.error(f"{model_key} model link is invalid")
            raise self.InvalidModelLink(f"{model_key} model link is invalid")

    def recognize(self, raw_img_path: str) -> Result:
        """Recognize raw image - find correct color format and resolution

        Args:
            raw_img_path (str): path to raw image

        Returns:
            Result: object of Result class with found color format, resolution and confidences for them
        """
        print(f"Searching for the top {self.RESOLUTION_RESULTS_N} best resolutions...")
        resolutions = self.resolution_finder.find_resolution(raw_img_path, best_results=self.RESOLUTION_RESULTS_N)
        best_result = self.Result()

        for resolution in resolutions:
            print(f"Searching for the best color format for {resolution.width}x{resolution.height} resolution...")
            color_formats_confidences = self.color_format_finder.find_color_format(raw_img_path, resolution.width)
            best_color_format = max(color_formats_confidences, key=color_formats_confidences.get)
            if color_formats_confidences[best_color_format] > best_result.color_format_confidence:
                best_result.color_format = best_color_format
                best_result.img_width = resolution.width
                best_result.img_height = resolution.height
                best_result.color_format_confidence = color_formats_confidences[best_color_format]
                best_result.resolution_confidence = resolution.confidence

        print("Determining the best result...")
        return best_result
