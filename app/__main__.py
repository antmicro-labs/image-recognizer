from os import environ

environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
environ["CUDA_VISIBLE_DEVICES"] = "-1"

from app.image_recognizer import ImageRecognizer # noqa

image_recognizer = ImageRecognizer()
result = image_recognizer.recognize("tests/test_data/abgr444/picture_nr_5_640x480_abgr444.raw")
print(result)
