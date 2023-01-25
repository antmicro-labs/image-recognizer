from image_recognizer import ImageRecognizer

import fire

def rec(path):
    imgRec = ImageRecognizer()
    data = imgRec.recognize(path)
    return { "format" : data.color_format, 
             "format_condifence" : data.color_format_confidence,
             "img_height" : data.img_height,
             "img_width" : data.img_width,
             "resolution_confidence" : data.resolution_confidence}

if __name__ == "__main__":
    fire.Fire(rec)
