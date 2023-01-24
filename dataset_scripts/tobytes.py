import os
from PIL import Image

DIRECTORY = "./original_images/"
SAVE_DIRECTORY = "./byte_images/"
files = os.listdir(DIRECTORY)

i = 1

for file in files:
   print(i)
   if file.endswith(".jpg"):
       with Image.open(DIRECTORY + file) as im:
            width, height = im.size
            raw_im = im.tobytes()
            file_path = os.path.join(SAVE_DIRECTORY, f"picture_nr_{i}_{width}x{height}.raw")
            with open(file_path, 'wb') as f:
                f.write(raw_im)
            i = i + 1
