# /a.out rgba32 640 427 byte_images/ picture_nr_1_640x427.raw

import subprocess
import os
import re
import random

DIRECTORY = "byte_images/"
files = os.listdir(DIRECTORY)
FORMATS = ["rgb332", "rgb565", "rgba32", "abgr444", "abgr555", "uyvy", "gray"]

i = 0

for file in files:
    if file.endswith(".raw"):
        i = i + 1
        match = re.search(r"(\d+)x(\d+)", file)
        if match:
            width = match.group(1)
            height = match.group(2)
        else:
            print("No match smthn is wrong")
        print(i)
        args = [
            "./a.out",
            random.choice(FORMATS),
            str(width),
            str(height),
            DIRECTORY,
            file,
        ]
        error = subprocess.run(args)
        if error == 1:
            break
        print(f"Progress {i}")
