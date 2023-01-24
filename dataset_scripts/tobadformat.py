import os
import random
import re

def pick_random(my_list, element_to_avoid):
  new_list = [e for e in my_list if e != element_to_avoid]
  return random.choice(new_list)

DIRECTORY = 'good_imgs/'
files = os.listdir(DIRECTORY)
COLOR_FORMATS = ["RGBA32", "BGRA32", "RGB332", "RGB565", "ARGB444", "ABGR555", "YUY2", "YVYU", "NV12", "I420", "RGGB" ]

i = 0
for file in files:
    i = i + 1
    match = re.search(r"_(?P<width>\d+)x(?P<height>\d+)_(?P<format>\w+)\.", file)
    if match:
        width = match.group("width")
        height = match.group("height")
        file_format = match.group("format")
    else:            
        match1 = re.search(r"_(?P<width>\d+)x(?P<height>\d+)\.",file)
        width = match1.group("width")
        height = match1.group("height")
        file_format = "rgb24"


    COLOR_FORMAT = pick_random(COLOR_FORMATS,file_format.upper())
    os.system(f'raviewer -f {DIRECTORY}/{file} -w {width} -H {height} -c {COLOR_FORMAT} -e ./bad_format_images/{file[:-4]}_{COLOR_FORMAT}')
    print(f"Progress: {i}")
