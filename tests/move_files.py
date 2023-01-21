import os
import re

REGEX = r".+_.+_.+_(\d+)x(\d+)_*(\w+)*.*"
TEST_DATA_FOLDER = "tests/test_data"


filenames = [
    f
    for f in os.listdir(TEST_DATA_FOLDER)
    if os.path.isfile(os.path.join(TEST_DATA_FOLDER, f))
]
data = [(re.search(REGEX, f), f) for f in filenames]

for format in [
    "RGB565",
    "RGBA32",
    "GRAY",
    "UYVY",
    "ABGR32",
    "ABGR444",
    "ABGR555",
    "RGB332",
    "RGB24"
]:
    filepath = os.path.join(TEST_DATA_FOLDER, format)
    if not (os.path.exists(filepath)):
        os.mkdir(filepath)
    for d in data:
        if d[0].group(3) == format or (d[0].group(3) is None and format == "RGB24"):
            os.rename(
                os.path.join(TEST_DATA_FOLDER, d[1]),
                os.path.join(TEST_DATA_FOLDER, format, d[1]),
            )
