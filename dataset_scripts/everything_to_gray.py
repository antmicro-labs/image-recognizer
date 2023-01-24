import os
import subprocess

directory = "magick_good/"
files = os.listdir(directory)

i = 1
for file in files:
    print("first: ", i)
    args = ["./magick", directory+file, "-colorspace", "gray", "-depth", "8", f"good/{i}.png"]
    try:
        error = subprocess.run(args)
    except:
        pass
    i = i + 1

directory1 = "magick_bad/"
files1 = os.listdir(directory1)
i = 1

for file1 in files1:
    print("second: ", i)
    args1 = ["./magick", directory1+file1, "-colorspace", "gray", "-depth", "8", f"bad/{i}.png"]
    try:
        error = subprocess.run(args1)
    except:
        pass
    i = i + 1

