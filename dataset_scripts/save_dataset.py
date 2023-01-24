import os
import tensorflow as tf

# Set the directories
GOOD_DIRECTORY = "./good/"
BAD_DIRECTORY = "./bad/"
GOOD = 1
BAD = 0

# Get the list of files in each directory
files1 = os.listdir(GOOD_DIRECTORY)
files2 = os.listdir(BAD_DIRECTORY)

i = 1
# Zip the lists of files and iterate over them
tensor_list = []
label_list = []
for file1, file2 in zip(files1, files2):
    # Open the files
    print(i)
    with open(os.path.join(GOOD_DIRECTORY, file1)) as f1:
        with open(os.path.join(BAD_DIRECTORY, file2)) as f2:
            print(i)
            image_string = open(GOOD_DIRECTORY + file1, "rb").read()

            good_tensor = tf.io.decode_image(image_string, channels=1, dtype=tf.uint8)
            good_tensor = tf.image.resize(good_tensor, [256, 256])
            image_string_bad = open(BAD_DIRECTORY + file2, "rb").read()

            bad_tensor = tf.io.decode_image(
                image_string_bad, channels=1, dtype=tf.uint8
            )
            bad_tensor = tf.image.resize(bad_tensor, [256, 256])

            tensor_list.append(good_tensor)
            label_list.append(GOOD)
            tensor_list.append(bad_tensor)
            label_list.append(BAD)

            i = i + 1

dataset = tf.data.Dataset.from_tensor_slices((tensor_list, label_list))
dataset.save("dataset.tf")
