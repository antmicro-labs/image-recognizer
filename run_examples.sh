#!/bin/bash
source ./.venv/bin/activate

python app recognize tests/test_data/abgr444/picture_nr_5_640x480_abgr444.raw color_format_confidence
python app recognize tests/test_data/RGB24/picture_nr_1_640x427.raw color_format
python app recognize tests/test_data/abgr444/picture_nr_5_640x480_abgr444.raw img_width
