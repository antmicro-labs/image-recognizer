from app.color_format import ColorFormatFinder

color_format_finder = ColorFormatFinder("models/default_color_formats_model.h5")
#color_format_finder.generate_color_formats_tensor("tests/test_data/gray/picture_nr_2_640x481_gray.raw", 640)
print(color_format_finder.find_color_format("tests/test_data/abgr555/picture_nr_11_640x604_abgr555.raw", 640))
