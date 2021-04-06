import os
from PIL import Image, ImageFilter




new_path = "./Data/is_on_edge/Blurred/pos/"
path = "./Data/is_on_edge/Clear/pos/"
files = os.listdir(path)


for file in files:
    or_image = Image.open(path+file)
    boxImage = or_image.filter(ImageFilter.BoxBlur(7))
    boxImage.save(new_path+file)


new_path = "./Data/is_on_edge/Blurred/neg/"
path = "./Data/is_on_edge/Clear/neg/"
files = os.listdir(path)


for file in files:
    or_image = Image.open(path+file)
    boxImage = or_image.filter(ImageFilter.BoxBlur(7))
    boxImage.save(new_path+file)

