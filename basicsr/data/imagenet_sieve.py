import os
from PIL import Image

imagenet_path = '/root/autodl-tmp/imagenet'
"""
TODO 

Given a folder `imagenet_path`, look for each image in 
this folder and remove images whose size is smaller than 128*128
"""
# img_list_orig = [file for file in os.scandir(imagenet_path) if file.is_file() and file.name.upper().endswith("JPEG")]
# for item in img_list_orig:
#     width, height = Image.open(os.path.join(imagenet_path, item.name)).size
#     if width < 128 or height < 128:
#         os.unlink(os.path.join(imagenet_path, item.name))

"""
TODO

Remove non-RGB images (single channel gray scale, etc.)
"""
img_list_orig = [file for file in os.scandir(imagenet_path) if file.is_file() and file.name.upper().endswith("JPEG")]
for item in img_list_orig:
    if Image.open(os.path.join(imagenet_path, item.name)).mode != 'RGB':
        os.unlink(os.path.join(imagenet_path, item.name))

