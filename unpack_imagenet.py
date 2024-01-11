import os
import os.path as osp 
import tarfile
import random 
import cv2
import pdb

VAL_SRC_DIR = '/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar'
DEST_DIR = '/root/autodl-tmp/imagenet'
if not osp.exists(DEST_DIR):
    os.makedirs(DEST_DIR)



if __name__ == '__main__':
    tar = tarfile.open(VAL_SRC_DIR)
    tar.extractall(DEST_DIR)
    print(f"Done writing files into {DEST_DIR}!")
