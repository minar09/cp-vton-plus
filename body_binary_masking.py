"""
Make updated body shape from updated segmentation
"""

import os
import numpy as np
import cv2
from PIL import Image
import sys


(cv_major, _, _) = cv2.__version__.split(".")
if cv_major != '4' and cv_major != '3':
    print('doesnot support opencv version')
    sys.exit()


# @TODO this is too simple and pixel based algorithm
def body_detection(image, seg_mask):
    # binary thresholding by blue ?
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 120])
    upper_blue = np.array([180, 38, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(image, image, mask=mask)

    # binary threshold by green ?
    b, g, r = cv2.split(result)
    filter = g.copy()
    ret, mask = cv2.threshold(filter, 10, 255, 1)

    # at least original segmentation is FG
    mask[seg_mask] = 1

    return mask


def make_body_mask(data_dir, seg_dir, image_name, mask_name, save_dir=None):
    print(image_name)

    # define paths
    img_pth = os.path.join(data_dir, image_name)
    seg_pth = os.path.join(seg_dir, mask_name)

    mask_path = None
    if save_dir is not None:
        mask_path = os.path.join(save_dir, mask_name)

    # Load images
    img = cv2.imread(img_pth)
    # segm = Image.open(seg_pth)
    # the png file should be 1-ch but it is 3 ch ^^;
    gray = cv2.imread(seg_pth, cv2.IMREAD_GRAYSCALE)
    _, seg_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    body_mask = body_detection(img, seg_mask)
    body_mask = body_mask + seg_mask
    body_mask[seg_mask] = 1
    cv2.imwrite(mask_path, body_mask)


def main():
    # define paths

    # root_dir = "data/viton_resize"
    root_dir = "data/"
    mask_folder = "image-mask"
    seg_folder = "image-parse-new"

    # data_mode = "train"
    data_mode = "test"
    image_folder = "image"

    image_dir = os.path.join(os.path.join(root_dir, data_mode), image_folder)
    seg_dir = os.path.join(os.path.join(root_dir, data_mode), seg_folder)

    image_list = sorted(os.listdir(image_dir))
    seg_list = sorted(os.listdir(seg_dir))

    mask_dir = os.path.join(os.path.join(root_dir, data_mode), mask_folder)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    for each in zip(image_list, seg_list):
        make_body_mask(image_dir, seg_dir, each[0], each[1], mask_dir)


if __name__ == '__main__':
    main()
