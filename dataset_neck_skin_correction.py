"""
Make updated body segmentation with new neck/skin label
"""


import os
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import sys
import shutil

N_CLASSES = 21
fine_width = 192
fine_height = 256

# colour map for LIP dataset (plus extra)
label_colours = [(0, 0, 0),  # 0=Background
                 (128, 0, 0),  # 1=Hat
                 (255, 0, 0),  # 2=Hair
                 (0, 85, 0),   # 3=Glove
                 (170, 0, 51),  # 4=Sunglasses
                 (255, 85, 0),  # 5=UpperClothes
                 (0, 0, 85),  # 6=Dress
                 (0, 119, 221),  # 7=Coat
                 (85, 85, 0),  # 8=Socks
                 (0, 85, 85),  # 9=Pants
                 (85, 51, 0),  # 10=Jumpsuits
                 (52, 86, 128),  # 11=Scarf
                 (0, 128, 0),  # 12=Skirt
                 (0, 0, 255),  # 13=Face
                 (51, 170, 221),  # 14=LeftArm
                 (0, 255, 255),  # 15=RightArm
                 (85, 255, 170),  # 16=LeftLeg
                 (170, 255, 85),  # 17=RightLeg
                 (255, 255, 0),  # 18=LeftShoe
                 (255, 170, 0),  # 19=RightShoe
                 (189, 183, 107)  # 20=Neck    # new added
                 ]

(cv_major, _, _) = cv2.__version__.split(".")
if cv_major != '4' and cv_major != '3':
    print('doesnot support opencv version')
    sys.exit()


def decode_labels(mask):
    """Decode segmentation masks.
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: num of classes
    Returns:
      A RGB image of the same size as the input.
    """

    mask = np.expand_dims(mask, axis=2)
    h, w, c = mask.shape

    outputs = np.zeros((h, w, 3), dtype=np.uint8)

    par_img = Image.new('RGB', (w, h))
    pixels = par_img.load()
    for j_, j in enumerate(mask[:, :, 0]):
        for k_, k in enumerate(j):
            if k < N_CLASSES:
                pixels[k_, j_] = label_colours[k]
    outputs = np.array(par_img)

    return outputs


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


def shape_from_contour(img, contour):
    dummy_mask = np.zeros((img.shape[0], img.shape[1], 3))
    dummy_mask = cv2.drawContours(
        dummy_mask, [contour], 0, (1, 0, 0), thickness=cv2.FILLED)
    x, y = np.where(dummy_mask[:, :, 0] == 1)
    inside_points = np.stack((x, y), axis=-1)
    return inside_points


#
# relabel the segmented mask with neck
# dir_dir  : input image file dir  path
# image_name : image file name
# mask_dir : original mask dir path
# mask_name : original mask image file
# save_dir  : the re-labeled dir path (same name as mask_name)
#
#
def update_image_segmentation(data_dir, mask_dir, image_name, mask_name, save_dir=None, save_vis=True):
    print(image_name)

    # define paths
    img_pth = os.path.join(data_dir, image_name)
    seg_pth = os.path.join(mask_dir, mask_name)

    updated_seg_pth = None
    updated_seg_vis_pth = None
    if save_dir is not None:
        updated_seg_pth = os.path.join(save_dir, mask_name)
        if save_vis:
            updated_seg_vis_pth = updated_seg_pth.replace("image-parse-new", "image-parse-new-vis")
            if not os.path.exists(updated_seg_vis_pth):
                os.makedirs(updated_seg_vis_pth)

    # Load image and make binary body mask
    img = cv2.imread(img_pth)

    # Load the segmentation in grayscale and make binary mask
    segmentation = Image.open(seg_pth)

    # the png file should be 1-ch but it is 3 ch ^^;
    gray = cv2.imread(seg_pth, cv2.IMREAD_GRAYSCALE)
    # print('shape of seg:', seg_pth, ':', gray.shape)
    # _, seg_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  # why 10? bg is 0
    _, seg_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    body_mask = body_detection(img, seg_mask)

    # Get the neck/skin region (plus extra mis-segmented)
    upper_body = body_mask - seg_mask
    upper_body[upper_body > 0] = 20
    upper_body_vis = upper_body.copy()

    # location info: @TODO by joint locations (neck should be between neck and hips vertically, between shoulder horizontally)
    # print(upper_body.shape)
    height, width = upper_body.shape
    upper_body[height//2:, :] = 0
    # noise reduction

    # get contours
    if cv_major == '4':
        contours, hier = cv2.findContours(
            upper_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    elif cv_major == '3':
        _, contours, hier = cv2.findContours(
            upper_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        return

    neck = None

    if len(contours) > 0:
        # draw in blue the contours that were founded
        cv2.drawContours(upper_body_vis, contours, -1, 255, 3)

        # find the biggest area
        c_neck = max(contours, key=cv2.contourArea)

        neck = shape_from_contour(img, c_neck)

        x, y, w, h = cv2.boundingRect(c_neck)
        # draw the book contour (in green)
        cv2.rectangle(upper_body_vis, (x, y), (x + w, y + h), (170, 230, 0), 2)

    # make neck region mask
    neck_mask = np.zeros((fine_height, fine_width)).astype(np.int)
    for each in neck:
        neck_mask[each[0]][each[1]] = 20

    # Add neck/skin to segmentation
    result = segmentation + neck_mask

    # handle overlapped pixels
    for i in range(1, 20):
        result[result == 20 + i] = i

    # save new segmentation
    if updated_seg_pth is not None:
        cv2.imwrite(updated_seg_pth, result)
        if save_vis:
            msk = decode_labels(result)
            parsing_im = Image.fromarray(msk)
            parsing_im.save('{}/{}_vis.png'.format(updated_seg_vis_pth, mask_name[:-4]))
    else:  # display for checking

        plt.suptitle(image_name)
        plt.subplot(1, 4, 1)
        plt.title("input")
        plt.axis('off')
        plt.imshow(img[:, :, ::-1])
        plt.subplot(1, 4, 2)
        plt.title("body silhouette")
        plt.axis('off')
        plt.imshow(body_mask)
        plt.subplot(1, 4, 3)
        plt.title("orig. mask")
        plt.axis('off')
        plt.imshow(segmentation)
        plt.subplot(1, 4, 4)
        plt.title("relabeled")
        plt.axis('off')
        msk = decode_labels(result)         # ???
        parsing_im = Image.fromarray(msk)   # ???
        plt.imshow(parsing_im)
        plt.show()


def main():
    # define paths

    root_dir = "data/"
    updated_seg_folder = "image-parse-new"

    # data_mode = "train"
    data_mode = "test"
    image_folder = "image"
    seg_folder = "image-parse"

    image_dir = os.path.join(os.path.join(root_dir, data_mode), image_folder)
    seg_dir = os.path.join(os.path.join(root_dir, data_mode), seg_folder)
    if updated_seg_folder is not None:
        updated_seg_dir = os.path.join(os.path.join(
            root_dir, data_mode), updated_seg_folder)
        if not os.path.exists(updated_seg_dir):
            os.makedirs(updated_seg_dir)
    else:
        updated_seg_dir = None

    image_list = sorted(os.listdir(image_dir))
    masks_list = sorted(os.listdir(seg_dir))

    try:
        shutil.rmtree(os.path.join(image_dir, '.ipynb_checkpoints'))
        shutil.rmtree(os.path.join(seg_dir, '.ipynb_checkpoints'))
    except:
        print("Clean")

    for each in zip(image_list, masks_list):
        mask = each[0].replace("jpg", "png")
        update_image_segmentation(
            image_dir, seg_dir, each[0], mask, updated_seg_dir)


if __name__ == '__main__':
    main()
