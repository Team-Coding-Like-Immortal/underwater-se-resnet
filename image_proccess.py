import os

import cv2 as cv
import numpy as np
from PIL.Image import Image
from imutils import *
import albumentations as a


def augmentation_image(img, new):
    scale = 800
    row = img.shape[1]
    col = img.shape[0]
    print(row, col)
    if not (row == 800 or col == 800):
        img_r = scale - col  # 第0个维度填充到800需要的像素点个数 col 425
        img_b = scale - row  # 第1个维度填充到800需要的像素点个数 row 300
        print(img_r, img_b)
        new[:col, :row] = img
        other_img = cv.resize(img, (row, img_r))
        new[col:scale, :row] = other_img
        another_img = cv.resize(img, (img_b, img_r))
        new[:img_r, row:scale] = another_img
        new[col:scale, row:scale] = another_img
        return new
    else:
        return img


image_list = os.listdir('af2020cv-2020-05-09-v5-dev-origin/data')
i = 1
print(image_list)
for img_path in image_list:
    load_path = 'af2020cv-2020-05-09-v5-dev-origin/data/' + str(img_path)
    save_path = 'af2020cv-2020-05-09-v5-dev/data_augmented/' + str(img_path)
    new = cv.imread('Image/black.jpg')
    img = cv.imread(load_path, cv.IMREAD_COLOR)
    new = augmentation_image(img, new)
    print('写入第 %d 张图片' % i)
    cv.imwrite(save_path, new)
    i += 1

# light = a.Compose(
#     [a.RandomBrightnessContrast(p=1),
#      a.RandomGamma(p=1),
#      a.CLAHE(p=1)], p=1
# )
#
# augmented_light = light(image=new)
# new = cv.resize(new, (224, 224))
# cv.imshow('test', new)
#
# # cv.imshow('test', new)

# cv.waitKey(0)
