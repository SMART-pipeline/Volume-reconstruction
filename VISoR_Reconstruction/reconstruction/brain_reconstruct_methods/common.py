import SimpleITK as sitk
import cv2
import numpy as np


def fill_outside(img: sitk.Image, value: int):
    img = sitk.GetArrayFromImage(img)
    img[0, 0] = 0
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
    cv2.floodFill(img,
                  mask,
                  (0, 0), value, value, value, cv2.FLOODFILL_FIXED_RANGE)
    img[img.shape[0] - 1, 0] = 0
    cv2.floodFill(img,
                  mask,
                  (0, img.shape[0] - 1), value, value, value, cv2.FLOODFILL_FIXED_RANGE)
    img[img.shape[0] - 1, img.shape[1] - 1] = 0
    cv2.floodFill(img,
                  mask,
                  (img.shape[1] - 1, img.shape[0] - 1), value, value, value, cv2.FLOODFILL_FIXED_RANGE)
    img[0, img.shape[1] - 1] = 0
    cv2.floodFill(img,
                  mask,
                  (img.shape[1] - 1, 0), value, value, value, cv2.FLOODFILL_FIXED_RANGE)
    img = sitk.GetImageFromArray(img)
    return img