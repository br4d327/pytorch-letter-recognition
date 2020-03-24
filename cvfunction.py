import cv2 as cv
import numpy as np
from PIL import Image


def image_processing(path):
    pil_image = Image.open(path)
    opencvImage = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    blur = cv.blur(opencvImage, (7, 7))
    ret, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    erosion = cv.erode(th, kernel, iterations=1)
    contours, _ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(cnt)
    crop = erosion[y:y + h, x:x + w]
    return crop


def get_square(image, square_size):
    height, width = image.shape
    if (height > width):
      differ = height
    else:
      differ = width
    differ += 4
    mask = np.zeros((differ, differ), dtype="uint8")
    x_pos = int((differ-width)/2)
    y_pos = int((differ-height)/2)
    mask[y_pos:y_pos+height, x_pos:x_pos+width] = image[0:height, 0:width]
    mask = cv.resize(mask, (square_size, square_size), interpolation=cv.INTER_AREA)
    return mask

