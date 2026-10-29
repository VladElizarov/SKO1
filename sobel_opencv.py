import cv2
import numpy as np
import time


def sobel_opencv(gray_image):
    start = time.perf_counter()

    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    result = np.clip(magnitude, 0, 255).astype(np.uint8)

    elapsed = time.perf_counter() - start
    return result, elapsed
