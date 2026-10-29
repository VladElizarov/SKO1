import numpy as np
import time


def sobel_native(gray_image):
    start = time.perf_counter()

    img = gray_image.astype(np.float64)
    p = np.pad(img, pad_width=1, mode='edge')

    grad_x = (
        -1 * p[0:-2, 0:-2] + 1 * p[0:-2, 2:] +
        -2 * p[1:-1, 0:-2] + 2 * p[1:-1, 2:] +
        -1 * p[2:,   0:-2] + 1 * p[2:,   2:]
    )

    grad_y = (
        -1 * p[0:-2, 0:-2] + -2 * p[0:-2, 1:-1] + -1 * p[0:-2, 2:] +
         1 * p[2:,   0:-2] +  2 * p[2:,   1:-1] +  1 * p[2:,   2:]
    )

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    result = np.clip(magnitude, 0, 255).astype(np.uint8)

    elapsed = time.perf_counter() - start
    return result, elapsed
