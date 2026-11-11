import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter


def min_max_normalize(values):
    low, high = values.min(), values.max()
    if high - low > 1e-8:
        return (values - low) / (high - low)
    return np.zeros_like(values)


def compute_lcm(gray, scales):
    image = gray.astype(np.float64)
    height, width = image.shape
    lcm = np.zeros((height, width), dtype=np.float64)
    for scale in scales:
        local_mean = uniform_filter(image, size=scale, mode="reflect")
        local_mean = np.maximum(local_mean, 1e-8)
        lcm = np.maximum(lcm, np.abs(image - local_mean) / local_mean)
    return lcm


def preprocess_sobel(image):
    array = np.array(image)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    magnitude = np.clip(magnitude / (magnitude.max() + 1e-8) * 255, 0, 255).astype(np.uint8)
    sobel_rgb = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(cv2.addWeighted(array, 1.0, sobel_rgb, 0.5, 0))
