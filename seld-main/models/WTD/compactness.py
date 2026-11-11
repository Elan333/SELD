import numpy as np
from scipy.ndimage import binary_erosion, label


def compute_compactness(component):
    area = float(component.sum())
    if area < 1:
        return 0.0
    perimeter = float((component & ~binary_erosion(component)).sum())
    if perimeter < 1:
        return 0.0
    return area / perimeter ** 2
