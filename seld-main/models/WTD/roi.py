import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def find_top_k_rois(lcm_map, k, roi_size, sigma):
    height, width = lcm_map.shape
    half = roi_size // 2
    valid = np.zeros((height, width), dtype=bool)
    valid[half:height - half, half:width - half] = True
    response = np.where(valid, gaussian_filter(lcm_map, sigma), -np.inf)
    rois = []
    for _ in range(k):
        center_y, center_x = np.unravel_index(int(np.argmax(response)), (height, width))
        if response[center_y, center_x] <= 0:
            break
        left, top = int(center_x - half), int(center_y - half)
        rois.append({
            "center": (int(center_x), int(center_y)),
            "offset": (left, top),
            "left": max(0, left),
            "top": max(0, top),
            "right": min(width, left + roi_size),
            "bottom": min(height, top + roi_size),
            "score": float(lcm_map[center_y, center_x]),
        })
        response[max(0, center_y - half):min(height, center_y + half),
                 max(0, center_x - half):min(width, center_x + half)] = -np.inf
    return rois


def crop_roi(image, roi, roi_size):
    patch = image.crop((roi["left"], roi["top"], roi["right"], roi["bottom"]))
    if patch.size == (roi_size, roi_size):
        return patch
    canvas = Image.new("RGB", (roi_size, roi_size))
    canvas.paste(patch, ((roi_size - patch.size[0]) // 2, (roi_size - patch.size[1]) // 2))
    return canvas
