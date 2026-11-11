import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label

from .compactness import compute_compactness


def saliency_contrast_ratio(saliency_map):
    values = saliency_map.flatten()
    high = values >= np.percentile(values, 90)
    low = ~high
    return abs(values[high].mean() - values[low].mean()) / (values[low].std() + 1e-8)


def detect_with_strict_morphology(saliency_map, contrast_ratio, offset, params):
    if contrast_ratio < params["cr_threshold"]:
        return []
    size = params["roi_size"]
    tensor = torch.from_numpy(saliency_map).float()[None, None]
    with torch.no_grad():
        upsampled = F.interpolate(tensor, size=(size, size), mode="bicubic", align_corners=False)
    response = np.clip(upsampled.squeeze().numpy(), 0, None)
    normalized = (response - response.min()) / (response.max() - response.min() + 1e-8)
    flat = normalized.flatten()
    mean, std = flat.mean(), flat.std()
    components = None
    count = 0
    for level in (mean + 1.5 * std, mean + std, mean + 0.5 * std):
        components, count = label((normalized > level).astype(np.uint8))
        if count > 0:
            break
    if count == 0:
        components, count = label((normalized > np.percentile(flat, 97)).astype(np.uint8))

    patch_size = params["patch_size"]
    border = params["border_margin"]
    detections = []
    for index in range(1, count + 1):
        component = components == index
        if component.sum() < 3:
            continue
        ys, xs = np.where(component)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        width, height = x2 - x1 + 1, y2 - y1 + 1
        area = width * height
        compactness = compute_compactness(component)
        if compactness < params["min_compactness"] * params["compactness_factor"]:
            continue
        if area < params["min_patch_area"] * patch_size ** 2:
            continue
        if area > params["max_patch_area"] * patch_size ** 2:
            continue
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > params["max_aspect_ratio"]:
            continue
        on_border = (x1 <= border or x2 >= size - 1 - border or
                     y1 <= border or y2 >= size - 1 - border)
        score = float(normalized[component].mean())
        if compactness < params["min_compactness"]:
            score *= params["compactness_factor"]
        if on_border:
            score *= 0.5
        detections.append({
            "box": [offset[0] + x1, offset[1] + y1, offset[0] + x2, offset[1] + y2],
            "score": score,
            "compactness": compactness,
            "aspect_ratio": aspect_ratio,
        })

    detections.sort(key=lambda detection: detection["score"], reverse=True)
    if not detections:
        peak = float(flat.max())
        if peak > mean + 0.25 * std:
            peak_y, peak_x = np.unravel_index(int(np.argmax(flat)), (size, size))
            x1 = max(0, peak_x - patch_size)
            x2 = min(size - 1, peak_x + patch_size)
            y1 = max(0, peak_y - patch_size)
            y2 = min(size - 1, peak_y + patch_size)
            detections.append({
                "box": [offset[0] + x1, offset[1] + y1, offset[0] + x2, offset[1] + y2],
                "score": peak,
                "compactness": 0.05,
                "aspect_ratio": 1.0,
            })
    return detections[:params["max_boxes_per_roi"]]
