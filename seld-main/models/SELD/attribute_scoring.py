import cv2
import numpy as np


def min_max(values):
    low, high = values.min(), values.max()
    return (values - low) / (high - low + 1e-8)


def local_contrast(gray, box):
    height, width = gray.shape
    x1, y1, x2, y2 = [int(value) for value in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    patch = gray[y1:y2, x1:x2]
    box_width, box_height = x2 - x1, y2 - y1
    margin = max(box_width, box_height)
    sx1, sy1 = max(0, x1 - margin), max(0, y1 - margin)
    sx2, sy2 = min(width, x2 + margin), min(height, y2 + margin)
    surround = gray[sy1:sy2, sx1:sx2].copy()
    inner_y, inner_x = y1 - sy1, x1 - sx1
    surround[inner_y:inner_y + box_height, inner_x:inner_x + box_width] = 0
    pixels = surround[surround > 0]
    return abs(float(patch.mean()) - float(pixels.mean())) / (float(pixels.std()) + 1.0)


def compute_final_scores(candidates, w_t):
    groups = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(candidate["img_path"], []).append(index)
    contrast = np.zeros(len(candidates), dtype=np.float64)
    for path, indices in groups.items():
        gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        for index in indices:
            contrast[index] = local_contrast(gray, candidates[index]["box"])
    temporal = np.array([candidate["temporal_iou"] for candidate in candidates], dtype=np.float64)
    contrast_norm = min_max(contrast)
    temporal_norm = min_max(temporal)
    for index, candidate in enumerate(candidates):
        candidate["local_contrast"] = float(contrast[index])
        candidate["temporal_consistency"] = float(temporal_norm[index])
        candidate["final_score"] = float(contrast_norm[index] * (1.0 + w_t * temporal_norm[index]))
    return sorted(candidates, key=lambda candidate: candidate["final_score"], reverse=True)
