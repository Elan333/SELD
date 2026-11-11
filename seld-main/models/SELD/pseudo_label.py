import json
import os
import shutil

import cv2
import yaml


def select_top(candidates, top_ratio):
    count = max(1, int(len(candidates) * top_ratio))
    return candidates[:count]


def sequence_short_name(sequence):
    return sequence.replace("Fixed-wing-UAV-", "").replace("'", "p")


def build_dataset(candidates, output_dir):
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    json_dir = os.path.join(output_dir, "jsons")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    groups = {}
    for candidate in candidates:
        groups.setdefault(candidate["img_path"], []).append(candidate)

    coco_images = []
    coco_annotations = []
    annotation_id = 1
    for image_id, (path, boxes) in enumerate(groups.items()):
        image = cv2.imread(path)
        height, width = image.shape[:2]
        sequence = boxes[0]["seq_name"]
        extension = os.path.splitext(path)[1]
        stem = os.path.splitext(os.path.basename(path))[0]
        name = f"{image_id:04d}_{sequence_short_name(sequence)}_IR_{stem}"
        shutil.copy2(path, os.path.join(images_dir, name + extension))

        lines = []
        for box in boxes:
            x1, y1, x2, y2 = box["box"]
            lines.append(
                f"0 {(x1 + x2) / (2.0 * width):.6f} {(y1 + y2) / (2.0 * height):.6f} "
                f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
            )
        with open(os.path.join(labels_dir, name + ".txt"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

        coco_images.append({
            "id": image_id,
            "file_name": name + extension,
            "width": int(width),
            "height": int(height),
            "sequence_name": sequence,
            "frame_id": int(stem.split("_")[-1]),
        })
        for box in boxes:
            x1, y1, x2, y2 = [int(round(value)) for value in box["box"]]
            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 0,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": int((x2 - x1) * (y2 - y1)),
                "iscrowd": 0,
                "final_score": round(box["final_score"], 4),
                "attributes": {
                    "detection_score": round(box["score"], 4),
                    "temporal_consistency": round(box["temporal_consistency"], 4),
                    "compactness": round(box["compactness"], 4),
                    "local_contrast": round(box["local_contrast"], 4),
                },
            })
            annotation_id += 1

    coco = {
        "info": {
            "description": "SELD pseudo labels for MMFW-UAV infrared sequences",
            "version": "1.0",
            "year": 2026,
        },
        "categories": [{"id": 0, "name": "aircraft", "supercategory": "uav"}],
        "images": coco_images,
        "annotations": coco_annotations,
    }
    with open(os.path.join(json_dir, "pseudo_labels_coco.json"), "w", encoding="utf-8") as handle:
        json.dump(coco, handle, indent=2, ensure_ascii=False)

    paths = list(groups.keys())
    with open(os.path.join(output_dir, "paths.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(paths) + "\n")

    data_yaml = {
        "path": os.path.abspath(output_dir),
        "train": "images",
        "val": "images",
        "names": {0: "aircraft"},
    }
    with open(os.path.join(output_dir, "data.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False)

    return {
        "images": len(coco_images),
        "boxes": len(coco_annotations),
        "dataset": os.path.abspath(output_dir),
    }
