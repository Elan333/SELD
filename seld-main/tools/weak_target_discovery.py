import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from PIL import Image

from models import load_config
from models.WTD import (
    TrajectoryConfirmationFilter,
    WeakTargetDetector,
    load_dinov3,
    max_temporal_iou,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 WTD: discover weak-target candidates in unlabelled infrared sequences."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dinov3-repo", default=None)
    parser.add_argument("--dinov3-weights", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def list_sequences(config):
    root = config["data_root"]
    names = config["sequences"] or sorted(
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name, config["split_dir"], config["image_dir"]))
    )
    return [(name, os.path.join(root, name, config["split_dir"])) for name in names]


def list_images(sequence_root, image_dir):
    directory = os.path.join(sequence_root, image_dir)
    return [os.path.join(directory, name) for name in sorted(os.listdir(directory))
            if name.lower().endswith((".jpg", ".png"))]


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.data_root:
        config["data_root"] = os.path.abspath(args.data_root)
    if args.dinov3_repo:
        config["dinov3_repo"] = os.path.abspath(args.dinov3_repo)
    if args.dinov3_weights:
        config["dinov3_weights"] = os.path.abspath(args.dinov3_weights)
    if args.output:
        config["candidates"] = os.path.abspath(args.output)
    if args.max_frames_per_sequence is not None:
        config["max_frames_per_sequence"] = args.max_frames_per_sequence

    torch.manual_seed(config["seed"])
    device = torch.device(args.device or config["device"])
    model = load_dinov3(config["dinov3_weights"], config["dinov3_repo"])
    detector = WeakTargetDetector(model, config, device)

    candidates = []
    for name, sequence_root in list_sequences(config):
        images = list_images(sequence_root, config["image_dir"])
        if config["max_frames_per_sequence"]:
            images = images[:config["max_frames_per_sequence"]]
        tracker = TrajectoryConfirmationFilter(**config["tracking"])
        history = []
        for path in images:
            with Image.open(path) as handle:
                detections, _ = detector.detect(handle.convert("RGB"))
            confirmed = tracker.filter(detections)
            for detection, iou in zip(confirmed, max_temporal_iou(confirmed, history)):
                candidates.append({
                    "img_path": path,
                    "seq_name": name,
                    "box": [int(value) for value in detection["box"]],
                    "score": detection["score"],
                    "compactness": detection["compactness"],
                    "aspect_ratio": detection["aspect_ratio"],
                    "temporal_iou": iou,
                })
            history.append(confirmed)
            if len(history) > config["tracking"]["buffer_len"]:
                history.pop(0)
        print(f"{name}: {len(images)} frames, {len(candidates)} candidates")

    detector.close()
    os.makedirs(os.path.dirname(config["candidates"]), exist_ok=True)
    with open(config["candidates"], "w", encoding="utf-8") as handle:
        json.dump(candidates, handle, indent=2)
    print(f"candidates: {len(candidates)}")
    print(config["candidates"])


if __name__ == "__main__":
    main()
