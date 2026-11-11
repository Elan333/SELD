import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from PIL import Image

from models import load_config
from models.WTD import WeakTargetDetector, calculate_iou, load_dinov3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate proposals on MMFW-UAV at IoU 0.15 (GT-level recall, mean IoU, FPS, proposals/frame)."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--method", choices=("wtd", "yolo"), default="wtd")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dinov3-repo", default=None)
    parser.add_argument("--dinov3-weights", default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
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


def load_gt(image_path, annotation_dir):
    annotation = os.path.join(
        os.path.dirname(os.path.dirname(image_path)),
        annotation_dir,
        os.path.splitext(os.path.basename(image_path))[0] + ".xml",
    )
    if not os.path.exists(annotation):
        return None
    box = ET.parse(annotation).getroot().find("object/bndbox")
    return [int(box.find(key).text) for key in ("xmin", "ymin", "xmax", "ymax")]


def evaluate_sequence(detect, images, annotation_dir, iou_threshold):
    started = time.time()
    total_gt = 0
    matched = 0
    proposals = 0
    iou_sum = 0.0
    iou_count = 0
    for path in images:
        boxes = detect(path)
        proposals += len(boxes)
        gt = load_gt(path, annotation_dir)
        if gt is None:
            continue
        total_gt += 1
        best = max((calculate_iou(box, gt) for box in boxes), default=0.0)
        if best >= iou_threshold:
            matched += 1
        if best > 0:
            iou_sum += best
            iou_count += 1
    elapsed = time.time() - started
    return {
        "frames": len(images),
        "gt": total_gt,
        "matched": matched,
        "proposals": proposals,
        "iou_sum": iou_sum,
        "iou_count": iou_count,
        "recall": matched / total_gt if total_gt else 0.0,
        "mean_iou": iou_sum / iou_count if iou_count else 0.0,
        "fps": len(images) / elapsed if elapsed > 0 else 0.0,
        "avg_proposals": proposals / len(images) if images else 0.0,
        "elapsed": elapsed,
    }


def summarize(results):
    frames = sum(result["frames"] for _, result in results)
    total_gt = sum(result["gt"] for _, result in results)
    matched = sum(result["matched"] for _, result in results)
    proposals = sum(result["proposals"] for _, result in results)
    iou_sum = sum(result["iou_sum"] for _, result in results)
    iou_count = sum(result["iou_count"] for _, result in results)
    elapsed = sum(result["elapsed"] for _, result in results)
    return {
        "frames": frames,
        "gt": total_gt,
        "matched": matched,
        "proposals": proposals,
        "recall": matched / total_gt if total_gt else 0.0,
        "mean_iou": iou_sum / iou_count if iou_count else 0.0,
        "fps": frames / elapsed if elapsed > 0 else 0.0,
        "avg_proposals": proposals / frames if frames else 0.0,
        "elapsed": elapsed,
    }


def yolo_detector(weights, config):
    from ultralytics import YOLO

    model = YOLO(weights)

    def detect(path):
        result = model.predict(
            path,
            imgsz=config["yolo"]["imgsz"],
            conf=config["detection"]["conf"],
            iou=config["yolo"]["nms_iou"],
            device=config["yolo"]["device"],
            verbose=False,
        )[0]
        return result.boxes.xyxy.cpu().numpy().tolist()

    return detect


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.data_root:
        config["data_root"] = os.path.abspath(args.data_root)
    if args.dinov3_repo:
        config["dinov3_repo"] = os.path.abspath(args.dinov3_repo)
    if args.dinov3_weights:
        config["dinov3_weights"] = os.path.abspath(args.dinov3_weights)
    if args.max_frames_per_sequence is not None:
        config["max_frames_per_sequence"] = args.max_frames_per_sequence
    iou_threshold = config["detection"]["iou_threshold"] if args.iou_threshold is None else args.iou_threshold

    detector = None
    if args.method == "wtd":
        device = torch.device(args.device or config["device"])
        model = load_dinov3(config["dinov3_weights"], config["dinov3_repo"])
        detector = WeakTargetDetector(model, config, device)

        def detect(path):
            with Image.open(path) as handle:
                detections, _ = detector.detect(handle.convert("RGB"))
            return [detection["box"] for detection in detections]
    else:
        detect = yolo_detector(args.weights or config["final_weights"], config)

    results = []
    for name, sequence_root in list_sequences(config):
        images = list_images(sequence_root, config["image_dir"])
        if config["max_frames_per_sequence"]:
            images = images[:config["max_frames_per_sequence"]]
        result = evaluate_sequence(detect, images, config["annotation_dir"], iou_threshold)
        results.append((name, result))

    if detector is not None:
        detector.close()

    print(f"{'sequence':24s} {'recall':>8s} {'mean IoU':>9s} {'FPS':>8s} {'proposals':>10s}")
    for name, result in results:
        print(f"{name:24s} {result['recall'] * 100:>7.2f}% {result['mean_iou']:>9.4f} "
              f"{result['fps']:>8.1f} {result['avg_proposals']:>10.2f}")
    total = summarize(results)
    print(f"{'total':24s} {total['recall'] * 100:>7.2f}% {total['mean_iou']:>9.4f} "
          f"{total['fps']:>8.1f} {total['avg_proposals']:>10.2f}")

    if args.output:
        output = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump({
                "method": args.method,
                "iou_threshold": iou_threshold,
                "total": total,
                "per_sequence": [{"sequence": name, **result} for name, result in results],
            }, handle, indent=2)
        print(output)


if __name__ == "__main__":
    main()
