import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import load_config
from models.SELD import build_dataset, compute_final_scores, select_top


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2: score the candidate pool and build a YOLO pseudo-label dataset."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--w-t", type=float, default=None)
    parser.add_argument("--top-ratio", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    candidates_path = args.candidates or config["candidates"]
    output_dir = args.output or config["pseudo_dataset"]
    w_t = config["scoring"]["w_t"] if args.w_t is None else args.w_t
    top_ratio = config["scoring"]["top_ratio"] if args.top_ratio is None else args.top_ratio

    with open(candidates_path, encoding="utf-8") as handle:
        candidates = json.load(handle)
    scored = compute_final_scores(candidates, w_t)
    selected = select_top(scored, top_ratio)
    summary = build_dataset(selected, output_dir)

    print(f"candidates: {len(scored)}")
    print(f"selected: {len(selected)}")
    print(f"images: {summary['images']}")
    print(f"boxes: {summary['boxes']}")
    print(summary["dataset"])


if __name__ == "__main__":
    main()
