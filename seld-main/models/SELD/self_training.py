import argparse
import os
import shutil

from models import load_config


def train_stage(weights, data, config, epochs, freeze, project, name):
    from ultralytics import YOLO

    yolo = config["yolo"]
    model = YOLO(weights)
    model.train(
        data=data,
        epochs=epochs,
        imgsz=yolo["imgsz"],
        batch=yolo["batch"],
        device=yolo["device"],
        freeze=freeze,
        patience=yolo["patience"],
        project=project,
        name=name,
        exist_ok=True,
        seed=config["seed"],
        amp=yolo["amp"],
        cos_lr=yolo["cos_lr"],
        close_mosaic=yolo["close_mosaic"],
        iou=yolo["nms_iou"],
        mosaic=yolo["mosaic"],
        mixup=yolo["mixup"],
        copy_paste=yolo["copy_paste"],
        scale=yolo["scale"],
        hsv_h=yolo["hsv_h"],
        hsv_s=yolo["hsv_s"],
        hsv_v=yolo["hsv_v"],
        degrees=yolo["degrees"],
        translate=yolo["translate"],
        fliplr=yolo["fliplr"],
        flipud=yolo["flipud"],
    )
    return os.path.join(project, name, "weights", "best.pt")


def run_two_stage(data, init_weights, config, project):
    head = train_stage(init_weights, data, config, config["yolo"]["epochs_stage1"],
                       config["yolo"]["freeze"], project, "stage1_head")
    return train_stage(head, data, config, config["yolo"]["epochs_stage2"], 0, project, "stage2_full")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: two-stage YOLO self-training on a fixed pseudo-label set.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--epochs-stage1", type=int, default=None)
    parser.add_argument("--epochs-stage2", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config["yolo"]["epochs_stage1"] = args.epochs_stage1 or config["yolo"]["epochs_stage1"]
    config["yolo"]["epochs_stage2"] = args.epochs_stage2 or config["yolo"]["epochs_stage2"]
    data = args.data or os.path.join(config["pseudo_dataset"], "data.yaml")
    weights = args.weights or config["yolo_weights"]
    project = args.project or config["yolo_project"]
    best = run_two_stage(data, weights, config, project)
    shutil.copy2(best, config["final_weights"])
    print(best)
    print(config["final_weights"])


if __name__ == "__main__":
    main()
