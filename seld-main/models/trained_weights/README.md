# Weights

Place the files below in this directory and keep the file names unchanged; the
paths in `configs/default.yaml` point here by default. Entries listed with a
download source are not redistributed and must be obtained from that source.

## 1. Backbone (download)

| File | Role | Source |
|---|---|---|
| `dinov3-vitl16-pretrain-lvd1689m/model.safetensors` | DINOv3 ViT-L/16 backbone used in Stage 1 | `facebookresearch/dinov3`, LVD-1689M pretrained weights |
| `dinov3-vitl16-pretrain-lvd1689m/config.json` | backbone configuration | same release |
| `dinov3-vitl16-pretrain-lvd1689m/preprocessor_config.json` | preprocessing metadata | same release |

The loader reads `model.safetensors`, converts the Hugging Face parameter names
to the native DINOv3 module names, and loads the result into
`dinov3.models.vision_transformer.vit_large`. The DINOv3 **code base** is a
separate clone: set `dinov3_repo` in `configs/default.yaml` to its root.

## 2. Detector initialisation (download)

| File | Role | Source |
|---|---|---|
| `yolo11s.pt` | COCO-pretrained YOLOv11-s, starting point of Stage 3 | Ultralytics model zoo |

## 3. Produced by this repository

| File | Role | Produced by |
|---|---|---|
| `seld_yolo11s.pt` | Stage 3 detector after head-only and full-network training | `scripts/train_detector.sh` |

The training script copies the best checkpoint of the second stage to
`final_weights` (`./models/trained_weights/seld_yolo11s.pt`).

## 4. Intermediate artefacts (generated, not versioned)

| Artefact | Path |
|---|---|
| candidate pool | `./outputs/all_candidates.json` |
| pseudo-label dataset (`images/`, `labels/`, `jsons/`, `data.yaml`) | `./outputs/pseudouavdata` |
| YOLO training runs | `./outputs/yolo` |

These paths are the defaults of `configs/default.yaml` and are ignored by git.
