# Datasets

No frames or annotations are distributed with this repository. SELD is trained
and evaluated on **MMFW-UAV** fixed-wing infrared sequences; any sequence set
that follows the layout below can be used instead.

## Layout

```
<data_root>/
├── Fixed-wing-UAV-A/
│   └── Bottom_Up/
│       ├── Infrared_Imgs/
│       │   ├── 0_1_000001.jpg
│       │   └── ...
│       └── Infrared_Anns/
│           ├── 0_1_000001.xml
│           └── ...
├── Fixed-wing-UAV-A'/
├── Fixed-wing-UAV-B/
├── ...
└── Fixed-wing-UAV-F'/
```

The reference set contains twelve sequences (`Fixed-wing-UAV-A` through
`Fixed-wing-UAV-F`, each with a primed variant). Frames are 1280×1024 JPEG
images; the annotation file shares the frame stem and lives in the sibling
`Infrared_Anns` directory. Every sequence provides three view splits —
`Bottom_Up`, `Horizontal` and `Top_Down`; `split_dir` selects the one used by the
pipeline.

`configs/default.yaml` names the four anchors:

```yaml
data_root: ./datasets/MMFW-UAV-DATASET
split_dir: Bottom_Up
image_dir: Infrared_Imgs
annotation_dir: Infrared_Anns
sequences: []
```

Sequence discovery is sorted, and frames inside a sequence are read in file-name
order, which is the temporal order used by trajectory confirmation.

## Annotation format

Each annotation holds one object:

```xml
<annotation>
  <object>
    <bndbox>
      <xmin>742</xmin>
      <ymin>318</ymin>
      <xmax>751</xmax>
      <ymax>327</ymax>
    </bndbox>
  </object>
</annotation>
```

Frames without an annotation file are excluded from the ground-truth statistics;
all proposals on them are ignored during evaluation.

## Subsets used by the pipeline

* **Stage 1 / Stage 2** use the infrared frames only. No annotation is read while
  candidates are mined or scored.
* **Stage 3** trains and monitors on the exported pseudo-label set `D_p`, because
  no labelled training split exists in the unlabelled setting.
* **Evaluation** reads the annotations to compute GT-level recall, mean IoU and
  the proposal statistics; it never feeds them back into the pipeline.

## Datasets used in the paper

| Dataset | Role in this work | Source | Redistributed |
|---|---|---|---|
| MMFW-UAV | training and evaluation | Sci Data 12, 185 (2025) | No |
| Anti-UAV410 | cross-dataset evaluation | IEEE TPAMI 46(5), 2023 | No |
| CST Anti-UAV | cross-dataset evaluation | ICCVW 2025 | No |
| IRDST | cross-dataset evaluation | IEEE TGRS 61, 2023 | No |

### MMFW-UAV

The sequences used for training and evaluation are taken from the publicly
released MMFW-UAV dataset:

> Liu, Y., Sun, Z., Xi, L. et al. MMFW-UAV dataset: multi-sensor and multi-view
> fixed-wing UAV dataset for air-to-air vision tasks. *Scientific Data* 12, 185
> (2025). https://doi.org/10.1038/s41597-025-04482-2

It is distributed under the Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0 International licence
(CC BY-NC-ND 4.0). Download it from the original release and place it under
`data_root` using the layout above.

### Anti-UAV410

> Huang, B., Li, J., Chen, J., Wang, G., Zhao, J. and Xu, T. Anti-UAV410: a
> thermal infrared benchmark and customized scheme for tracking drones in the
> wild. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 46(5),
> 2852-2865, 2023.

Used for cross-dataset evaluation. Not redistributed here.

### CST Anti-UAV

> Xie, B., Zhang, C., Wang, F., Liu, P., Lu, F., Chen, Z. and Hu, W. CST
> Anti-UAV: a thermal infrared benchmark for tiny UAV tracking in complex
> scenes. *IEEE/CVF International Conference on Computer Vision Workshops
> (ICCVW)*, 6216-6225, 2025.

Used for cross-dataset evaluation. Not redistributed here.

### IRDST

> Sun, H., Bai, J., Yang, F. and Bai, X. Receptive-field and direction induced
> attention network for infrared dim small target detection with a large-scale
> dataset IRDST. *IEEE Transactions on Geoscience and Remote Sensing* 61, 1-13,
> 2023.

Used for cross-dataset evaluation. Not redistributed here.

### Licence of third-party datasets

The original releases of Anti-UAV410, CST Anti-UAV and IRDST do not state an
explicit licence in their publications. They are used here for non-commercial
research only and are not redistributed. Users should obtain them from the
original authors and comply with their terms.

The repository and the checkpoints released with it are distributed under
CC BY-NC-ND 4.0 (see [`../LICENSE`](../LICENSE)).
