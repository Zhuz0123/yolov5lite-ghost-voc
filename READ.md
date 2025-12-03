# YOLOv5-Lite with GhostNet Backbone on VOC2007+2012

> Customized lightweight object detection on PASCAL VOC, with Ghost-style backbone, multi-stage training and augmentation tuning.

---

## 1. Project Overview

This repository is **my customized version** of YOLOv5-Lite, where I:

- Re-designed the **backbone** to use a Ghost-style lightweight architecture.
- Trained and tuned the model on **PASCAL VOC 2007+2012 (20 classes)**.
- Performed **two-stage experiments (ExpA / ExpB)** to improve accuracy under laptop GPU constraints (MX450).
- Achieved a final performance of:

> **mAP@0.5 ≈ 0.394, mAP@0.5:0.95 ≈ 0.191 on VOC val**

This project is intended both as a **research-style reproduction & improvement** and as a **portfolio project** demonstrating model modification, training, and experimental design.

---

## 2. Dataset

- **Dataset**: PASCAL VOC 2007 + VOC 2012
- **Classes**: 20
- **Preparation** (typical structure):

```text
YOLOv5-Lite/
└── datasets/
    └── VOC/
        ├── images/
        │   ├── train/   # merged VOC2007+2012 trainval
        │   └── val/     # VOC2007 test
        └── labels/
            ├── train/
            └── val/

    Dataset config: data/voc.yaml (20 classes, train/val paths pointing to the folders above)

3. Model Architecture: Ghost-style YOLOv5-Lite

Config file: models/v5Lite-ghost-s.yaml

Key points:

    Backbone:

        Replace standard Conv backbone with GhostConv + C3 to reduce FLOPs and parameters.

        Output feature maps at three scales: P3 / P4 / P5.

    Head:

        PANet-style FPN head with upsampling + Concat + C3.

        Detection head on P3, P4, P5 with anchors adapted to VOC.

Example (excerpt):

# v5Lite-ghost-s.yaml

nc: 20
anchors:
  - [10,13, 16,30, 33,23]
  - [30,61, 62,45, 59,119]
  - [116,90, 156,198, 373,326]

backbone:
  - [-1, 1, Conv, [32, 3, 2]]          # 0
  - [-1, 1, Conv, [64, 3, 2]]          # 1
  - [-1, 1, GhostConv, [64, 3, 2]]     # 2
  - [-1, 1, C3, [64, True]]            # 3
  - [-1, 1, GhostConv, [128, 3, 2]]    # 4
  - [-1, 1, C3, [128, True]]           # 5
  - [-1, 1, GhostConv, [256, 3, 2]]    # 6
  - [-1, 1, C3, [256, True]]           # 7

head:
  - [7, 1, Conv, [128, 1, 1]]          # 8
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 9
  - [[-1, 5], 1, Concat, [1]]          # 10
  - [-1, 1, C3, [128, False]]          # 11
  - [-1, 1, Conv, [64, 1, 1]]          # 12
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 13
  - [[-1, 3], 1, Concat, [1]]          # 14
  - [-1, 1, C3, [64, False]]           # 15
  - [-1, 1, Conv, [64, 3, 2]]          # 16
  - [[-1, 11], 1, Concat, [1]]         # 17
  - [-1, 1, C3, [128, False]]          # 18
  - [-1, 1, Conv, [128, 3, 2]]         # 19
  - [[-1, 7], 1, Concat, [1]]          # 20
  - [-1, 1, C3, [256, False]]          # 21
  - [[15, 18, 21], 1, Detect, [nc, anchors]]

    Note: In the final experiments (ExpA, ExpB), freeze: is commented out in the YAML to keep the config clean; freezing is controlled from the training script / code when needed.

4. Training Setup

Environment (GPU-limited):

    GPU: NVIDIA GeForce MX450 (2GB)

    Framework: PyTorch 2.4.1

    Python: 3.8 (Conda env yolo5lite)

    Batch size: 2 (due to VRAM)

    Optimizer: SGD with momentum (momentum=0.937, weight_decay=5e-4)

    Mixed precision: torch.cuda.amp (on most runs)

Hyperparameters (file: data/hyp.scratch-low.yaml, edited):

lr0: 0.001
lrf: 0.2
momentum: 0.937
weight_decay: 0.0005

box: 0.05
cls: 0.5
obj: 1.0
iou_t: 0.2
anchor_t: 4.0
cls_pw: 1.0
obj_pw: 1.0
fl_gamma: 0.0

hsv_h: 0.015
hsv_s: 0.6
hsv_v: 0.4
degrees: 0.0
translate: 0.08
scale: 0.4
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5

# tuned in later experiments:
mosaic: 0.20
mixup: 0.05

5. Experiments
5.1 Experimental Design

I designed two main improvement stages:

    ExpA – Higher resolution finetuning

        Start from the best 416×416 model.

        Increase input size to 512×512.

        Keep all layers trainable.

        Objective: exploit higher resolution while staying within MX450 memory limits.

    ExpB – Augmentation & partial backbone freezing

        Start from the best ExpA weights.

        Slightly tune data augmentation (mosaic=0.20, mixup=0.05).

        Freeze most of the backbone layers during finetuning to stabilise features and reduce overfitting.

        Objective: improve generalization and convergence stability.

5.2 Training Commands
Initial Ghost backbone training @ 416 (baseline for this project)

python train.py \
  --cfg models/v5Lite-ghost-s.yaml \
  --weights '' \
  --data data/voc.yaml \
  --img-size 416 \
  --batch-size 2 \
  --epochs 40 \
  --hyp data/hyp.scratch-low.yaml \
  --workers 2

Best model saved at e.g.:

runs/train/exp15/weights/best.pt

ExpA – 512×512 finetuning from 416 best

python train.py \
  --cfg models/v5Lite-ghost-s.yaml \
  --weights runs/train/exp15/weights/best.pt \
  --data data/voc.yaml \
  --img-size 512 \
  --batch-size 2 \
  --epochs 25 \
  --hyp data/hyp.scratch-low.yaml \
  --workers 2

Produces runs/train/exp17/weights/best.pt (your exact run id may differ).
ExpB – Augmentation + partial freezing (from ExpA best)

In code, I freeze most backbone layers (e.g. early Conv/GhostConv/C3 blocks) and train only head + some deeper layers. Command example:

python train.py \
  --cfg models/v5Lite-ghost-s.yaml \
  --weights runs/train/exp17/weights/best.pt \
  --data data/voc.yaml \
  --img-size 512 \
  --batch-size 2 \
  --epochs 20 \
  --hyp data/hyp.scratch-low.yaml \
  --workers 2

Best model saved as:

runs/train/exp18/weights/best.pt

This is the final recommended model for inference.
6. Results
6.1 Overall mAP Comparison (VOC val)
Stage	Input Size	Init Weights	Strategy	mAP@0.5	mAP@0.5:0.95
1	416×416	Scratch (Ghost backbone)	Baseline training	~0.268	~0.124
2	512×512	From Stage 1 best (ExpA)	Higher-res finetuning	~0.344	~0.164
3	512×512	From ExpA best (ExpB)	Aug tuned + partial backbone freezing	~0.394	~0.191

    All metrics are taken from training logs on VOC val (2510 images, 7818 labels).

6.2 Per-class performance (ExpB best snapshot)

Example (from the last logged epoch of ExpB):

    Strong classes: person, car, bus, dog, cat, horse, etc.

    Hard classes: small/occluded objects or heavy background (e.g. bird, bottle, pottedplant).

(Full per-class AP is available in the training log.)
7. Inference
7.1 Command-line detection

Use the best ExpB checkpoint (update the path if your run index differs):

python detect.py \
  --weights runs/train/exp18/weights/best.pt \
  --source datasets/VOC/images/val \
  --img-size 512 \
  --conf-thres 0.25 \
  --iou-thres 0.45 \
  --save-txt \
  --save-conf \
  --project runs/detect \
  --name voc_expB \
  --exist-ok

    Input folder can be:

        datasets/VOC/images/val (VOC val images), or

        A custom folder of your own test images.

    Outputs:

        Annotated images in runs/detect/voc_expB/

        YOLO txt predictions in runs/detect/voc_expB/labels/ (if --save-txt is used)

7.2 Example Usage

# Single image
python detect.py \
  --weights runs/train/exp18/weights/best.pt \
  --source path/to/your/image.jpg \
  --img-size 512 \
  --conf-thres 0.25

8. Project Highlights (for CV)

    Implemented a Ghost-style lightweight backbone inside YOLOv5-Lite from scratch by directly editing the model YAML.

    Built a full VOC2007+2012 training pipeline on Ubuntu, including data organization, label conversion, and config tuning.

    Designed and executed multi-stage experiments (baseline → ExpA → ExpB), achieving stepwise improvements under tight GPU memory limits:

        mAP@0.5: 0.268 → 0.344 → 0.394

    Investigated failure cases on cats/dogs/birds/horses with complex backgrounds and iterated on augmentation and training strategy.

    Packaged the project with command-line training/inference and a documented README for reproducibility.

9. How to Reproduce

# 1. Clone this repo
git clone https://github.com/Zhuz0123/yolov5lite-ghost-voc.git
cd yolov5lite-ghost-voc

# 2. (Optional) create conda env
conda create -n yolo5lite python=3.8 -y
conda activate yolo5lite

# 3. Install dependencies
pip install -r requirements.txt

# 4. Prepare VOC dataset under datasets/VOC
#    (follow the folder structure in Section 2)

# 5. Train (you can start from scratch or from my best weights)
python train.py --cfg models/v5Lite-ghost-s.yaml --data data/voc.yaml ...

# 6. Run inference
python detect.py --weights runs/train/exp18/weights/best.pt --source datasets/VOC/images/val
