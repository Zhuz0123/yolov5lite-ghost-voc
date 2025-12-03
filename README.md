🚀 YOLOv5-Lite + Ghost Backbone for VOC (20-Class Lightweight Object Detection)

A lightweight YOLOv5-Lite improved with Ghost-style backbone & enhanced training strategies — optimized for speed-accuracy trade-off on VOC dataset.

中文简介：本项目基于 YOLOv5-Lite，替换原 CSPDarkNet-Lite 主干为 Ghost-style 轻量网络，并结合输入尺寸提升与增强策略优化，实现更优的轻量化检测效果。

🔥 Highlights — What was improved
模块	改进策略	效果
Backbone	替换为 Ghost-style 主干（GhostConv + GhostBottleneck）	减小参数量 & 提升速度
数据增强	Mosaic + Mixup 策略微调	缓解过拟合 & 提升鲁棒性
输入尺寸	416 → 512	提升小目标检测能力
实验方案	ExpA（不冻结） vs ExpB（冻结前 10 层）	ExpB 取得最佳效果
推理速度	轻量结构保持高 FPS	适用边缘端部署
📌 Final Performance Summary
方法	Resolution	Backbone	Freeze	mAP@0.5	备注
Baseline YOLOv5-Lite	416	原版	✗	0.32	原始效果
ExpA	512	Ghost	✗	0.344	输入尺寸提升
🚀 ExpB (Best)	512	Ghost	✓（前 10 层）	0.394 → 0.400	最优权重 best.pt

ExpB 相比 ExpA 净提升约 +5.6% mAP@0.5。

📸 Detection Demo (8 Images)

结果展示由 ExpB 的 best.pt 推理得到

<div align="center"> <img src="demo_images/000026.jpg" width="32%" /> <img src="demo_images/000113.jpg" width="32%" /> <img src="demo_images/000117.jpg" width="32%" /> <img src="demo_images/000150.jpg" width="32%" /> <img src="demo_images/000225.jpg" width="32%" /> <img src="demo_images/000236.jpg" width="32%" /> <img src="demo_images/000486.jpg" width="32%" /> <img src="demo_images/000842.jpg" width="32%" /> </div>
🔧 Environment
环境	版本建议
Python	3.8 – 3.10
PyTorch	≥ 1.11 (建议 2.0+)
CUDA	11.x / 12.x
GPU	≥ 4 GB 显存可训练，2 GB 可推理（如 MX450）

安装依赖：

pip install -r requirements.txt

🧠 Inference (推理)
python detect.py \
  --weights runs/train/exp18/weights/best.pt \
  --source demo_images \
  --img 512 \
  --conf 0.25


📌 推荐：

results → runs/detect/exp/

🏋️‍♂️ Training (训练复现)
python train.py \
  --cfg models/v5Lite-ghost-s.yaml \
  --weights '' \
  --data data/voc.yaml \
  --img-size 512 \
  --batch-size 2 \
  --epochs 25 \
  --hyp data/hyp.scratch-low.yaml \
  --workers 2

继续训练 / 微调
python train.py \
  --cfg models/v5Lite-ghost-s.yaml \
  --weights runs/train/exp18/weights/best.pt \
  --data data/voc.yaml \
  --img-size 512

📦 Pretrained Weights
文件	说明
best.pt	ExpB 最优模型（推荐部署）
last.pt	最后一轮 checkpoint

📌 权重下载链接
https://github.com/Zhuz0123/yolov5lite-ghost-voc/runs/train/exp18/weights/best.pt

🧱 Project Structure
YOLOv5-Lite
├─ models
│  ├─ v5Lite-ghost-s.yaml       # 改进后的 Ghost 主干
├─ data/voc.yaml                # VOC 数据配置
├─ runs/train                   # 训练日志与权重
├─ demo_images                  # 示例推理图片（8 张）
└─ detect.py / train.py         # 推理 / 训练脚本
📜 License

本项目遵循 GPL-3.0 协议，用于研究与非商业用途。

📧 Contact

如有交流合作意向欢迎联系：

Author: Zhuz0123  
Email: 953153859@qq.com
