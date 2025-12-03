🦾 YOLOv5-Lite-Ghost-VOC — Lightweight High-Accuracy Object Detection

🚀 轻量级改进版 YOLOv5-Lite（Ghost + VOC 20 类目标检测）
本项目面向 移动端 / 边缘端部署 + 工程实用场景，在保持极低模型规模的前提下显著提升检测精度。

🔥 Highlights（项目亮点）
改进点	效果
将 YOLOv5-Lite Backbone 替换为 Ghost-style Backbone	参数量↓ 推理速度↑
分阶段训练（无冻结 → 冻结 Backbone 微调）	小数据训练效果更稳
增强策略优化（Mosaic + MixUp）	缓解复杂背景/猫狗/鸟/马漏检
输入尺寸升级（416 → 512）	小目标检测能力提升
手动可视化错误案例 & 模型诊断	定位类别偏差、目标密集场景弱点

📌 Final Performance
Model Version	IMG Size	Params	Best mAP@0.5	Best mAP@0.5:.95
原版 YOLOv5-Lite-s (VOC)	416	—	0.156	—
Ghost-Backbone + 512 + 数据增强	512	1.4M	0.344	—
+ 冻结 Backbone 精调（最终版本）	512	1.4M	0.394 ➜ 0.40 🎯	0.191

📌 ≈ +156% 提升 相比初始阶段。

🖼️ Inference Demo（检测效果）

<img src="demo_images/000026.jpg" width="200">	<img src="demo_images/000113.jpg" width="200">	<img src="demo_images/000117.jpg" width="200">	<img src="demo_images/000150.jpg" width="200">
<img src="demo_images/000225.jpg" width="200">	<img src="demo_images/000236.jpg" width="200">	<img src="demo_images/000486.jpg" width="200">	<img src="demo_images/000842.jpg" width="200">

🔍 模型对猫/狗/鸟/马等复杂背景目标有明显提升。

🚀 Quick Start
1️⃣ 安装依赖
pip install -r requirements.txt

2️⃣ 推理（使用最佳模型 best.pt）
python detect.py --weights best.pt --source demo_images

3️⃣ 训练（重现最终版本）
python train.py \
  --cfg models/v5Lite-ghost-s.yaml \
  --weights '' \
  --data data/voc.yaml \
  --img-size 512 \
  --batch-size 2 \
  --epochs 25 \
  --hyp data/hyp.scratch-low.yaml \
  --workers 2

📂 Project Structure
YOLOv5-Lite-Ghost-VOC
│── models/                   # Ghost backbone + Detect head
│── data/
│   ├── voc.yaml              # VOC dataset config (20 classes)
│── weights/
│   ├── best.pt               # 最终最佳权重（推荐推理）
│── demo_images/              # 推理示例图片
│── train.py / detect.py      # 训练&推理入口
│── README.md                 # 本文档

🔍 Error Case Diagnosis（模型诊断 & 分析）

背景复杂场景 → 提升显著

密集目标（同类高密度） → 有待提高

鸟、马、猫、狗在高速/遮挡状态下仍存在漏检

📌 若继续优化，可尝试：

EMA 迁移

RepConv / C2f-Ghost

Soft-NMS / DIoU-NMS 替换

可变形卷积 DCNv3

🏗️ Deployment（可选）
端	支持情况
TensorRT FP16	✔（推荐）
ONNX Runtime	✔
OpenVINO	✔
Mobile (NCNN)	预计适配良好（Ghost结构轻量）

模型导出示例：

python export.py --weights best.pt --include onnx

👨‍💻 Author & Contact
信息	内容
作者	Zhuz0123
邮箱	953153859@qq.com
仓库地址
⭐ https://github.com/Zhuz0123/yolov5lite-ghost-voc
欢迎 Star 🌟 ，有任何问题欢迎 Issue 技术交流！