import os
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

# VOC root
voc_root = Path("datasets/VOCdevkit")
sets = [('VOC2007', 'trainval'), ('VOC2007', 'val'), ('VOC2012', 'trainval')]

classes = [
    'aeroplane','bicycle','bird','boat','bottle',
    'bus','car','cat','chair','cow','diningtable',
    'dog','horse','motorbike','person','pottedplant',
    'sheep','sofa','train','tvmonitor'
]

save_dir = Path("datasets/VOC")
(save_dir / "images").mkdir(parents=True, exist_ok=True)
(save_dir / "labels").mkdir(parents=True, exist_ok=True)

train_list, val_list = [], []

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

for year, image_set in sets:
    image_dir = voc_root / year / "JPEGImages"
    label_dir = voc_root / year / "Annotations"
    list_file = voc_root / year / "ImageSets/Main" / f"{image_set}.txt"
    with open(list_file) as f:
        image_ids = f.read().strip().split()

    for image_id in tqdm(image_ids, desc=f"Processing {year} {image_set}"):
        xml_file = label_dir / f"{image_id}.xml"
        img_file = image_dir / f"{image_id}.jpg"
        out_img = save_dir / "images" / f"{image_id}.jpg"
        out_label = save_dir / "labels" / f"{image_id}.txt"

        os.system(f"cp {img_file} {out_img}")

        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        with open(out_label, 'w') as out:
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in classes: 
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert((w, h), b)
                out.write(" ".join([str(cls_id)] + [f"{a:.6f}" for a in bb]) + '\n')

        # split set
        if image_set == "val":
            val_list.append(f"datasets/VOC/images/{image_id}.jpg")
        else:
            train_list.append(f"datasets/VOC/images/{image_id}.jpg")


# Write train/val list files
with open(save_dir / "train.txt", 'w') as f:
    f.write("\n".join(train_list))
with open(save_dir / "val.txt", 'w') as f:
    f.write("\n".join(val_list))

print("VOC → YOLO conversion finished successfully!")
