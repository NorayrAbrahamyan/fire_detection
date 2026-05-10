import cv2
import os
import random
import shutil
from pathlib import Path


def get_gt_boxes(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            x_c, y_c, nw, nh = map(float, parts[1:])
            x1 = int((x_c - nw / 2) * img_w)
            y1 = int((y_c - nh / 2) * img_h)
            x2 = int((x_c + nw / 2) * img_w)
            y2 = int((y_c + nh / 2) * img_h)
            boxes.append({'box': [x1, y1, x2, y2], 'class': cls_id})

    return boxes


def get_fire_smoke_crops(data_split, limit_per_class):
    img_dir = Path(f"data/{data_split}/images")
    label_dir = Path(f"data/{data_split}/labels")
    out_dir = Path(f"classifier_data/{data_split}")

    counts = {'fire': 0, 'smoke': 0}

    label_files = list(label_dir.glob("*.txt"))
    random.shuffle(label_files)

    for label_file in label_files:
        if all(counts[c] >= limit_per_class for c in counts):
            break

        img_path = img_dir / label_file.name.replace(".txt", ".jpg")
        if not img_path.exists():
            img_path = img_dir / label_file.name.replace(".txt", ".png")
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w, _ = img.shape

        gt_boxes = get_gt_boxes(label_file, w, h)

        for i, gt in enumerate(gt_boxes):
            cls_name = 'fire' if gt['class'] == 0 else 'smoke'
            if counts[cls_name] >= limit_per_class:
                continue

            x1, y1, x2, y2 = gt['box']
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            save_path = out_dir / cls_name / f"{label_file.stem}_{i}.jpg"
            cv2.imwrite(str(save_path), crop)
            counts[cls_name] += 1

    return counts


def get_background_crops_from_coco(coco_dir, data_split, limit):
    out_dir = Path(f"classifier_data/{data_split}/background")
    coco_images = list(Path(coco_dir).glob("*.jpg"))

    random.shuffle(coco_images)

    count = 0
    for img_path in coco_images:
        if count >= limit:
            break

        img = cv2.imread(str(img_path)) 
        if img is None:
            continue
        h, w, _ = img.shape

        if h < 100 or w < 100:
            continue

        for j in range(2):
            if count >= limit:
                break

            crop_h = random.randint(int(h * 0.3), int(h * 0.7))
            crop_w = random.randint(int(w * 0.3), int(w * 0.7))
            y1 = random.randint(0, h - crop_h)
            x1 = random.randint(0, w - crop_w)

            crop = img[y1:y1 + crop_h, x1:x1 + crop_w]
            if crop.size == 0:
                continue

            save_path = out_dir / f"coco_{img_path.stem}_{j}.jpg"
            cv2.imwrite(str(save_path), crop)
            count += 1

    return count

def prepare(data_split, fire_smoke_limit, bg_limit, coco_dir):
    print(f"\n{data_split.upper()}")

    for cls in ['fire', 'smoke', 'background']:
        os.makedirs(f"classifier_data/{data_split}/{cls}", exist_ok=True)

    #Fire/Smoke from YOLO
    counts = get_fire_smoke_crops(data_split, fire_smoke_limit)
    print(f"  Fire:  {counts['fire']}")
    print(f"  Smoke: {counts['smoke']}")

    #Background from COCO
    print("Background crops COCO-ից...")
    bg_count = get_background_crops_from_coco(coco_dir, data_split, bg_limit)
    print(f"  Background: {bg_count}")


if __name__ == "__main__":
    coco_dir = "val2017"

    if os.path.exists("classifier_data"):
        shutil.rmtree("classifier_data")
        print("Old classifier_data cleaned")

    prepare(data_split="train", fire_smoke_limit=5000, bg_limit=5000, coco_dir=coco_dir)
    prepare(data_split="valid", fire_smoke_limit=1000, bg_limit=1000, coco_dir=coco_dir)