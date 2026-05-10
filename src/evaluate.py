import torch
import cv2
import numpy as np
from torchvision import transforms
from model import fire_smoke_classifier
from PIL import Image
import os
from pathlib import Path

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
classes = ['background', 'fire', 'smoke']

model = fire_smoke_classifier(3)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

conf_threshold = 0.999
nms_threshold = 0.05
iou_threshold = 0.15


def has_fire_or_smoke_colors(crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    total = crop_bgr.shape[0] * crop_bgr.shape[1]

    green_mask  = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    if green_mask.sum() / 255 / total > 0.30:
        return False

    fire_mask1  = cv2.inRange(hsv, (0,   100, 100), (35,  255, 255))
    fire_mask2  = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    fire_ratio  = cv2.bitwise_or(fire_mask1, fire_mask2).sum() / 255 / total
    smoke_ratio = cv2.inRange(hsv, (0, 0, 50), (180, 50, 200)).sum() / 255 / total

    return fire_ratio > 0.05 or smoke_ratio > 0.50


def load_ground_truth(label_path, img_w, img_h):
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, nw, nh = map(float, parts)
            x1 = int((xc - nw/2) * img_w)
            y1 = int((yc - nh/2) * img_h)
            x2 = int((xc + nw/2) * img_w)
            y2 = int((yc + nh/2) * img_h)
            gt_boxes.append({'box': [x1, y1, x2, y2], 'class': int(cls)})

    return gt_boxes


def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter  = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    areaA  = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB  = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union  = float(areaA + areaB - inter)
    return inter / union if union > 0 else 0


def detect_image(img_path, label_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return [], []
    h_img, w_img, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gt_data = load_ground_truth(str(label_path), w_img, h_img)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    crops, boxes = [], []
    for (x, y, w_box, h_box) in rects[:500]:
        if w_box < 100 or h_box < 100:
            continue
        if (w_box * h_box) > (w_img * h_img * 0.5):
            continue
        crop_bgr = img[y:y+h_box, x:x+w_box]
        if not has_fire_or_smoke_colors(crop_bgr):
            continue
        crops.append(transform(Image.fromarray(rgb[y:y+h_box, x:x+w_box])))
        boxes.append([x, y, w_box, h_box])

    if not crops:
        return gt_data, []

    all_confs, all_preds = [], []
    for i in range(0, len(crops), 64):
        batch = torch.stack(crops[i:i+64]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1)
            confs, preds = torch.max(probs, dim=1)
        all_confs.extend(confs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    detections = []
    for target_cls_id in [1, 2]:
        cls_boxes, cls_confs = [], []
        for i in range(len(all_preds)):
            if all_preds[i] == target_cls_id and all_confs[i] > conf_threshold:
                cls_boxes.append(boxes[i])
                cls_confs.append(float(all_confs[i]))
        if not cls_boxes:
            continue
        nms_result = cv2.dnn.NMSBoxes(cls_boxes, cls_confs, conf_threshold, nms_threshold)
        if len(nms_result) > 0:
            for idx in nms_result.flatten():
                detections.append({
                    'box': cls_boxes[idx],
                    'conf': cls_confs[idx],
                    'cls_id': target_cls_id
                })

    return gt_data, detections


def evaluate(data_split="test", max_images=100):
    img_dir   = Path(f"data/{data_split}/images")
    label_dir = Path(f"data/{data_split}/labels")

    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    img_files = img_files[:max_images]
    print(f"\nEvaluating {len(img_files)} images from {data_split}...\n")

    stats = {
        'fire':  {'tp': 0, 'fp': 0, 'fn': 0},
        'smoke': {'tp': 0, 'fp': 0, 'fn': 0},
    }

    for idx, img_path in enumerate(img_files):
        label_path = label_dir / img_path.name.replace(img_path.suffix, ".txt")
        gt_data, detections = detect_image(img_path, label_path)

        # GT-s per class
        gt_fire  = [g for g in gt_data if g['class'] == 0]
        gt_smoke = [g for g in gt_data if g['class'] == 1]

        # Detections per class
        det_fire  = [d for d in detections if d['cls_id'] == 1]
        det_smoke = [d for d in detections if d['cls_id'] == 2]

        for cls_name, gt_list, det_list in [('fire',  gt_fire,  det_fire),('smoke', gt_smoke, det_smoke)]:
            matched_gt = set()
            for det in det_list:
                x, y, w, h = det['box']
                pred_box = [x, y, x+w, y+h]
                best_iou, best_gt_idx = 0, -1

                for gt_idx, gt in enumerate(gt_list):
                    if gt_idx in matched_gt:
                        continue
                    iou = calculate_iou(pred_box, gt['box'])
                    if iou > best_iou:
                        best_iou    = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    stats[cls_name]['tp'] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    stats[cls_name]['fp'] += 1

            stats[cls_name]['fn'] += len(gt_list) - len(matched_gt)

        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(img_files)} processed...")

    #Results 
    print("\n" + "="*45)
    print(f"{'':10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("="*45)

    for cls_name in ['fire', 'smoke']:
        tp = stats[cls_name]['tp']
        fp = stats[cls_name]['fp']
        fn = stats[cls_name]['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0)

        print(f"{cls_name:10} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}")
        print(f"{'':10} TP={tp} FP={fp} FN={fn}")

    print("="*45)

    # Overall
    total_tp = sum(stats[c]['tp'] for c in stats)
    total_fp = sum(stats[c]['fp'] for c in stats)
    total_fn = sum(stats[c]['fn'] for c in stats)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0)
    print(f"{'OVERALL':10} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}")
    print("="*45)


if __name__ == "__main__":
    evaluate(data_split="test", max_images=100)