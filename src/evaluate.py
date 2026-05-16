import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import shutil

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

CONF_THRESHOLD_FIRE = 0.95 
CONF_THRESHOLD_SMOKE = 0.97
NMS_THRESHOLD = 0.01        
IOU_EVAL_THRESHOLD = 0.15    

FP_SAVE_DIR = Path("detected_errors/false_positives")
if FP_SAVE_DIR.exists(): 
    shutil.rmtree(FP_SAVE_DIR)
FP_SAVE_DIR.mkdir(parents=True, exist_ok=True)

def get_model(num_classes):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5), 
        nn.Linear(num_ftrs, num_classes)
    )
    return model

model = get_model(3) 
model_path = "models/best_model.pth"

if os.path.exists(model_path):
    print(f"✅ Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
else:
    print("❌ ERROR: Model file not found!")

model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_ground_truth(label_path, img_w, img_h):
    gt_boxes = []
    if not os.path.exists(label_path): return gt_boxes
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5: continue
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
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union = float(areaA + areaB - inter)
    return inter / union if union > 0 else 0

def detect_image(img_path, label_path):
    img = cv2.imread(str(img_path))
    if img is None: return [], []
    h_img, w_img, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gt_data = load_ground_truth(str(label_path), w_img, h_img)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    crops, boxes = [], []
    for (x, y, w_box, h_box) in rects[:800]:
        if w_box < 40 or h_box < 40: continue
        crops.append(transform(Image.fromarray(rgb[y:y+h_box, x:x+w_box])))
        boxes.append([x, y, w_box, h_box])

    if not crops: return gt_data, []

    all_confs, all_preds = [], []
    batch_size = 64
    for i in range(0, len(crops), batch_size):
        batch = torch.stack(crops[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)
        all_confs.extend(confs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    detections = []
    for target_cls_id in [1, 2]: 
        current_thresh = CONF_THRESHOLD_FIRE if target_cls_id == 1 else CONF_THRESHOLD_SMOKE
        cls_boxes, cls_confs = [], []
        for i in range(len(all_preds)):
            if all_preds[i] == target_cls_id and all_confs[i] > current_thresh:
                cls_boxes.append(boxes[i])
                cls_confs.append(float(all_confs[i]))
        
        if not cls_boxes: continue
        
        nms_result = cv2.dnn.NMSBoxes(cls_boxes, cls_confs, current_thresh, NMS_THRESHOLD)
        if len(nms_result) > 0:
            for idx in nms_result.flatten():
                detections.append({'box': cls_boxes[idx], 'conf': cls_confs[idx], 'cls_id': target_cls_id - 1})
    
    return gt_data, detections

def evaluate(data_split="test", max_images=40):
    img_dir = Path(f"data/{data_split}/images")
    label_dir = Path(f"data/{data_split}/labels")
    img_files = (list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))[:max_images]

    print(f"\n🔍 Evaluating {len(img_files)} images from {data_split} set...")
    stats = {'fire': {'tp': 0, 'fp': 0, 'fn': 0}, 'smoke': {'tp': 0, 'fp': 0, 'fn': 0}}

    for idx, img_path in enumerate(img_files):
        label_path = label_dir / img_path.name.replace(img_path.suffix, ".txt")
        gt_data, detections = detect_image(img_path, label_path)
        raw_img = cv2.imread(str(img_path))

        for cls_id, cls_name in [(0, 'fire'), (1, 'smoke')]:
            cls_gt = [g for g in gt_data if g['class'] == cls_id]
            cls_det = [d for d in detections if d['cls_id'] == cls_id]

            matched_gt = set()
            for d_idx, det in enumerate(cls_det):
                x, y, w, h = det['box']
                pred_box = [x, y, x+w, y+h]
                best_iou, best_gt_idx = 0, -1

                for g_idx, gt in enumerate(cls_gt):
                    if g_idx in matched_gt: continue
                    iou = calculate_iou(pred_box, gt['box'])
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, g_idx

                if best_iou >= IOU_EVAL_THRESHOLD and best_gt_idx >= 0:
                    stats[cls_name]['tp'] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    stats[cls_name]['fp'] += 1
                    # Save False Positives for analysis
                    fp_crop = raw_img[max(0,y):y+h, max(0,x):x+w]
                    if fp_crop.size > 0:
                        cv2.imwrite(str(FP_SAVE_DIR / f"{cls_name}_fp_{idx}_{d_idx}.jpg"), fp_crop)

            stats[cls_name]['fn'] += len(cls_gt) - len(matched_gt)

    print("\n" + "="*50)
    print(f"{'Class':10} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10}")
    print("-"*50)
    for cls_name in ['fire', 'smoke']:
        tp, fp, fn = stats[cls_name]['tp'], stats[cls_name]['fp'], stats[cls_name]['fn']
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * p * r / (p + r) if (p + r) > 0 else 0)
        print(f"{cls_name:10} | {p:>10.3f} | {r:>10.3f} | {f1:>10.3f}")
        print(f"{'':10} | (TP={tp}, FP={fp}, FN={fn})")
    print("="*50)

if __name__ == "__main__":
    evaluate(data_split="test", max_images=40)