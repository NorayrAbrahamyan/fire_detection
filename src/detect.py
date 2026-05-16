import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
import os

# --- DEVICE ---
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
classes = ['background', 'fire', 'smoke']

# --- MODEL DEFINITION (Պետք է համընկնի train.py-ի հետ) ---
def get_model(num_classes):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    # Ճիշտ նույն կառուցվածքը, ինչ օգտագործվել է մարզման ժամանակ
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5), 
        nn.Linear(num_ftrs, num_classes)
    )
    return model

# --- MODEL SETUP ---
model = get_model(len(classes))
model_path = "models/best_model.pth"

if os.path.exists(model_path):
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"✅ Model loaded successfully from {model_path}")
else:
    print(f"❌ ERROR: Model not found at {model_path}")

model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- UTILS ---

def has_fire_or_smoke_colors(crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, w = crop_bgr.shape[:2]
    total_pixels = h * w
    
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / total_pixels

    if green_ratio > 0.60: # Եթե 60%-ից ավելին կանաչ է, սա կրակ չէ
        return False
    return True 

def load_ground_truth(label_path, img_w, img_h):
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.split()
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
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union = float(boxAArea + boxBArea - interArea)
    return interArea / union if union > 0 else 0

# --- MAIN DETECTION ---

def detect_and_evaluate(image_path, label_path):
    img = cv2.imread(image_path)
    if img is None: 
        print(f"Նկարը չգտնվեց: {image_path}")
        return
    h_img, w_img, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display = img.copy()
    
    gt_data = load_ground_truth(label_path, w_img, h_img)
    for gt in gt_data:
        b = gt['box']
        cv2.rectangle(display, (b[0], b[1]), (b[2], b[3]), (255, 180, 0), 3)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    crops, boxes = [], []
    for (x, y, w_box, h_box) in rects[:400]:
        if w_box < 40 or h_box < 40: continue 

        crop_bgr = img[y:y+h_box, x:x+w_box]
        if not has_fire_or_smoke_colors(crop_bgr): continue

        crops.append(transform(Image.fromarray(rgb[y:y+h_box, x:x+w_box])))
        boxes.append([x, y, w_box, h_box])

    if not crops:
        print("No Proposals to classify.")
        cv2.imshow("Result", display); cv2.waitKey(0); return

    all_confs, all_preds = [], []
    batch_size = 64
    for i in range(0, len(crops), batch_size):
        batch = torch.stack(crops[i:i+batch_size]).to(device)
        with torch.no_grad():
            out = model(batch)
            probs = torch.softmax(out, dim=1)
            confs, preds = torch.max(probs, dim=1)
        all_confs.extend(confs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    fire_thresh, smoke_thresh = 0.90, 0.98
    kept_boxes = []
    
    for target_cls_id in [1, 2]:
        current_thresh = fire_thresh if target_cls_id == 1 else smoke_thresh
        cls_boxes, cls_confs = [], []
        for i in range(len(all_preds)):
            if all_preds[i] == target_cls_id and all_confs[i] > current_thresh:
                cls_boxes.append(boxes[i])
                cls_confs.append(float(all_confs[i]))
        
        if not cls_boxes: continue
        
        # Խստացում 4: Շատ խիստ NMS (0.1 -> 0.01)
        indices = cv2.dnn.NMSBoxes(cls_boxes, cls_confs, current_thresh, 0.01)
        if len(indices) > 0:
            for idx in indices.flatten():
                kept_boxes.append({
                    'box': cls_boxes[idx],
                    'conf': cls_confs[idx],
                    'cls_id': target_cls_id
                })

    correct_count = 0
    for det in kept_boxes:
        x, y, w, h = det['box']
        pred_box = [x, y, x+w, y+h]
        yolo_cls_id = det['cls_id'] - 1 
        
        is_tp = any(calculate_iou(pred_box, gt['box']) > 0.15 and yolo_cls_id == gt['class'] for gt in gt_data)
        
        color = (0, 255, 0) if is_tp else (0, 0, 255) # Կանաչ եթե TP է, Կարմիր եթե FP
        if is_tp: correct_count += 1

        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
        label = f"{classes[det['cls_id']]} {det['conf']:.2f}"
        cv2.putText(display, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    print(f"Final Detections: {len(kept_boxes)} | Correct (TP): {correct_count}")
    cv2.imshow("Result - Cleaned Detection", display)
    cv2.waitKey(0)

if __name__ == "__main__":
    img_name = "WEBFire1709_jpg.rf.3d2a0ae2bca55e9761ef100afe853495"
    img_path = f"data/test/images/{img_name}.jpg"
    lbl_path = f"data/test/labels/{img_name}.txt"
    detect_and_evaluate(img_path, lbl_path)