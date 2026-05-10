import torch
import cv2
import numpy as np
from torchvision import transforms
from model import fire_smoke_classifier
from PIL import Image
import os

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


def has_fire_or_smoke_colors(crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, w = crop_bgr.shape[:2]
    total = h * w

    green_mask  = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    green_ratio = green_mask.sum() / 255 / total
    if green_ratio > 0.30:
        return False

    fire_mask1 = cv2.inRange(hsv, (0,   100, 100), (35,  255, 255))
    fire_mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    fire_mask  = cv2.bitwise_or(fire_mask1, fire_mask2)
    fire_ratio = fire_mask.sum() / 255 / total

    smoke_mask  = cv2.inRange(hsv, (0, 0, 50), (180, 50, 200))
    smoke_ratio = smoke_mask.sum() / 255 / total

    return fire_ratio > 0.05 or smoke_ratio > 0.50


def load_ground_truth(label_path, img_w, img_h):
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.split()
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
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union = float(boxAArea + boxBArea - interArea)
    return interArea / union if union > 0 else 0


def detect_and_evaluate(image_path, label_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Նկարը չի բեռնվել: {image_path}")
        return
    h_img, w_img, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display = img.copy()

    gt_data = load_ground_truth(label_path, w_img, h_img)
    print(f"Ground Truth boxes: {len(gt_data)}")
    for gt in gt_data:
        cls_name = 'fire' if gt['class'] == 0 else 'smoke'
        b = gt['box']
        cv2.rectangle(display, (b[0], b[1]), (b[2], b[3]), (255, 180, 0), 3)
        cv2.putText(display, f"GT:{cls_name}", (b[0], b[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

    # Selective Search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    print(f"Selective Search proposals: {len(rects)}")

    crops, boxes = [], []
    skipped_color = 0

    for (x, y, w_box, h_box) in rects[:500]:
        if w_box < 100 or h_box < 100:
            continue
        if (w_box * h_box) > (w_img * h_img * 0.5):
            continue

        crop_bgr = img[y:y+h_box, x:x+w_box]

        # Color filter
        if not has_fire_or_smoke_colors(crop_bgr):
            skipped_color += 1
            continue

        crop_rgb = rgb[y:y+h_box, x:x+w_box]
        crops.append(transform(Image.fromarray(crop_rgb)))
        boxes.append([x, y, w_box, h_box])

    if not crops:
        print("No Proposals")
        cv2.imshow("Result", display)
        cv2.waitKey(0)
        return

    batch_size = 64
    all_confs, all_preds = [], []
    for i in range(0, len(crops), batch_size):
        batch = torch.stack(crops[i:i+batch_size]).to(device)
        with torch.no_grad():
            out = model(batch)
            probs = torch.softmax(out, dim=1)
            confs, preds = torch.max(probs, dim=1)
        all_confs.extend(confs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    conf_np = np.array(all_confs)
    pred_np = np.array(all_preds)
    for cls_id, cls_name in enumerate(classes):
        mask = pred_np == cls_id
        if mask.sum() > 0:
            print(f"  '{cls_name}': {mask.sum()} | "
                  f"max={conf_np[mask].max():.3f} "
                  f"mean={conf_np[mask].mean():.3f}")

    conf_threshold = 0.999
    nms_threshold = 0.05

    # Per-class NMS
    kept_boxes = []
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
                kept_boxes.append({
                    'box': cls_boxes[idx],
                    'conf': cls_confs[idx],
                    'cls_id': target_cls_id
                })

    print(f"Final detections after NMS: {len(kept_boxes)}")

    for det in kept_boxes:
        x, y, w_b, h_b = det['box']
        pred_box    = [x, y, x + w_b, y + h_b]
        pred_cls_id = det['cls_id']
        yolo_cls    = pred_cls_id - 1

        is_correct = any(
            calculate_iou(pred_box, gt['box']) > 0.25
            and yolo_cls == gt['class']
            for gt in gt_data
        )

        color = (0, 255, 0) if is_correct else (0, 0, 255)
        cv2.rectangle(display, (x, y), (x + w_b, y + h_b), color, 2)
        cv2.putText(display, f"{classes[pred_cls_id]} {det['conf']:.2f}",
                    (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    correct = sum(
        1 for d in kept_boxes
        if any(
            calculate_iou([d['box'][0], d['box'][1],
                           d['box'][0]+d['box'][2],
                           d['box'][1]+d['box'][3]],
                           gt['box']) > 0.25 and (d['cls_id'] - 1) == gt['class'] for gt in gt_data
        )
    )
    print(f"Correct: {correct}/{len(kept_boxes)} | "
          f"False Positives: {len(kept_boxes) - correct}")

    cv2.imshow("Blue=GT | Green=Correct | Red=FP", display)
    cv2.waitKey(0)


if __name__ == "__main__":
    img_name = "-189475_png.rf.b85d47a7ffdbb6a4f1e03160fb8cd426"
    img_path = f"data/test/images/{img_name}.jpg"
    lbl_path = f"data/test/labels/{img_name}.txt"
    detect_and_evaluate(img_path, lbl_path)