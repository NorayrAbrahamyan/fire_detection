import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
import os
import sys

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
classes = ['background', 'fire', 'smoke']

CONF_THRESHOLD_FIRE = 0.85   
CONF_THRESHOLD_SMOKE = 0.85
NMS_THRESHOLD = 0.01         

def get_model(num_classes):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5), 
        nn.Linear(num_ftrs, num_classes)
    )
    return model

model = get_model(len(classes))
model_path = "models/best_model.pth"

if os.path.exists(model_path):
    print(f"✅ Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
else:
    print(f"❌ ERROR: Model file not found at {model_path}")
    sys.exit(1)

model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def run_inference(image_path, output_path="result.jpg"):
    img = cv2.imread(image_path)
    if img is None: 
        print(f"❌ Նկարը չգտնվեց: {image_path}")
        return
    
    img = cv2.resize(img, (640, 480))
    
    print(f"🕒 Processing image: {image_path}...")
    h_img, w_img, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display = img.copy()

    # 1. Selective Search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    crops, boxes = [], []
    for (x, y, w_box, h_box) in rects[:600]:
        if w_box < 40 or h_box < 40: continue
        crops.append(transform(Image.fromarray(rgb[y:y+h_box, x:x+w_box])))
        boxes.append([x, y, w_box, h_box])

    if not crops:
        print("ℹ️ No proposals to classify.")
        cv2.imwrite(output_path, display)
        return

    # 2. Batch Inference
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

    # 3. Post-Processing (Threshold & NMS)
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
                detections.append({
                    'box': cls_boxes[idx], 
                    'conf': cls_confs[idx], 
                    'cls_name': classes[target_cls_id]
                })

    # 4. Drawing Results
    if not detections:
        print("ℹ️ Nothing detected.")
    else:
        print(f"🔥 Found {len(detections)} objects.")

    for det in detections:
        x, y, w, h = det['box']
        color = (255, 0, 0) if det['cls_name'] == 'fire' else (0, 165, 255) 
        cv2.rectangle(display, (x, y), (x+w, y+h), color, 3)
        label = f"{det['cls_name']}: {det['conf']:.2f}"
        cv2.putText(display, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 5. Save Result
    cv2.imwrite(output_path, display)
    print(f"💾 Result saved to: {output_path}")

if __name__ == "__main__":
    my_image = "src/test/pexels-yavuz-solgun-26647055-28536734-scaled.jpg" 
    output_image = "result_output.jpg" 
    
    run_inference(my_image, output_image)