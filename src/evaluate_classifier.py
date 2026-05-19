import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path

#DEVICE CONFIGURATION 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
classes = ['background', 'fire', 'smoke']

#MODEL DEFINITION
def get_model(num_classes):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5), 
        nn.Linear(num_ftrs, num_classes)
    )
    return model

#MODEL INITIALIZATION 
model = get_model(3) 
model_path = "models/best_model.pth"

if os.path.exists(model_path):
    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
else:
    print("ERROR: Model file not found!")
    exit()

model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#EVALUATION FUNCTION
def evaluate_pure_classifier(data_split="test", max_images=200):
    img_dir = Path(f"data/{data_split}/images")
    label_dir = Path(f"data/{data_split}/labels")
    img_files = (list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))[:max_images]

    print(f"Running Pure Classifier Test on {len(img_files)} Ground Truth patches...")
    
    correct = 0
    total = 0

    class_correct = {1: 0, 2: 0} # 1: fire, 2: smoke
    class_total = {1: 0, 2: 0}

    for img_path in img_files:
        label_path = label_dir / img_path.name.replace(img_path.suffix, ".txt")
        if not label_path.exists(): continue

        img = cv2.imread(str(img_path))
        if img is None: continue
        h_img, w_img, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5: continue
                yolo_cls, xc, yc, nw, nh = map(float, parts)
                target_resnet_cls = int(yolo_cls) + 1 

                x1 = max(0, int((xc - nw/2) * w_img))
                y1 = max(0, int((yc - nh/2) * h_img))
                x2 = min(w_img, int((xc + nw/2) * w_img))
                y2 = min(h_img, int((yc + nh/2) * h_img))

                crop = rgb[y1:y2, x1:x2]
                if crop.size == 0: continue

                tensor = transform(Image.fromarray(crop)).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(tensor)
                    pred_cls = torch.argmax(output, dim=1).item()

                if pred_cls == target_resnet_cls:
                    correct += 1
                    class_correct[target_resnet_cls] += 1
                
                total += 1
                class_total[target_resnet_cls] += 1

    print("\n" + "="*40)
    print("PURE CLASSIFIER ACCURACY REPORT")
    print("="*40)
    if total > 0:
        overall_acc = (correct / total) * 100
        print(f"Overall Accuracy: {overall_acc:.2f}% ({correct}/{total})")
        print("-"*40)
        for cid, cname in [(1, 'fire'), (2, 'smoke')]:
            if class_total[cid] > 0:
                c_acc = (class_correct[cid] / class_total[cid]) * 100
                print(f"Class '{cname}' Accuracy: {c_acc:.2f}% ({class_correct[cid]}/{class_total[cid]})")
            else:
                print(f"Class '{cname}': No Ground Truth data found.")
    else:
        print("No ground truth labels processed.")
    print("="*40)

if __name__ == "__main__":
    evaluate_pure_classifier(data_split="test", max_images=5000)