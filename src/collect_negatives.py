import cv2
import torch
import os
from pathlib import Path
from model import fire_smoke_classifier
from torchvision import transforms
from PIL import Image

# Կարգավորումներ
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = fire_smoke_classifier(3)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def collect_hard_negatives(img_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    img_files = list(Path(img_dir).glob("*.jpg"))
    count = 0

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        # Selective Search՝ գտնելու այն հատվածները, որոնք մոդելին "խաբում են"
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()

        for (x, y, w, h) in rects[:300]: # Վերցնում ենք առաջին 300-ը
            if w < 100 or h < 100: continue
            
            crop_bgr = img[y:y+h, x:x+w]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            input_tensor = transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = model(input_tensor)
                probs = torch.softmax(out, dim=1)
                conf, pred = torch.max(probs, dim=1)
            
            # Եթե մոդելը 90%-ից ավել վստահությամբ ասում է Fire (1) կամ Smoke (2), 
            # բայց մենք գիտենք, որ սա սխալ է (Hard Negative)
            if conf > 0.90 and pred.item() in [1, 2]:
                save_path = f"{output_dir}/neg_{count}.jpg"
                cv2.imwrite(save_path, crop_bgr)
                count += 1
                if count % 50 == 0: print(f"Collected {count} hard negatives...")

if __name__ == "__main__":
    # Աշխատեցրու սա այն դիրեկտորիայի վրա, որտեղ շատ սխալներ է անում
    collect_hard_negatives("data/train/images", "classifier_data/train/background_hard")