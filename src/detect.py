import torch
import cv2
import numpy as np
from torchvision import transforms
from model import fire_smoke_classifier
from PIL import Image

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
classes = ['background', 'fire', 'smoke']

model = fire_smoke_classifier(num_classes=3)
model.load_state_dict(torch.load("models/fire_smoke_classifier.pth"))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def detect_fire_rcnn(image_path):
    img = cv2.imread(image_path)
    if img is None: return

    height, width, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_display = img.copy()

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    
    boxes = []
    confidences = []
    class_ids = []
    
    for (x, y, w, h) in rects[:500]:
        if w < 50 or h < 50: continue
        
        crop = img_rgb[y:y+h, x:x+w]
        pil_img = Image.fromarray(crop)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, pred = torch.max(probs, 0)
            
            res_label = classes[pred.item()]
            conf_val = conf.item()

        if res_label == 'fire' and conf_val > 0.99:
            boxes.append([x, y, w, h])
            confidences.append(float(conf_val))
            class_ids.append(pred.item())
            
        elif res_label == 'smoke' and conf_val > 0.98: 
            if y > height * 0.85: continue
            if w > width * 0.8: continue
            if w * h < 5000: continue  
            if w * h > width * height * 0.3: continue
            if w > h * 2: continue
            
            boxes.append([x, y, w, h])
            confidences.append(float(conf_val))
            class_ids.append(pred.item())

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.80, 0.05)

    if len(indices) > 0:
        for i in indices:
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
            x, y, w, h = boxes[idx]
            label = classes[class_ids[idx]]
            conf_val = confidences[idx]

            color = (0, 0, 255) if label == 'fire' else (0, 255, 255)
            cv2.rectangle(img_display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_display, f"{label} {conf_val:.2f}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Detection - Trial Mode", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_fire_rcnn("data/valid/images/-109170_png.rf.376c81847b294d990665d8d36c79b20e.jpg")