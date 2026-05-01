import cv2
import os
from pathlib import Path

def process_data(data_split, limit_per_class=5000):
    img_dir = f"data/{data_split}/images"
    label_dir = f"data/{data_split}/labels"
    output_base = f"classifier_data/{data_split}"

    for cls in ['fire', 'smoke', 'background']:
        os.makedirs(f"{output_base}/{cls}", exist_ok=True)

    counts = {'fire': 0, 'smoke': 0}

    for label_file in Path(label_dir).glob("*.txt"):
        if counts['fire'] >= limit_per_class and counts['smoke'] >= limit_per_class:
            break

        img_path = Path(img_dir) / label_file.name.replace(".txt", ".jpg")
        if not img_path.exists():
            img_path = Path(img_dir) / label_file.name.replace(".txt", ".png")

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w, _ = img.shape

        with open(label_file, 'r') as f:
            for i, line in enumerate(f.readlines()):
                parts = line.split()

                if len(parts) != 5:
                    continue

                cls_id = int(parts[0])
                cls_name = 'fire' if cls_id == 0 else 'smoke'

                if counts[cls_name] >= limit_per_class:
                    continue

                x_c, y_c, nw, nh = map(float, parts[1:])

                x1 = int((x_c - nw / 2) * w)
                y1 = int((y_c - nh / 2) * h)
                x2 = int((x_c + nw / 2) * w)
                y2 = int((y_c + nh / 2) * h)

                crop = img[max(0, y1):min(h, y2),
                            max(0, x1):min(w, x2)]

                if crop.size == 0:
                    continue

                save_path = f"{output_base}/{cls_name}/{label_file.stem}_{i}.jpg"
                cv2.imwrite(save_path, crop)

                counts[cls_name] += 1

    print(f"{data_split}  fire: {counts['fire']}, smoke: {counts['smoke']}")

if __name__ == "__main__":
    process_data("train", limit_per_class=5000)
    process_data("valid", limit_per_class=1000)