import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import fire_smoke_classifier
import os

def train_model():
    batch_size = 32
    epochs = 5
    learning_rate = 0.001

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),

        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05
        ),

        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # 🔥 smoke-ի համար շատ կարևոր

        transforms.ToTensor(),
 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder('classifier_data/train', transform=train_transform)
    val_dataset = datasets.ImageFolder('classifier_data/valid', transform=val_transform)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(train_dataset.class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = fire_smoke_classifier(num_classes=3)
    weights_path = "models/fire_smoke_classifier.pth"
    if os.path.exists(weights_path):
        print(f"Loading existing weights from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda par: par.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                

        train_acc = train_correct / len(train_dataset)
        val_acc = val_correct / len(val_dataset)
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f} | Acc: {val_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/fire_smoke_classifier.pth")

if __name__ == '__main__':
    train_model()