import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import fire_smoke_classifier

def train():
    epochs = 10
    batch_size = 32
    lr = 1e-4
    patience = 5       
    save_path = "models/best_model.pth"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder("classifier_data/train", transform=train_transform)
    val_ds = datasets.ImageFolder("classifier_data/valid", transform=val_transform)

    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = fire_smoke_classifier(num_classes=3).to(device)

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print("Previous model loaded")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    #Training Loop
    best_val_loss = float('inf')
    epochs_no_improve = 0

    os.makedirs("models", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += imgs.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = 100 * train_correct / train_total

        #Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += imgs.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total

        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch:2d}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.1f}%")

        #Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved (val loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement ({epochs_no_improve}/{patience})")

        #Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping")
            break

    print(f"\nTraining finished: Best val loss: {best_val_loss:.4f}")
    print(f"Model saved: {save_path}")


if __name__ == "__main__":
    train()