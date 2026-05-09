import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import fire_smoke_classifier

def train():
    EPOCHS = 10
    BATCH_SIZE = 32
    LR = 1e-4
    PATIENCE  = 5       
    SAVE_PATH = "models/best_model.pth"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder("classifier_data/train",
                                    transform=train_transform)
    val_ds   = datasets.ImageFolder("classifier_data/valid",
                                    transform=val_transform)

    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model = fire_smoke_classifier(num_classes=3).to(device)

    if os.path.exists(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
        print("Նախկին model բեռնված։ Շարունակում ենք...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # ── Training Loop ────────────────────────────────────────────────
    best_val_loss   = float('inf')
    epochs_no_improve = 0

    os.makedirs("models", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):

        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * imgs.size(0)
            _, predicted   = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total   += imgs.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = 100. * train_correct / train_total

        # ── Validation ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total   += imgs.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total

        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch:2d}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.1f}%")

        # ── Save best model ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Best model saved (val loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{PATIENCE})")

        # ── Early stopping ──
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping — {PATIENCE} epoch բարելավում չկա։")
            break

    print(f"\nTraining ավարտ։ Best val loss: {best_val_loss:.4f}")
    print(f"Model պահված: {SAVE_PATH}")


if __name__ == "__main__":
    train()