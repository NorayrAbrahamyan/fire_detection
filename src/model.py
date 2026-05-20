import torch.nn as nn
from torchvision import models

def fire_smoke_classifier(num_classes=3, is_training=False):
    model = models.resnet18(weights='IMAGENET1K_V1' if is_training else None)
    for param in model.parameters():
        param.requires_grad = False
        
    if is_training:
        for param in model.layer4.parameters():
            param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model