import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

def fire_smoke_classifier(num_classes=3):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

if __name__ == '__main__':
    m = fire_smoke_classifier()
    print(m.fc)