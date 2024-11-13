import torch.nn as nn
import torchvision.models as models

from Models.Model import Model

class ResNet18(Model):
    def __init__(self, device, num_classes, train_loader, val_loader, test_loader, pretrained=True, freeze_features=True):
        super().__init__(device, train_loader, val_loader, test_loader)
        
        self.model = models.resnet18(pretrained=pretrained)
        
        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)