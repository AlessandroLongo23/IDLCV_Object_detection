import torch.nn as nn
import torchvision.models as models

from Models.Model import Model

class VGG16(Model):
    def __init__(self, device, num_classes, train_loader, val_loader, test_loader, pretrained=True, freeze_features=True):
        super().__init__(device, train_loader, val_loader, test_loader)
        
        self.model = models.vgg16(pretrained=pretrained)
        
        if freeze_features:
            for param in self.model.features.parameters():
                param.requires_grad = False
        
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)