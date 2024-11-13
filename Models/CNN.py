import torch.nn as nn

from Models.Model import Model
from utils.globalConst import *

class CNN(Model):
    def __init__(self, device, num_classes, train_loader, val_loader, test_loader):
        super().__init__(device, train_loader, val_loader, test_loader)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
    
        self.fc1 = nn.Sequential(
            nn.Linear(64 * int(IMG_SIZE // 4.0) * int(IMG_SIZE // 4.0), 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        return x