# Mô hình CNN đơn giản
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def get_model(model_name, num_classes=10):
    """
    Factory function để lấy mô hình theo tên
    """
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
        
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity() # Bỏ maxpool đầu tiên
        
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        # Sửa classifier cuối
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model
        
    elif model_name == 'simple_cnn':
        return SimpleCNN()
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, num_classes) # Dynamic num_classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x