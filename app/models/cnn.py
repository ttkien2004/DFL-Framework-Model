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
        # Load ResNet18 (không pretrain để test khả năng học từ đầu)
        model = models.resnet18(weights=None)
        
        # Sửa lớp đầu tiên cho phù hợp với ảnh nhỏ 32x32 của CIFAR-10
        # (Mặc định ResNet dùng kernel 7x7 và stride 2, làm ảnh nhỏ bị mất thông tin quá nhanh)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity() # Bỏ maxpool đầu tiên
        
        # Sửa lớp cuối cùng (Output layer)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        # Sửa classifier cuối
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model
        
    elif model_name == 'simplenet':
        return SimpleNet()
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x