# Mô hình CNN đơn giản
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def get_model(model_name, num_classes=10):
    """
    Factory function để lấy mô hình theo tên
    """
    model_name = model_name.lower()
    
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
    elif model_name == 'vgg9':
        return VGG9(num_classes=num_classes)
    elif model_name == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes)
    elif model_name == 'resnet20':
        return ResNet20(num_classes=num_classes)
    elif model_name == 'health_mlp':
        return HealthMLP(num_classes=num_classes)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")

class SimpleCNN(nn.Module):
    # def __init__(self, num_classes=10):
    #     super(SimpleCNN, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 32, 3)  # Changed from 3 to 1 channel
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(32, 64, 3)
    #     self.fc1 = nn.Linear(64 * 5 * 5, 64)  # Fixed for MNIST
    #     self.fc2 = nn.Linear(64, num_classes) # Dynamic num_classes

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = torch.flatten(x, 1)
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Chuyển từ 3 kênh (RGB) sang 1 kênh (Grayscale) cho MNIST
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # 64 channels * 5 * 5 (kích thước sau 2 lần pool) = 1600
        self.fc1 = nn.Linear(64 * 5 * 5, 64) 
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class VGG9(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG9, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64), # 8 groups
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 2 * 2, 512),
            nn.LayerNorm(512), # LayerNorm thay cho BatchNorm1d
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==========================================
# 3. RESNET-20 (Custom for Federated Learning)
# ==========================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, planes)  # GroupNorm thay cho BatchNorm
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_planes = 16

        # Lớp Convolution đầu tiên
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, 16)
        
        # 3 cụm Residual Layers (Mỗi cụm 3 block)
        self.layer1 = self._make_layer(BasicBlock, 16, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)
        
        # Lớp Linear cuối cùng
        self.linear = nn.Linear(64 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Dùng AdaptiveAvgPool2d để tự động lấy trung bình bất chấp kích thước ảnh đầu vào (32x32 hoặc 64x64)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class HarMLP(nn.Module):
    def __init__(self, input_dim=561, num_classes=6): 
        # UCI HAR có 561 đặc trưng 1D và 6 nhãn hành vi (Walking, Sitting, Laying,...)
        super(HarMLP, self).__init__()
        
        # Lớp ẩn 1: Tăng cường trích xuất đặc trưng
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256) # Dùng LayerNorm thân thiện với Federated Learning
        
        # Lớp ẩn 2
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        
        # Lớp ẩn 3
        self.fc3 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)
        
        # Lớp phân loại đầu ra
        self.fc_out = nn.Linear(64, num_classes)
        
        # Dropout chống học vẹt (Overfitting) do dữ liệu IoT thường nhiễu
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Đảm bảo input x là tensor 2D: (batch_size, 561)
        x = x.view(x.size(0), -1) 
        
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        
        x = self.fc_out(x)
        return x
    
class HealthMLP(nn.Module):
    def __init__(self, input_dim=36, num_classes=2): 
        super(HealthMLP, self).__init__()
        
        # Lớp ẩn 1
        self.fc1 = nn.Linear(input_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        
        # Lớp ẩn 2
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        
        # Lớp ẩn 3
        self.fc3 = nn.Linear(64, 32)
        self.ln3 = nn.LayerNorm(32)
        
        # Phân loại đầu ra (Anomaly hay Normal)
        self.fc_out = nn.Linear(32, num_classes)
        
        # Dropout để chống Overfitting trên thiết bị IoT
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        
        x = self.fc_out(x)
        return x