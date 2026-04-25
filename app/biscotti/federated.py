import torch
import torch.nn as nn
from opacus import PrivacyEngine
from app.models.cnn import get_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import copy
from app.utils.data_loader import PersonalHealthDataset
import numpy as np

class FederatedNode:
    def __init__(self, model_name='simple_cnn', num_classes=10, dataset_name='mnist'):
        # Tự động chuyển đổi mô hình dựa trên tập dữ liệu
        # if dataset_name == 'health':
        #     model_name = 'health_mlp'
        #     num_classes = 2
        #     input_dim = 36
        
        # self.model = get_model(model_name, num_classes)
        # self.dataset_name = dataset_name
        
        # if self.dataset_name == 'mnist':
        #     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        #     self.dataset = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform), batch_size=32, shuffle=True)
        
        # elif self.dataset_name == 'health':
        #     csv_path = './data/personal_health_data.csv'
        #     full_dataset = PersonalHealthDataset(csv_path)
            
        #     # Chia 80-20 với seed cố định
        #     train_size = int(0.8 * len(full_dataset))
        #     test_size = len(full_dataset) - train_size
        #     generator = torch.Generator().manual_seed(42)
        #     train_ds, _ = random_split(full_dataset, [train_size, test_size], generator=generator)
            
        #     # Gán thuộc tính targets để tương thích với các hàm chia dữ liệu Non-IID
        #     train_ds.targets = np.array([full_dataset.y[i].item() for i in train_ds.indices])
        #     self.dataset = DataLoader(train_ds, batch_size=32, shuffle=True)
        self.model = get_model(model_name, num_classes)
        self.dataset = None
        
        # Opacus privacy engine
        self.privacy_engine = PrivacyEngine()
        self.is_private_attached = False
    
    def set_local_dataset(self, local_subset, batch_size=32):
        """Hàm mới để gán dữ liệu Dirichlet Subset cho nút"""
        self.dataset = DataLoader(local_subset, batch_size=batch_size, shuffle=True)

    def train_local(self, global_weights, noise_scale=1.0, is_malicious=False, attack_type="NONE", src_class=3, tgt_class=5):
        if global_weights:
            self.model.load_state_dict(global_weights)
            
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        # Chỉ attach privacy engine 1 lần (Opacus thường báo lỗi nếu attach lại)
        if not self.is_private_attached:
            try:
                self.model, optimizer, self.dataset = self.privacy_engine.make_private(
                    module=self.model, optimizer=optimizer, data_loader=self.dataset, noise_multiplier=noise_scale, max_grad_norm=1.0
                )
                self.is_private_attached = True
            except Exception:
                pass

        self.model.train()
        device = next(self.model.parameters()).device

        # Vòng lặp huấn luyện cục bộ
        for data, target in self.dataset:
            data, target = data.to(device), target.to(device)

            # ---------------------------------------------------------
            # LOGIC TẤN CÔNG BỀ MẶT DỮ LIỆU (DATA-POISONING)
            # ---------------------------------------------------------
            if is_malicious:
                if attack_type == "LABEL_FLIPPING":
                    # Đảo nhãn src_class thành tgt_class
                    mask = (target == src_class)
                    target[mask] = tgt_class
                    
                elif attack_type == "BACKDOOR":
                    # Giả lập Trigger: Đánh dấu pixel góc và gán lại nhãn
                    data[:, :, 0:3, 0:3] = 2.5 
                    target[:] = tgt_class

            optimizer.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

        # Lấy bản sao trọng số mới nhất
        updated_weights = copy.deepcopy(self.model.state_dict())

        # ---------------------------------------------------------
        # LOGIC TẤN CÔNG BỀ MẶT MÔ HÌNH (MODEL-POISONING)
        # ---------------------------------------------------------
        if is_malicious and attack_type == "GAUSS":
            for name, param in updated_weights.items():
                # Tiêm nhiễu Gaussian với độ lệch chuẩn 0.5 để phá hoại hội tụ
                noise = torch.randn_like(param) * 0.5 
                updated_weights[name] = param + noise

        return updated_weights

def aggregate_updates(accepted_updates, global_weights):
    if not accepted_updates:
        return global_weights
    
    # Trung bình hóa FedAvg
    aggregated_weights = copy.deepcopy(accepted_updates[0])
    for key in aggregated_weights.keys():
        for i in range(1, len(accepted_updates)):
            aggregated_weights[key] += accepted_updates[i][key]
        aggregated_weights[key] = torch.div(aggregated_weights[key], len(accepted_updates))
        
    return aggregated_weights