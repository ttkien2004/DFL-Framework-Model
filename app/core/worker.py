# Xử lý các thuật toán chính: CoCo, BALANCE, Aggregation
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from app.models.cnn import SimpleCNN
from config import Config

# Class này xử lý phân cụm (DFCA), Huấn luyện cục bộ và thêm nhiễu LDP
class WorkerNode:
    def __init__(self, node_id, data_loader):
        self.id = node_id
        self.data_loader = data_loader
        self.device = Config.DEVICE
        self.model = SimpleCNN().to(self.device)
        self.cluster_id = None  # Sẽ được gán sau bước Clustering
        self.loss_fn = nn.CrossEntropyLoss()

    def evaluate_loss_on_model(self, model_state):
        """Bước 1: Tính Loss để chọn cụm, (DFCA)"""
        self.model.load_state_dict(model_state)
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for X, y in self.data_loader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self.model(X)
                loss = self.loss_fn(preds, y) 
                
                total_loss += loss.item() * X.size(0)
                total_samples += X.size(0)
                
        return total_loss / total_samples if total_samples > 0 else float('inf')

    def join_cluster(self, cluster_models):
        """Worker tự chọn cụm có Loss thấp nhất"""
        losses = {}
        for cid, model_state in cluster_models.items():
            loss = self.evaluate_loss_on_model(model_state)
            losses[cid] = loss
            
        self.cluster_id = min(losses, key=losses.get)
        print(f"Worker {self.id} joined Cluster {self.cluster_id} with loss {losses[self.cluster_id]:.4f}")


    def train(self, epochs=1):
        """Bước 2: Huấn luyện cục bộ (Local Training)"""
        self.model.train()
        # ... logic training loop (SGD) ...
        print(f"Worker {self.id} finished training.")
        return self.model.state_dict()

    def apply_ldp(self, params, epsilon=0.5):
        """Bước 3: Thêm nhiễu LDP (Gaussian/Laplace)"""
        noisy_params = {}
        for k, v in params.items():
            noise = torch.normal(0, Config.LDP_EPSILON, size=v.shape).to(self.device)
            noisy_params[k] = v + noise
        return noisy_params