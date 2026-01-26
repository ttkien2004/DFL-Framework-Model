# Xử lý các thuận toán chính: CoCo, BALANCE, Aggregation
import torch
import numpy as np
from app.models.cnn import SimpleCNN
from config import Config

# PIPELINE: CHIA CỤM -> COCO CHO TỪNG CỤM -> LOCAL TRAINNING -> AGGREGATE

# Class này xử lý phân cụm (DFCA), Huấn luyện cục bộ và thêm nhiễu LDP
class WorkerNode:
    def __init__(self, node_id, data_loader):
        self.id = node_id
        self.data_loader = data_loader # Pytorch DataLoader
        self.device = Config.DEVICE
        self.model = SimpleCNN().to(self.device)
        self.cluster_id = None  # Sẽ được gán sau bước Clustering

    def evaluate_loss_on_model(self, model_state):
        """Bước 1: Tính Loss để chọn cụm, (DFCA)"""
        self.model.load_state_dict(model_state) # load learned weights of model
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
        losses = {cid: self.evaluate_loss_on_model(model) for cid, model in cluster_models.items()}
        self.cluster_id = min(losses, key=losses.get)
        print(f"Worker {self.id} joined Cluster {self.cluster_id} with loss {losses[self.cluster_id]:.4f}")

    # Local update
    def train(self, local_epochs=1):
        """ Huấn luyện cục bộ bằng SGD
        """
        self.model.train()

        for epoch in range(local_epochs):
            for X, y in self.data_loader:
                X, y = X.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(X)
                loss = self.loss_fn(preds, y)
                loss.backward()
                self.optimizer.step()

        print(f"[Worker {self.id}] finished local training.")
        return self.model.state_dict()

    # Tổng hợp mô hình 
    def graph_sequential_aggregate(worker_states, adjacency_matrix):
        """
        worker_states: dict {node_id: state_dict}
        adjacency_matrix: 2D list / numpy array / torch tensor (N x N)
        """

        new_states = {}

        N = len(adjacency_matrix)

        for i in range(N):
            aggregated_state = None
            r = 0

            for j in range(N):
                if adjacency_matrix[i][j] == 0:
                    continue

                r += 1
                neighbor_state = worker_states[j]

                if aggregated_state is None:
                    aggregated_state = {
                        k: v.clone()
                        for k, v in neighbor_state.items()
                    }
                else:
                    for k in aggregated_state:
                        aggregated_state[k] = (
                            (r - 1) / r * aggregated_state[k]
                            + 1 / r * neighbor_state[k]
                        )

            new_states[i] = aggregated_state

        return new_states





    def apply_ldp(self, params, epsilon=0.5):
        """Bước 3: Thêm nhiễu LDP (Gaussian/Laplace)"""
        noisy_params = {}
        for k, v in params.items():
            noise = torch.normal(0, Config.LDP_EPSILON, size=v.shape).to(self.device)
            noisy_params[k] = v + noise
        return noisy_params