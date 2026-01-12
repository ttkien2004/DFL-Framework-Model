import torch
from app.models.cnn import SimpleCNN
from config import Config
from app.utils.helpers import federated_averaging, compute_euclidean_distance, compute_model_hash
# Class này xử lý CoCom lọc BALANCE và tổng hợp
class ClusterHead:
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        self.global_model = SimpleCNN().to(Config.DEVICE)
        self.members = []

    def receive_update(self, worker_id, noisy_params):
        """Nhận cập nhật từ worker"""
        self.members.append(noisy_params)

    def balance_filtering(self, threshold=10.0):
        """Bước 4: Lọc mô hình độc hại (BALANCE Adaptive)"""
        valid_updates = []
        for update in self.members:
            # Mock logic tính khoảng cách Euclid và so với ngưỡng
            norm_diff = 5.0 # giả lập
            if norm_diff < Config.BALANCE_THRESHOLD:
                valid_updates.append(update)
        return valid_updates

    def aggregate(self):
        """Bước 5: Tổng hợp mô hình (FedAvg)"""
        updates = self.balance_filtering()
        if not updates:
            return None
        
        # Logic trung bình trọng số (Avg)
        avg_state = updates[0] # Simplification
        # ... thực hiện cộng gộp ...
        self.global_model.load_state_dict(avg_state)
        self.members = [] # Reset cho vòng sau
        return self.global_model.state_dict()