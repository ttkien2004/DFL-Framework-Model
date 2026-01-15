# Xử lý các thuận toán chính: CoCo, BALANCE, Aggregation
import torch
import numpy as np
import math
from app.models.cnn import SimpleCNN
from config import Config
from app.core.ldp_helpers import LDP
# Clas này xử lý phân cụm (DFCA), Huấn luyện cục bộ và thêm nhiễu LDP
class WorkerNode:
    def __init__(self, node_id, data_loader):
        self.id = node_id
        self.data_loader = data_loader
        self.device = Config.DEVICE
        self.model = SimpleCNN().to(self.device)
        self.cluster_id = None  # Sẽ được gán sau bước Clustering

    def evaluate_loss_on_model(self, model_state):
        """Bước 1: Tính Loss để chọn cụm (DFCA)"""
        self.model.load_state_dict(model_state)
        # ... logic tính loss trên local data ...
        return np.random.random() # Mock return loss

    def join_cluster(self, cluster_models):
        """Worker tự chọn cụm có Loss thấp nhất"""
        losses = {cid: self.evaluate_loss_on_model(model) for cid, model in cluster_models.items()}
        self.cluster_id = min(losses, key=losses.get)
        print(f"Worker {self.id} joined Cluster {self.cluster_id}")

    def train(self, epochs=1):
        """Bước 2: Huấn luyện cục bộ (Local Training)"""
        self.model.train()
        # ... logic training loop (SGD) ...
        print(f"Worker {self.id} finished training.")
        return self.model.state_dict()

    def apply_ldp(self, params):
        """
        Thực hiện LDP-Gauss theo 2 bước: Clipping và Adding Noise.
        Input: params (state_dict của model sau khi train)
        Output: noisy_params (state_dict đã được bảo vệ)
        """
        if not Config.ENABLE_LDP:
            return params

        # --- BƯỚC A: CẮT GỌN THAM SỐ (CLIPPING) - Công thức (29) ---
        
        # 1. Tính chuẩn L2 toàn cục (Global L2 Norm) của vector trọng số ||w_i||
        total_norm = 0.0
        for v in params.values():
            # Chỉ tính norm trên các tensor kiểu float (bỏ qua int64 như buffer đếm bước)
            if v.dtype in [torch.float, torch.float32, torch.float64]:
                param_norm = v.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = math.sqrt(total_norm)

        # 2. Tính hệ số cắt gọn (Clipping Scale)
        # scale = max(1, ||w_i|| / C)
        clip_threshold = Config.LDP_CLIPPING_THRESHOLD
        clip_scale = max(1.0, total_norm / clip_threshold)

        # --- BƯỚC B: CỘNG NHIỄU GAUSS - Công thức (30) ---
        
        noisy_params = {}
        sigma = LDP.get_ldp_sigma() # Độ lệch chuẩn của nhiễu
        
        for k, v in params.items():
            # 1. Thực hiện cắt gọn: w_bar = w / scale
            # Nếu norm < C thì scale = 1 (giữ nguyên), ngược lại sẽ bị thu nhỏ
            clipped_param = v / clip_scale
            
            # 2. Tạo ma trận nhiễu N(0, sigma^2) cùng kích thước với tham số
            if v.dtype in [torch.float, torch.float32, torch.float64]:
                noise = torch.normal(mean=0.0, std=sigma, size=v.shape).to(self.device)
                
                # 3. Cộng nhiễu: w_DP = w_bar + N
                noisy_params[k] = clipped_param + noise
            else:
                # Với các tham số không phải trọng số (VD: batchnorm running stats), giữ nguyên
                noisy_params[k] = v

        # (Tùy chọn) In log để kiểm tra xem có bị cắt nhiều không
        # if clip_scale > 1.0:
        #     print(f"Worker {self.id}: Model clipped! Norm={total_norm:.2f} > C={clip_threshold}")

        return noisy_params