import torch
import math
import numpy as np
from config import Config

class Balance:
    @staticmethod
    def flatten_model(self, state_dict):
        """Chuyển toàn bộ model thành 1 vector duy nhất để tính Norm"""
        # Chỉ lấy các tensor kiểu float/int, bỏ qua các buffer không cần thiết
        tensors = [p.view(-1) for p in state_dict.values()]
        return torch.cat(tensors).to(Config.DEVICE)
    
    @staticmethod
    def calculate_adaptive_threshold(self, global_norm, distances, round_k):
        """
        Tính ngưỡng động dựa trên công thức BALANCE
        """
        total_rounds = Config.NUM_ROUNDS
        
        # 1. Tính Baseline Threshold (Giảm dần theo thời gian)
        # Omega chạy từ 0 -> 1 khi round tăng dần
        omega = round_k / max(1, total_rounds) 
        
        # Baseline = Gamma * exp(-Lambda * Omega) * ||W_global||
        baseline = (Config.BALANCE_GAMMA * math.exp(-Config.BALANCE_LAMBDA * omega) * global_norm)
        
        if not distances: 
            return baseline

        # 2. Tính Spatial Scale (Dựa trên độ phân tán của các Worker gửi lên)
        # Tính sai số tương đối: q_i = ||w_i - w_global|| / ||w_global||
        qs = [(dist / (global_norm + 1e-12)) for dist in distances]
        
        # Lấy trung vị (Median) để tránh bị ảnh hưởng bởi Outlier (kẻ tấn công)
        med_q = np.median(qs)
        
        # Scale = max(1.0, Beta * med_q / Q0)
        spatial_scale = max(1.0, Config.BALANCE_BETA * (med_q / Config.BALANCE_Q0))
        
        # 3. Ngưỡng cuối cùng
        threshold = baseline * spatial_scale
        
        print(f"[BALANCE-CH{self.cluster_id}] Round {round_k}: Baseline={baseline:.4f}, Scale={spatial_scale:.2f}, Threshold={threshold:.4f}")
        return threshold