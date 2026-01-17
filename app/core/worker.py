# Xử lý các thuận toán chính: CoCo, BALANCE, Aggregation
import torch
import numpy as np
import math
from app.models.cnn import SimpleCNN
from config import Config
from app.core.ldp_helpers import LDP
from app.core.attacks import AttackFactory
from app.utils.data_loader import get_dataloader
from app.models.cnn import get_model
# Clas này xử lý phân cụm (DFCA), Huấn luyện cục bộ và thêm nhiễu LDP
class WorkerNode:
    def __init__(self, node_id, dataset_name='cifar10'):
        self.id = node_id
        # self.data_loader = data_loader
        self.dataset_name = dataset_name
        self.device = Config.DEVICE
        self.model = SimpleCNN().to(self.device)
        self.cluster_id = None  # Sẽ được gán sau bước Clustering
        self.attack_type = "NONE"
        self.attack_strategy = AttackFactory.get_strategy("NONE")

        # Khởi tạo Optimizer & Loss
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=Config.LEARNING_RATE, 
            momentum=Config.MOMENTUM
        )
        # Các tham số sử dụng cho giải thuật DFCA
        self.cluster_models_cache = {}
        self.update_counts = {} # số lượng cập nhật đã nhận cho mỗi cụm

    def reload_dataset(self, dataset_name):
        self.dataset_name = dataset_name

        self.data_loader, self.num_classes, _ = get_dataloader(
            dataset_name, self.id, Config.NUM_WORKERS
        )
        # Khởi tạo Model mới (khớp với num_classes)
        self.model = get_model(Config.MODEL_NAME, num_classes=self.num_classes).to(self.device)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=Config.LEARNING_RATE)

        print(f"[Worker {self.id}] Reloaded: {dataset_name} (Classes: {self.num_classes})")

    def set_attack_profile(self, attack_type):
        self.attack_type = AttackFactory.get_strategy(attack_type)

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

    def train(self):
        """Bước 2: Huấn luyện cục bộ (Local Training)"""
        # self.model.train()
        # # ... logic training loop (SGD) ...
        # print(f"Worker {self.id} finished training.")
        # return self.model.state_dict()
        return self.attack_strategy.execute(self)

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
    
    def apply_dfca_gossip_update(self, neighbor_cluster_id, neighbor_params):
        """
        Thực hiện cập nhật Sequential Running Average theo công thức DFCA (26).
        
        :param neighbor_cluster_id: ID cụm của mô hình hàng xóm gửi tới (c*)
        :param neighbor_params: Tham số mô hình của hàng xóm (w_{j,c*})
        """
        
        # Trường hợp 1: Nếu đây là lần đầu tiên Worker thấy mô hình của cụm này
        # Thì khởi tạo bằng chính mô hình hàng xóm (tương đương r=0)
        if neighbor_cluster_id not in self.cluster_models_cache:
            # Deep copy để tránh tham chiếu vùng nhớ
            self.cluster_models_cache[neighbor_cluster_id] = {
                k: v.clone().to(self.device) for k, v in neighbor_params.items()
            }
            self.update_counts[neighbor_cluster_id] = 1
            print(f"[Worker {self.id}] Initialized cache for Cluster {neighbor_cluster_id}")
            return

        # Trường hợp 2: Đã có bản sao, thực hiện công thức trung bình
        r = self.update_counts[neighbor_cluster_id]
        
        # Lấy w_{i,c*} (Mô hình cục bộ hiện tại cho cụm đó)
        local_cache_params = self.cluster_models_cache[neighbor_cluster_id]
        
        # Tính hệ số Alpha và Beta
        # alpha = r / (r + 1)
        # beta  = 1 / (r + 1)
        alpha = r / (r + 1.0)
        beta = 1.0 / (r + 1.0)
        
        updated_params = {}
        
        # Thực hiện cộng gộp từng lớp (Layer-wise aggregation)
        for name in local_cache_params.keys():
            if name in neighbor_params:
                # w_new = alpha * w_old + beta * w_neighbor
                # Đảm bảo tensor nằm trên cùng device (CPU/GPU)
                w_old = local_cache_params[name]
                w_neighbor = neighbor_params[name].to(self.device)
                
                updated_params[name] = (alpha * w_old) + (beta * w_neighbor)
            else:
                # Nếu layer không khớp (hiếm gặp), giữ nguyên cũ
                updated_params[name] = local_cache_params[name]

        # Cập nhật lại bộ nhớ đệm
        self.cluster_models_cache[neighbor_cluster_id] = updated_params
        
        # Tăng biến đếm r
        self.update_counts[neighbor_cluster_id] += 1
        
        # (Tùy chọn) Nếu cụm này trùng với cụm hiện tại của Worker, 
        # cập nhật luôn vào model chính để train tiếp
        if neighbor_cluster_id == self.cluster_id:
            self.model.load_state_dict(updated_params)
            print(f"[Worker {self.id}] Updated MAIN model via Gossip from Cluster {neighbor_cluster_id} (r={r+1})")