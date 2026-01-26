# Xử lý các thuận toán chính: CoCo, BALANCE, Aggregation
import torch
import numpy as np
import math
from app.models.cnn import SimpleCNN
from config import Config
from app.core.ldp_helpers import LDP
from app.core.attacks import AttackFactory
from app.utils.data_loader import get_dataloader, get_dataloader_from_indices
from app.models.cnn import get_model
import torch.nn.functional as F
# Clas này xử lý phân cụm (DFCA), Huấn luyện cục bộ và thêm nhiễu LDP
class WorkerNode:
    def __init__(self, node_id, config, device):
        self.id = node_id
        self.config = config
        self.device = device
        # self.data_loader = data_loader
        self.dataset_name = config.get('dataset', 'cifar10')
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs',5)
        
        # Khởi tạo DataLoader mặc định (IID)
        num_workers = config.get('num_workers', 10)
        self.data_loader, self.num_classes, self.input_channels = get_dataloader(
            self.dataset_name, self.id, num_workers, self.batch_size
        )
        model_name = config.get('model', 'simple_cnn')
        self.model = get_model(model_name, num_classes=self.num_classes).to(self.device)
        # self.model = SimpleCNN(num_classes=self.num_classes).to(self.device)
        self.cluster_id = None  # Sẽ được gán sau bước Clustering
        self.cluster_head_id = None

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

        # Các thuộc tính dùng chung cho DFL (Baseline & Proposed)
        self.neighbors = []         # Danh sách hàng xóm
        self.received_updates = {}  # Bộ đệm chứa model nhận được từ hàng xóm

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

    def train(self):
        """Bước 2: Huấn luyện cục bộ (Local Training)"""
        # self.model.train()
        # # ... logic training loop (SGD) ...
        # print(f"Worker {self.id} finished training.")
        # return self.model.state_dict()
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                   lr=self.config.get('learning_rate', 0.01),
                                   momentum=0.9)
        if self.attack_strategy:
            return self.attack_strategy.execute(self)
        else:
            return self._standard_train()
        # return self.attack_strategy.execute(self)
    
    def _standard_train(self, flip_labels=False):
        self.model.train()
        for epoch in range(self.epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        
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
    
    def apply_new_indices(self, indices, dataset_name):
        """Được gọi bởi Scenario Class để gán dữ liệu mới"""
        self.data_indices = indices
        
        # Hàm này sẽ tạo DataLoader từ danh sách index có sẵn, không cần tính toán lại
        self.data_loader, self.num_classes, _ = get_dataloader_from_indices(
            dataset_name, indices=indices, batch_size=32,train=True
        )
        print(f"[Worker {self.id}] Applied new dataset partition ({len(indices)} samples)")

    @property
    def is_malicious(self):
        # Nếu strategy chưa được khởi tạo (None), coi như lành tính
        from app.core.attacks import HonestStrategy
        return self.attack_strategy is not None and not isinstance(self.attack_strategy, HonestStrategy)
    
    def evaluate_gaussian_metrics(self, test_loader):
        """
        Đánh giá chi tiết: Trả về Accuracy, Loss và MSE.
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total_mse = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward
                logits = self.model(data)
                probs = F.softmax(logits, dim=1) # Chuyển về xác suất [0, 1]
                
                # 1. Tính CrossEntropy Loss (Mặc định)
                test_loss += self.criterion(logits, target).item()
                
                # 2. Tính Accuracy
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                # 3. Tính MSE (Mean Squared Error)
                # Cần one-hot encode target để so sánh với vector xác suất
                target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
                batch_mse = F.mse_loss(probs, target_one_hot, reduction='sum').item()
                total_mse += batch_mse
                
                total_samples += len(data)

        # Tổng hợp kết quả
        avg_loss = test_loss / len(test_loader)
        accuracy = correct / total_samples
        avg_mse = total_mse / total_samples
        
        # Error Rate = 1 - Accuracy
        error_rate = 1.0 - accuracy

        return {
            "accuracy": accuracy,
            "error_rate": error_rate,
            "loss": avg_loss,
            "mse": avg_mse
        }
    
    def evaluate_label_flipping(self, test_loader, src_class, tgt_class):
        """
        Chuyên dùng cho Label Flipping: Tính ASR, Recall, Precision.
        """
        self.model.eval()
        correct = 0
        total = 0
        
        # Biến đếm riêng cho LF
        src_total = 0
        src_correct = 0
        src_flipped_to_tgt = 0 # Tử số của ASR
        
        tgt_pred_total = 0
        tgt_true_positive = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                logits = self.model(data)
                preds = logits.argmax(dim=1)
                
                # Metric cơ bản
                correct += preds.eq(target).sum().item()
                total += len(target)

                # Metric Label Flipping
                # 1. Source Class stats
                src_mask = (target == src_class)
                src_total += src_mask.sum().item()
                src_correct += (preds[src_mask] == src_class).sum().item()
                src_flipped_to_tgt += (preds[src_mask] == tgt_class).sum().item()

                # 2. Target Class stats
                tgt_pred_mask = (preds == tgt_class)
                tgt_pred_total += tgt_pred_mask.sum().item()
                tgt_true_positive += ((preds == tgt_class) & (target == tgt_class)).sum().item()

        return {
            "accuracy": correct / total,
            "error_rate": 1.0 - (correct / total),
            "src_recall": src_correct / (src_total + 1e-9),
            "asr": src_flipped_to_tgt / (src_total + 1e-9),
            "tgt_precision": tgt_true_positive / (tgt_pred_total + 1e-9)
        }
    
    def evaluate_backdoor(self, test_loader, target_class):
        """
        Đánh giá kịch bản Backdoor.
        Trả về:
        - Clean Accuracy (BA): Độ chính xác trên dữ liệu sạch.
        - Attack Success Rate (ASR): Tỷ lệ dữ liệu có trigger bị nhận diện thành target_class.
        """
        self.model.eval()
        
        # Metrics cho dữ liệu SẠCH (Main Task)
        clean_correct = 0
        clean_total = 0
        clean_loss = 0
        
        # Metrics cho dữ liệu BACKDOOR
        bd_success = 0 # Số mẫu có trigger bị đoán thành target_class
        bd_total = 0   # Tổng số mẫu dùng để test backdoor
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # --- 1. ĐÁNH GIÁ TRÊN CLEAN DATA ---
                logits = self.model(data)
                clean_loss += self.criterion(logits, target).item()
                preds = logits.argmax(dim=1)
                
                clean_correct += preds.eq(target).sum().item()
                clean_total += len(target)
                
                # --- 2. ĐÁNH GIÁ TRÊN TRIGGERED DATA (ASR) ---
                # Chỉ đánh giá ASR trên các mẫu KHÔNG thuộc target_class
                # (Vì nếu ảnh gốc đã là target_class thì đoán đúng không tính là tấn công thành công)
                non_target_indices = (target != target_class)
                
                if non_target_indices.sum().item() > 0:
                    # Lấy ra các ảnh không phải target
                    data_bd = data[non_target_indices].clone()
                    
                    # Gắn Trigger vào các ảnh này
                    data_bd = self._add_trigger(data_bd)
                    
                    # Dự đoán trên ảnh đã gắn trigger
                    logits_bd = self.model(data_bd)
                    preds_bd = logits_bd.argmax(dim=1)
                    
                    # ASR: Bao nhiêu ảnh đã bị lái sang target_class?
                    bd_success += (preds_bd == target_class).sum().item()
                    bd_total += len(data_bd)

        return {
            "accuracy": clean_correct / clean_total, # Main Task Accuracy
            "error_rate": 1.0 - (clean_correct / clean_total),
            "loss": clean_loss / len(test_loader),
            "asr": bd_success / (bd_total + 1e-9)    # Backdoor ASR
        }

    def _add_trigger(self, images):
        """
        Hàm helper để gắn trigger lên ảnh test.
        Logic này PHẢI KHỚP với logic tấn công (ví dụ: pattern 3x3 pixel góc phải).
        """
        # Giả sử trigger là một ô vuông trắng 3x3 ở góc dưới bên phải
        # images shape: [Batch, Channel, Height, Width]
        from app.utils.trigger import add_pixel_pattern
        patch_images = images.clone()
        return add_pixel_pattern(patch_images,self.device)
    
    def set_neighbors(self, neighbor_indices):
        """
        Thiết lập danh sách láng giềng cho node này (Topology P2P).
        Args:
            neighbor_indices (list[int]): Danh sách ID của các node hàng xóm.
        """
        # Loại bỏ chính mình ra khỏi danh sách (đề phòng)
        self.neighbors = [nid for nid in neighbor_indices if nid != self.node_id]
        
        # Log kiểm tra (nếu cần)
        # print(f"Node {self.node_id} connected to: {self.neighbors}")

    def get_neighbors(self):
        """Trả về danh sách láng giềng"""
        return self.neighbors