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
from app.core.ipfs import StorageService
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
        self.epoch_loss = 0.0
        self.local_test_loader = None
        
        # Khởi tạo DataLoader mặc định (IID)
        num_workers = config.get('num_workers', 10)
        self.data_loader, self.num_classes, self.input_channels = get_dataloader(
            self.dataset_name, self.id, num_workers, self.batch_size
        )
        model_name = config.get('model', 'simple_cnn')
        # 🔑 CHANGE: Initialize model on device directly (not CPU first)
        # This reduces CPU→GPU transfers
        self.model = get_model(model_name, num_classes=self.num_classes).to(self.device)

        self.cluster_id = None  # Sẽ được gán sau bước Clustering
        self.cluster_head_id = None

        self.attack_strategy = AttackFactory.get_strategy("NONE")

        # Khởi tạo Optimizer & Loss
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=Config.LEARNING_RATE, 
            momentum=Config.MOMENTUM,
            weight_decay=Config.WEIGHT_DECAY
        )
        # Các tham số sử dụng cho giải thuật DFCA
        self.cluster_models_cache = {}
        self.update_counts = {} # số lượng cập nhật đã nhận cho mỗi cụm

        # Các thuộc tính dùng chung cho DFL (Baseline & Proposed)
        self.neighbors = []         # Danh sách hàng xóm
        self.received_updates = {}  # Bộ đệm chứa model nhận được từ hàng xóm

        # Kết nối tới IPFS để lấy k-model
        self.storage = StorageService()
    
    def reset_state(self):
        self.update_counts = {} # Reset bộ đếm gossip
        self.received_updates = {}
        self.cluster_models_cache = {}

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

    def fetch_latest_models(self, blockchain):
        """
        Hàm chạy đầu vòng mới:
        1. Gọi Smart Contract lấy Hashes.
        2. Tải Model từ IPFS.
        """
        print(f"[Worker {self.id}] Fetching latest K-Models...")
        
        # 1. Gọi Smart Contract (View Function - Miễn phí)
        cids_map = blockchain.get_latest_k_model_hashes()
        
        if not cids_map:
            print(" -> Registry empty. Waiting for initialization...")
            return {}

        downloaded_models = {}
        
        # 2. Tải dữ liệu từ Storage (Off-chain)
        for cid_id, ipfs_hash in cids_map.items():
            # Kiểm tra cache xem đã tải chưa (trong thực tế)
            model_state = self.storage.download_model(ipfs_hash)
            
            if model_state:
                downloaded_models[cid_id] = model_state
                total_norm = 0.0
                for p in model_state.values():
                    if p.dtype == torch.float32:
                        total_norm += p.norm(2).item()
                
                print(f" -> Downloaded Model Hash {ipfs_hash[:6]}. Weight Norm: {total_norm:.4f}")
                
                if total_norm > 1000:
                    print(" [CẢNH BÁO] Model này đã bị NỔ (Weights quá lớn)!")
            else:
                print(f" -> Failed to download Cluster {cid_id} (Hash: {ipfs_hash})")
        
        print(f" -> Successfully fetched {len(downloaded_models)} models.")
        return downloaded_models

    def evaluate_loss_on_model(self, model_state):
        """
        Tính Loss để chọn cụm (DFCA). ✅ OPTIMIZED: Keep model on GPU
        """
        self.model.load_state_dict(model_state)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        loss_fn = torch.nn.CrossEntropyLoss()
        
        try:
            with torch.no_grad():
                for X, y in self.data_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    preds = self.model(X)
                    loss = loss_fn(preds, y)
                    total_loss += loss.item() * X.size(0)
                    total_samples += X.size(0)
            
            return total_loss / total_samples if total_samples > 0 else float('inf')
        finally:
            # 🔑 CHANGE: DON'T move to CPU - keep model on GPU
            # This allows subsequent operations to use GPU directly
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def join_cluster(self, cluster_models):
        """Worker tự chọn cụm có Loss thấp nhất"""
        losses = {cid: self.evaluate_loss_on_model(model) for cid, model in cluster_models.items()}
        best_cid = min(losses, key=losses.get)
        min_loss = losses[best_cid]

        self.cluster_id = best_cid
        self.current_loss = min_loss
        print(f"Worker {self.id} joined Cluster {self.cluster_id} with loss {losses[self.cluster_id]:.4f}")

    def join_cluster_via_blockchain(self, blockchain):
        # Client tự tải model về
        cluster_models = self.fetch_latest_models(blockchain=blockchain)
        if not cluster_models:
            print("Worker cannot join cluster: No models avalable")
        # Tính loss và chọn cụm
        losses = {cid: self.evaluate_loss_on_model(model) for cid, model in cluster_models.items()}
        best_cid = min(losses, key=losses.get)
        min_loss = losses[best_cid]

        self.cluster_id = best_cid
        self.current_loss = min_loss
        self.model.load_state_dict(cluster_models[best_cid])
        print(f"Worker {self.id} joined Cluster {self.cluster_id} with loss {losses[self.cluster_id]:.4f}")

    # def train(self):
    #     """Bước 2: Huấn luyện cục bộ (Local Training)"""
    #     # self.model.train()
    #     # # ... logic training loop (SGD) ...
    #     self.optimizer = torch.optim.SGD(self.model.parameters(), 
    #                                lr=self.config.get('learning_rate', 0.01),
    #                                momentum=0.9)
    #     if self.attack_strategy.is_malicious:
    #         weights = self.attack_strategy.execute(self)
    #         return {
    #             "weights": weights,
    #             "loss": None,
    #         }
    #     else:
    #         weights, loss = self._standard_train()
    #         return {
    #             "weights": weights,
    #             "loss": loss
    #         }
        # return self.attack_strategy.execute(self)
    # def apply_ldp_sparse(self, delta, sparsity=0.01): # sparsity=0.01 nghĩa là giữ 1%
    #     """
    #     LDP dạng thưa (Sparse LDP):
    #     Chỉ giữ lại Top-k tham số thay đổi nhiều nhất và cộng nhiễu vào đó.
    #     Giúp giảm Norm của nhiễu đi căn bậc 2 của (1/sparsity) lần.
    #     """
    #     if not Config.ENABLE_LDP:
    #         return delta

    #     noisy_delta = {}
        
    #     # 1. Tính toán ngưỡng Top-k
    #     all_values = []
    #     for v in delta.values():
    #         if v.dtype in [torch.float, torch.float32]:
    #             all_values.append(v.view(-1).abs())
        
    #     if not all_values: return delta
        
    #     full_vec = torch.cat(all_values)
    #     total_params = full_vec.numel()
    #     k = int(total_params * sparsity)
    #     if k < 1: k = 1
        
    #     # Tìm giá trị ngưỡng (threshold) để lọt vào top k
    #     top_k_threshold = torch.kthvalue(full_vec, total_params - k + 1).values.item()

    #     # 2. Lấy Sigma và Clipping Threshold
    #     # Cần config C rất nhỏ cho update (ví dụ 0.05 hoặc 0.1)
    #     clip_threshold = Config.LDP_CLIPPING_THRESHOLD 
    #     sigma = LDP.get_ldp_sigma()
    #     print(f"SIgma applied in LDP for model: {sigma}")
        
    #     # 3. Áp dụng Mask & Noise
    #     for k, v in delta.items():
    #         if v.dtype not in [torch.float, torch.float32]:
    #             noisy_delta[k] = v
    #             continue

    #         # Mask: 1 nếu thuộc Top-k, 0 nếu không
    #         mask = (v.abs() >= top_k_threshold).float()
            
    #         # Clipping: Cắt update trong khoảng [-C, C]
    #         # scale = max(1, norm/C) -> Ở đây làm đơn giản là clamp từng giá trị
    #         clipped_v = torch.clamp(v, -clip_threshold, clip_threshold)
            
    #         # Tạo nhiễu (chỉ cộng vào những nơi mask = 1)
    #         noise = torch.normal(0, sigma, v.shape, device=self.device)
            
    #         # Update mới = (Clipped_Update + Noise) * Mask
    #         # Những chỗ không quan trọng sẽ biến thành 0 (Sparsification)
    #         noisy_delta[k] = (clipped_v + noise) * mask
            
    #     return noisy_delta
    def apply_ldp_sparse(self, delta, sparsity=0.01):
        """
        LDP dạng thưa (Sparse LDP) - Phiên bản Hợp nhất (Unified)
        Hoạt động an toàn cho cả SimpleCNN (không BN) và VGG/ResNet (có BN).
        """
        if not Config.ENABLE_LDP:
            return delta

        noisy_delta = {}
        trainable_keys = []
        
        # 1. BỘ LỌC THÔNG MINH (Bỏ qua BatchNorm stats & non-floats)
        for k, v in delta.items():
            # Nếu là SimpleCNN, điều kiện 'running' sẽ false -> lọt vào else an toàn
            if 'running' in k or 'num_batches' in k or v.dtype not in [torch.float, torch.float32]:
                noisy_delta[k] = v  # Giữ nguyên, không cộng nhiễu
            else:
                trainable_keys.append(k)

        # 2. Tính toán ngưỡng Top-k
        all_values = [delta[k].view(-1).abs() for k in trainable_keys]
        if not all_values: 
            return noisy_delta
        
        full_vec = torch.cat(all_values)
        total_params = full_vec.numel()
        k = max(1, int(total_params * sparsity))
        
        # Tìm giá trị ngưỡng (threshold)
        top_k_threshold = torch.kthvalue(full_vec, total_params - k + 1).values.item()

        # 3. Lấy tham số LDP
        clip_threshold = Config.LDP_CLIPPING_THRESHOLD 
        sigma = LDP.get_ldp_sigma()
        print(f"Sigma applied in LDP for model: {sigma}")
        
        # 4. Tính L2 Norm của các phần tử Top-k (Chuẩn DP-SGD)
        sparse_norm_sq = 0.0
        for key in trainable_keys:
            v = delta[key]
            mask = (v.abs() >= top_k_threshold).float()
            sparse_norm_sq += torch.sum((v * mask) ** 2).item()
            
        sparse_norm = (sparse_norm_sq ** 0.5)
        # Hệ số thu nhỏ (Chỉ thu nhỏ nếu norm > ngưỡng)
        clip_factor = max(1.0, sparse_norm / clip_threshold)

        # 5. Áp dụng Clipping, Nhiễu & Mask
        for key in trainable_keys:
            v = delta[key]
            
            # Mask: 1 nếu thuộc Top-k, 0 nếu không
            mask = (v.abs() >= top_k_threshold).float()
            
            # Clipping toàn cục (Giữ nguyên hướng của Vector Update tốt hơn torch.clamp)
            clipped_v = v / clip_factor
            
            # Tạo nhiễu (Dùng v.device để không bị lỗi xung đột CPU/GPU)
            noise = torch.normal(0, sigma, v.shape, device=v.device)
            
            # Update mới = (Clipped + Noise) * Mask
            noisy_delta[key] = (clipped_v + noise) * mask
            
        return noisy_delta
    def train(self):
        """
        Huấn luyện cục bộ có tích hợp LDP trên Update (Delta).
        Quy trình:
        1. Lưu W_old.
        2. Train/Attack -> ra W_new.
        3. Tính Delta = W_new - W_old.
        4. Apply LDP lên Delta.
        5. Trả về W_final = W_old + Noisy_Delta.
        
        ✅ OPTIMIZED: Keep model on GPU throughout, LDP works on GPU tensors
        """
        # BƯỚC 1: Đảm bảo model ở GPU, snapshot W_old
        self.model = self.model.to(self.device)
        w_old = {k: v.clone().detach() for k, v in self.model.state_dict().items()}
        
        weights_new = None
        train_loss = None
        
        # Khởi tạo lại Optimizer ở mỗi vòng để xóa sạch Momentum cũ
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=Config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=1e-4 
        )
        
        # BƯỚC 2: Thực hiện Training hoặc Tấn công
        try:
            if self.attack_strategy.is_malicious:
                weights_new = self.attack_strategy.execute(self)
                train_loss = None
            else:
                weights_new, train_loss = self._standard_train()

            # BƯỚC 3 & 4: Tính Delta trên GPU và Áp dụng LDP
            if Config.ENABLE_LDP and self.config.get('system_mode') == 'PROPOSED':
                # a. Tính Delta - KEEP ON GPU
                delta = {}
                final_weights = {}
                untouched_params = {}
                
                for k in weights_new.keys():
                    if k in w_old and weights_new[k].dtype in [torch.float, torch.float32]:
                        # Keep on GPU for LDP processing
                        delta[k] = weights_new[k].to(self.device) - w_old[k].to(self.device)
                    else:
                        untouched_params[k] = weights_new[k].to(self.device)

                # b. Áp dụng LDP lên GPU tensors (apply_ldp_sparse đã support device)
                noisy_delta = self.apply_ldp_sparse(delta, sparsity=0.2)

                # c. Tái tạo Model Weights trên GPU
                for k in weights_new.keys():
                    if k in noisy_delta and k in w_old and weights_new[k].dtype in [torch.float, torch.float32]:
                        final_weights[k] = (w_old[k].to(self.device) + noisy_delta[k]).cpu()
                    elif k in untouched_params:
                        final_weights[k] = untouched_params[k].cpu()
                    else:
                        final_weights[k] = weights_new[k].cpu()
            else:
                # Nếu không dùng LDP thì trả về nguyên gốc (convert to CPU for storage)
                final_weights = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                for k, v in weights_new.items()}

            # Cập nhật model và GIỮ TRÊN GPU cho phase tiếp theo
            self.model.load_state_dict(final_weights)
            self.model = self.model.to(self.device)

            return {
                "weights": final_weights,
                "loss": train_loss
            }
        except Exception as e:
            print(f"[Worker {self.id}] Error in train(): {e}")
            raise
        finally:
            # 🔑 CHANGE: DON'T move model to CPU - keep on device for aggregation
            # Only clear cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _standard_train(self):
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        _cal_epoch_loss = 0.0 # Giá trị trung bình của hàm mất mát sau mõi vòng
        for epoch in range(self.epochs):
            batch_losses = []
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                # Cắt gọn gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()
                batch_losses.append(loss.item())
            _cal_epoch_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
            # print(f"Worker {self.id} - Epoch {epoch+1}/{self.epochs} - -WTF: {_cal_epoch_loss:.4f}")
        
        self.epoch_loss = _cal_epoch_loss
        return self.model.state_dict(), _cal_epoch_loss

    def set_epoch_loss(self, epoch_loss):
        self.epoch_loss = epoch_loss
    
    def get_epoch_loss(self):
        return self.epoch_loss
        # --- BƯỚC B: CỘNG NHIỄU GAUSS - Công thức (30) ---
        
    # Local update
    # def train(self, local_epochs=1):
    #     """ Huấn luyện cục bộ bằng SGD
    #     """
    #     self.model.train()

    #     for epoch in range(local_epochs):
    #         for X, y in self.data_loader:
    #             X, y = X.to(self.device), y.to(self.device)

    #             self.optimizer.zero_grad()
    #             preds = self.model(X)
    #             loss = self.loss_fn(preds, y)
    #             loss.backward()
    #             self.optimizer.step()

    #     print(f"[Worker {self.id}] finished local training.")
    #     return self.model.state_dict()

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
        sigma = LDP.get_ldp_sigma() / math.sqrt(self.config.get('num_workers')) # Độ lệch chuẩn của nhiễu
        print(f"Sigma applied in LDP for model is: {sigma}")
        
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
    
    def apply_new_indices(self, train_indices, test_indices, dataset_name):
        """Được gọi bởi Scenario Class để gán dữ liệu mới"""
        # self.data_indices = indices
        
        # Tạo dataloader cho trrain
        self.data_loader, self.num_classes, _ = get_dataloader_from_indices(
            dataset_name, indices=train_indices, batch_size=32,train=True
        )

        # Tạo dataloader cho Local Test
        self.local_test_loader, _, _ = get_dataloader_from_indices(
            dataset_name=dataset_name, indices=test_indices, batch_size=32, train=False
        )
        print(f"[Worker {self.id}] Applied new dataset partition ({len(train_indices)} samples)")

    @property
    def is_malicious(self):
        # Nếu strategy chưa được khởi tạo (None), coi như lành tính
        from app.core.attacks import HonestStrategy
        return self.attack_strategy is not None and not isinstance(self.attack_strategy, HonestStrategy)
    
    #------------------------------------Thử nghiệm hàm evaluate mơi--------------------------------------------------
    # def evaluate_gaussian_metrics(self, test_loader):
    #     """
    #     Đánh giá chi tiết: Trả về Accuracy, Loss và MSE.
    #     """
    #     for param in self.model.parameters():
    #         if torch.isnan(param).any() or torch.isinf(param).any():
    #             print(f"[Worker {self.id}] Model params are NaN/Inf! Returning Worst Metrics.")
    #             return {
    #                 "accuracy": 0.0,
    #                 "error_rate": 1.0,
    #                 "loss": 10000.0, # Giá trị Loss cực lớn tượng trưng
    #                 "mse": 1.0       # MSE max (giữa 0 và 1) thường là 1 hoặc 2
    #             }
    #     self.model.to(self.device)
    #     self.model.eval()
    #     test_loss = 0
    #     correct = 0
    #     total_mse = 0.0
    #     total_samples = 0
    #     try:
    #         with torch.no_grad():
    #             for data, target in test_loader:
    #                 data, target = data.to(self.device), target.to(self.device)
                    
    #                 # Forward
    #                 logits = self.model(data)
    #                 probs = F.softmax(logits, dim=1) # Chuyển về xác suất [0, 1]
                    
    #                 # 1. Tính CrossEntropy Loss (Mặc định)
    #                 test_loss += self.criterion(logits, target).item()
                    
    #                 # 2. Tính Accuracy
    #                 pred = logits.argmax(dim=1, keepdim=True)
    #                 correct += pred.eq(target.view_as(pred)).sum().item()
                    
    #                 # 3. Tính MSE (Mean Squared Error)
    #                 # Cần one-hot encode target để so sánh với vector xác suất
    #                 target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
    #                 batch_mse = F.mse_loss(probs, target_one_hot, reduction='sum').item()
    #                 total_mse += batch_mse
                    
    #                 total_samples += len(data)

    #         # Tổng hợp kết quả
    #         avg_loss = test_loss / len(test_loader)
    #         accuracy = correct / total_samples
    #         avg_mse = total_mse / total_samples
            
    #         # Error Rate = 1 - Accuracy
    #         error_rate = 1.0 - accuracy

    #         return {
    #             "accuracy": accuracy,
    #             "error_rate": error_rate,
    #             "loss": avg_loss,
    #             "mse": avg_mse
    #         }
    #     finally:
    #         self.model.to('cpu')
    
    # def evaluate_label_flipping(self, test_loader, src_class, tgt_class):
    #     """
    #     Chuyên dùng cho Label Flipping: Tính ASR, Recall, Precision.
    #     """
    #     self.model.eval()
    #     self.model.to(self.device)
    #     correct = 0
    #     total = 0
        
    #     # Biến đếm riêng cho LF
    #     src_total = 0
    #     src_correct = 0
    #     src_flipped_to_tgt = 0 # Tử số của ASR
        
    #     tgt_pred_total = 0
    #     tgt_true_positive = 0
    #     try:
    #         with torch.no_grad():
    #             for data, target in test_loader:
    #                 data, target = data.to(self.device), target.to(self.device)
    #                 logits = self.model(data)
    #                 preds = logits.argmax(dim=1)
                    
    #                 # Metric cơ bản
    #                 correct += preds.eq(target).sum().item()
    #                 total += len(target)

    #                 # Metric Label Flipping
    #                 # 1. Source Class stats
    #                 src_mask = (target == src_class)
    #                 src_total += src_mask.sum().item()
    #                 src_correct += (preds[src_mask] == src_class).sum().item()
    #                 src_flipped_to_tgt += (preds[src_mask] == tgt_class).sum().item()

    #                 # 2. Target Class stats
    #                 tgt_pred_mask = (preds == tgt_class)
    #                 tgt_pred_total += tgt_pred_mask.sum().item()
    #                 tgt_true_positive += ((preds == tgt_class) & (target == tgt_class)).sum().item()

    #         return {
    #             "accuracy": correct / total,
    #             "error_rate": 1.0 - (correct / total),
    #             "src_recall": src_correct / (src_total + 1e-9),
    #             "asr": src_flipped_to_tgt / (src_total + 1e-9),
    #             "tgt_precision": tgt_true_positive / (tgt_pred_total + 1e-9)
    #         }
    #     finally:
    #         self.model.to('cpu')
    
    # def evaluate_backdoor(self, test_loader, target_class):
    #     """
    #     Đánh giá kịch bản Backdoor.
    #     Trả về:
    #     - Clean Accuracy (BA): Độ chính xác trên dữ liệu sạch.
    #     - Attack Success Rate (ASR): Tỷ lệ dữ liệu có trigger bị nhận diện thành target_class.
    #     """
    #     for param in self.model.parameters():
    #         if torch.isnan(param).any() or torch.isinf(param).any():
    #             print(f"[Worker {self.id}] Backdoor Check: Model params NaN/Inf -> Return Worst.")
    #             return {
    #                 "accuracy": 0.0,
    #                 "error_rate": 1.0,
    #                 "loss": 10000.0,
    #                 "asr": 1.0 # Model hỏng thì ASR coi như 0 (hoặc 1 tuỳ định nghĩa)
    #             }
    #     self.model.to(self.device)
    #     self.model.eval()
        
    #     # Metrics cho dữ liệu SẠCH (Main Task)
    #     clean_correct = 0
    #     clean_total = 0
    #     clean_loss = 0
        
    #     # Metrics cho dữ liệu BACKDOOR
    #     bd_success = 0 # Số mẫu có trigger bị đoán thành target_class
    #     bd_total = 0   # Tổng số mẫu dùng để test backdoor
    #     try:
    #         with torch.no_grad():
    #             for data, target in test_loader:
    #                 data, target = data.to(self.device), target.to(self.device)
                    
    #                 # --- 1. ĐÁNH GIÁ TRÊN CLEAN DATA ---
    #                 logits = self.model(data)
    #                 clean_loss += self.criterion(logits, target).item()
    #                 preds = logits.argmax(dim=1)
                    
    #                 clean_correct += preds.eq(target).sum().item()
    #                 clean_total += len(target)
                    
    #                 # --- 2. ĐÁNH GIÁ TRÊN TRIGGERED DATA (ASR) ---
    #                 # Chỉ đánh giá ASR trên các mẫu KHÔNG thuộc target_class
    #                 # (Vì nếu ảnh gốc đã là target_class thì đoán đúng không tính là tấn công thành công)
    #                 non_target_indices = (target != target_class)
                    
    #                 if non_target_indices.sum().item() > 0:
    #                     # Lấy ra các ảnh không phải target
    #                     data_bd = data[non_target_indices].clone()
                        
    #                     # Gắn Trigger vào các ảnh này
    #                     data_bd = self._add_trigger(data_bd)
                        
    #                     # Dự đoán trên ảnh đã gắn trigger
    #                     logits_bd = self.model(data_bd)
    #                     preds_bd = logits_bd.argmax(dim=1)
                        
    #                     # ASR: Bao nhiêu ảnh đã bị lái sang target_class?
    #                     bd_success += (preds_bd == target_class).sum().item()
    #                     bd_total += len(data_bd)

    #         return {
    #             "accuracy": clean_correct / clean_total, # Main Task Accuracy
    #             "error_rate": 1.0 - (clean_correct / clean_total),
    #             "loss": clean_loss / len(test_loader),
    #             "asr": bd_success / (bd_total + 1e-9)    # Backdoor ASR
    #         }
    #     finally:
    #         self.model.to('cpu')

    def evaluate_backdoor(self, target_class, k_models_dict=None, custom_test_loader=None):
        """
        Đánh giá kịch bản Backdoor. Hỗ trợ cả Baseline và Proposed.
        """
        loader = custom_test_loader if custom_test_loader is not None else getattr(self, 'local_test_loader', None)
        if loader is None:
            return {"accuracy": 0.0, "error_rate": 1.0, "loss": 10000.0, "asr": 1.0}

        original_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        self.model.to(self.device)

        try:
            # --- TÌM MODEL TỐT NHẤT DỰA TRÊN DỮ LIỆU SẠCH ---
            if k_models_dict:
                best_loss = float('inf')
                best_state = None
                self.model.eval()
                with torch.no_grad():
                    for cid, model_state in k_models_dict.items():
                        self.model.load_state_dict(model_state)
                        client_loss = 0.0
                        total = 0
                        # Worker chỉ đánh giá loss sạch (Nó không biết về backdoor)
                        for data, target in loader:
                            data, target = data.to(self.device), target.to(self.device)
                            outputs = self.model(data)
                            client_loss += self.criterion(outputs, target).item() * data.size(0)
                            total += data.size(0)
                        
                        avg_loss = client_loss / total if total > 0 else float('inf')
                        if avg_loss < best_loss:
                            best_loss = avg_loss
                            best_state = model_state
                
                if best_state:
                    self.model.load_state_dict(best_state)

            # --- KIỂM TRA NAN/INF ---
            for param in self.model.parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    return {"accuracy": 0.0, "error_rate": 1.0, "loss": 10000.0, "asr": 1.0}

            # --- BẮT ĐẦU ĐÁNH GIÁ (CLEAN & ASR) ---
            self.model.eval()
            clean_correct, clean_total, clean_loss = 0, 0, 0.0
            bd_success, bd_total = 0, 0
            
            with torch.no_grad():
                for data, target in loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # 1. Đánh giá Clean Data
                    logits = self.model(data)
                    clean_loss += self.criterion(logits, target).item()
                    preds = logits.argmax(dim=1)
                    clean_correct += preds.eq(target).sum().item()
                    clean_total += len(target)
                    
                    # 2. Đánh giá Backdoor (ASR)
                    non_target_indices = (target != target_class)
                    if non_target_indices.sum().item() > 0:
                        data_bd = data[non_target_indices].clone()
                        data_bd = self._add_trigger(data_bd) # Gọi hàm gắn trigger của bạn
                        
                        logits_bd = self.model(data_bd)
                        preds_bd = logits_bd.argmax(dim=1)
                        
                        bd_success += (preds_bd == target_class).sum().item()
                        bd_total += len(data_bd)

            acc = clean_correct / clean_total if clean_total > 0 else 0.0
            return {
                "accuracy": acc,
                "error_rate": 1.0 - acc,
                "loss": clean_loss / len(loader),
                "asr": bd_success / (bd_total + 1e-9)
            }
        finally:
            self.model.load_state_dict(original_state)
            self.model.to('cpu')
    
    def evaluate_gaussian_metrics(self, k_models_dict=None, custom_test_loader=None):
        """
        Đánh giá Gaussian Metrics. Hỗ trợ cả Baseline và Proposed (k-models).
        """
        loader = custom_test_loader if custom_test_loader is not None else getattr(self, 'local_test_loader', None)
        if loader is None:
            return {"accuracy": 0.0, "error_rate": 1.0, "loss": 10000.0, "mse": 1.0}

        original_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        self.model.to(self.device)

        try:
            # --- TÌM MODEL TỐT NHẤT TRONG K-MODELS (Nếu là luồng Proposed) ---
            if k_models_dict:
                best_loss = float('inf')
                best_state = None
                self.model.eval()
                with torch.no_grad():
                    for cid, model_state in k_models_dict.items():
                        self.model.load_state_dict(model_state)
                        client_loss = 0.0
                        total = 0
                        for data, target in loader:
                            data, target = data.to(self.device), target.to(self.device)
                            outputs = self.model(data)
                            client_loss += self.criterion(outputs, target).item() * data.size(0)
                            total += data.size(0)
                        
                        avg_loss = client_loss / total if total > 0 else float('inf')
                        if avg_loss < best_loss:
                            best_loss = avg_loss
                            best_state = model_state
                
                if best_state:
                    self.model.load_state_dict(best_state) # Load model thắng cuộc

            # --- KIỂM TRA NAN/INF ---
            for param in self.model.parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    return {"accuracy": 0.0, "error_rate": 1.0, "loss": 10000.0, "mse": 1.0}

            # --- BẮT ĐẦU ĐÁNH GIÁ METRICS ---
            self.model.eval()
            test_loss, correct, total_mse, total_samples = 0.0, 0, 0.0, 0
            
            with torch.no_grad():
                for data, target in loader:
                    data, target = data.to(self.device), target.to(self.device)
                    logits = self.model(data)
                    probs = F.softmax(logits, dim=1)
                    
                    test_loss += self.criterion(logits, target).item()
                    
                    pred = logits.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    
                    target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
                    total_mse += F.mse_loss(probs, target_one_hot, reduction='sum').item()
                    
                    total_samples += len(data)

            avg_loss = test_loss / len(loader)
            accuracy = correct / total_samples if total_samples > 0 else 0.0
            avg_mse = total_mse / total_samples if total_samples > 0 else 1.0

            return {
                "accuracy": accuracy,
                "error_rate": 1.0 - accuracy,
                "loss": avg_loss,
                "mse": avg_mse
            }
        finally:
            self.model.load_state_dict(original_state) # Phục hồi trọng số gốc
            self.model.to('cpu')

    def evaluate_label_flipping(self, src_class, tgt_class, k_models_dict=None, custom_test_loader=None):
        """
        Đánh giá Label Flipping (ASR, Recall, Precision).
        """
        loader = custom_test_loader if custom_test_loader is not None else getattr(self, 'local_test_loader', None)
        if loader is None:
            return {"accuracy": 0.0, "error_rate": 1.0, "src_recall": 0.0, "asr": 1.0, "tgt_precision": 0.0}

        original_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        self.model.to(self.device)

        try:
            # --- TÌM MODEL TỐT NHẤT TRONG K-MODELS ---
            if k_models_dict:
                best_loss = float('inf')
                best_state = None
                self.model.eval()
                with torch.no_grad():
                    for cid, model_state in k_models_dict.items():
                        self.model.load_state_dict(model_state)
                        client_loss = 0.0
                        total = 0
                        for data, target in loader:
                            data, target = data.to(self.device), target.to(self.device)
                            outputs = self.model(data)
                            client_loss += self.criterion(outputs, target).item() * data.size(0)
                            total += data.size(0)
                        
                        avg_loss = client_loss / total if total > 0 else float('inf')
                        if avg_loss < best_loss:
                            best_loss = avg_loss
                            best_state = model_state
                
                if best_state:
                    self.model.load_state_dict(best_state)

            # --- BẮT ĐẦU ĐÁNH GIÁ ---
            self.model.eval()
            correct, total = 0, 0
            src_total, src_correct, src_flipped_to_tgt = 0, 0, 0
            tgt_pred_total, tgt_true_positive = 0, 0
            
            with torch.no_grad():
                for data, target in loader:
                    data, target = data.to(self.device), target.to(self.device)
                    logits = self.model(data)
                    preds = logits.argmax(dim=1)
                    
                    correct += preds.eq(target).sum().item()
                    total += len(target)

                    src_mask = (target == src_class)
                    src_total += src_mask.sum().item()
                    src_correct += (preds[src_mask] == src_class).sum().item()
                    src_flipped_to_tgt += (preds[src_mask] == tgt_class).sum().item()

                    tgt_pred_mask = (preds == tgt_class)
                    tgt_pred_total += tgt_pred_mask.sum().item()
                    tgt_true_positive += ((preds == tgt_class) & (target == tgt_class)).sum().item()

            acc = correct / total if total > 0 else 0.0
            return {
                "accuracy": acc,
                "error_rate": 1.0 - acc,
                "src_recall": src_correct / (src_total + 1e-9),
                "asr": src_flipped_to_tgt / (src_total + 1e-9),
                "tgt_precision": tgt_true_positive / (tgt_pred_total + 1e-9)
            }
        finally:
            self.model.load_state_dict(original_state)
            self.model.to('cpu')
    
    def evaluate_privacy(self):
        """
        Đánh giá rủi ro quyền riêng tư dựa trên chiến lược tấn công hiện tại.
        Hỗ trợ cả MIA và GIA.
        """
        # Nếu node này không phải malicious hoặc không có strategy -> An toàn (giả định)
        if not hasattr(self, 'attack_strategy') or self.attack_strategy is None:            
            return {}

        # Gọi hàm evaluate của strategy tương ứng
        if hasattr(self.attack_strategy, 'evaluate'):
            return self.attack_strategy.evaluate(self)
        
        return {}
    # -----------------------------------------------------------------------------------
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
    
    def evaluate_detailed(self, test_loader):
        """
        Hàm đánh giá toàn diện dùng cho Engine tính Robustness Metrics.
        Input: test_loader (Global Test Set)
        Output: Dict {accuracy, error_rate, loss, mse}
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total_mse = 0.0
        total_samples = 0
        
        # Dùng reduction='sum' để cộng dồn chính xác theo số sample
        criterion = torch.nn.CrossEntropyLoss(reduction='sum') 
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 1. Forward Pass
                logits = self.model(data)
                probs = F.softmax(logits, dim=1) # Chuyển logit sang xác suất [0,1]
                
                # 2. Tính Loss (CrossEntropy)
                loss = criterion(logits, target)
                test_loss += loss.item()
                
                # 3. Tính Accuracy
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                # 4. Tính MSE (Mean Squared Error) giữa vector dự đoán và one-hot label
                # Cần thiết để đánh giá mức độ nhiễu/poisoning
                target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
                batch_mse = F.mse_loss(probs, target_one_hot, reduction='sum').item()
                total_mse += batch_mse
                
                total_samples += len(data)

        if total_samples == 0:
            return {"accuracy": 0.0, "error_rate": 0.0, "loss": 0.0, "mse": 0.0}

        # Tính trung bình
        avg_loss = test_loss / total_samples
        accuracy = correct / total_samples
        avg_mse = total_mse / total_samples
        error_rate = 1.0 - accuracy

        return {
            "accuracy": accuracy,
            "error_rate": error_rate,
            "loss": avg_loss,
            "mse": avg_mse
        }
    def evaluate_standard(self, test_loader):
        """
        Đánh giá hiệu năng chuẩn (Clean Metrics) cho kịch bản bình thường.
        Chỉ tính toán Accuracy và Loss để tối ưu tốc độ.
        """
        # Kiểm tra NaN/Inf trước khi đo (đề phòng model nổ)
        for param in self.model.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"[Worker {self.id}] Model params are NaN/Inf! Returning Worst Metrics.")
                return {"accuracy": 0.0, "error_rate": 1.0, "loss": 10000.0}

        self.model.to(self.device)
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total_samples = 0
        try:
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Forward
                    logits = self.model(data)
                    
                    # 1. Tính Accuracy
                    pred = logits.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    
                    # 2. Tính Loss (CrossEntropy)
                    # Kỹ thuật chuẩn: Nhân loss của batch với số lượng sample trong batch đó
                    loss = self.criterion(logits, target)
                    test_loss += loss.item() * data.size(0) 
                    
                    total_samples += data.size(0)

            # Tổng hợp kết quả
            avg_loss = test_loss / total_samples if total_samples > 0 else 0.0
            accuracy = correct / total_samples if total_samples > 0 else 0.0
            
            return {
                "accuracy": accuracy,
                "error_rate": 1.0 - accuracy,
                "loss": avg_loss
            }
        finally:
            self.model.to('cpu')
    
    # Tính accuracy và max.ter mới
    def evaluate_k_models(self, k_models_dict):
        """
        Worker tự thử nghiệm k mô hình (từ Ủy ban) trên tập test cục bộ.
        Trả về kết quả của mô hình có Loss thấp nhất (chuẩn IFCA).
        """
        # Nếu không có model nào hoặc worker không có tập test, trả về điểm liệt
        # print(not k_models_dict, not hasattr(self, 'local_test_loader'), self.local_test_loader is None, flush=True)
        if not k_models_dict or not hasattr(self, 'local_test_loader') or self.local_test_loader is None:
            return {"accuracy": 0.0, "error_rate": 1.0, "loss": 10000.0}

        best_loss = float('inf')
        best_acc = 0.0
        best_cid = None

        # 1. BẢO HIỂM: Lưu lại trọng số gốc của Worker trước khi thử model mới
        original_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        self.model.to(self.device)
        self.model.eval()

        # 2. Bắt đầu đánh giá từng model trong k-models
        with torch.no_grad():
            for cid, model_state in k_models_dict.items():
                # Load trọng số của model cụm (cid) vào
                self.model.load_state_dict(model_state)
                
                client_loss = 0.0
                correct = 0
                total = 0

                # Chạy suy luận trên tập Test cục bộ
                for data, target in self.local_test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = self.model(data)
                    
                    loss = self.criterion(outputs, target)
                    client_loss += loss.item() * data.size(0)
                    
                    pred = outputs.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += data.size(0)

                # Tính trung bình Loss và Accuracy
                avg_loss = client_loss / total if total > 0 else float('inf')
                acc = correct / total if total > 0 else 0.0

                # 3. CHỌN MÔ HÌNH TỐT NHẤT: Nếu loss thấp hơn best_loss thì cập nhật
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_acc = acc
                    best_cid = cid

        # 4. TRẢ LẠI HIỆN TRẠNG: Khôi phục trọng số gốc cho Worker
        self.model.load_state_dict(original_state)
        self.model.to('cpu')

        return {
            "accuracy": best_acc,
            "error_rate": 1.0 - best_acc,
            "loss": best_loss,
            "best_cluster_id": best_cid
        }