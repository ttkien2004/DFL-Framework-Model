import torch
import numpy as np
from app.models.cnn import SimpleCNN, get_model
from config import Config
from app.utils.helpers import federated_averaging, compute_euclidean_distance, compute_model_hash
from app.core.coco_helpers import CoCo
from app.core.balance_helpers import Balance
# Class này xử lý CoCom lọc BALANCE và tổng hợp
class ClusterHead:
    def __init__(self, cluster_id, dataset_name='cifar10'):
        self.cluster_id = cluster_id
        self.global_model = SimpleCNN().to(Config.DEVICE)
        self.members = []

        self.reload_model(dataset_name)
        # Khởi tạo model dựa trên Config
        try:
            self.global_model = get_model(Config.MODEL_NAME).to(Config.DEVICE)
        except:
            self.global_model = SimpleCNN().to(Config.DEVICE)

        # Trạng thái Topology của cụm
        self.topology_matrix = None

        # Dữ liệu cho CoCo
        self.member_metrics = {}
        self.current_topology = None
        self.d_max_prev = 0.5 # Giá trị d_max khởi tạo
        self.beta2 = 0.1 # hệ số làm mềm d_max

        self.pending_models = [] # Bộ nhớ tạm để chứa model worker gửi lên

    def reload_model(self, dataset_name):
        if dataset_name == 'gtsrb': num_classes = 43
        else: num_classes = 10

        self.global_model = get_model(Config.MODEL_NAME, num_classes=num_classes)
        self.pending_models = []
        print(f"[CH {self.cluster_id}] Reset global model for {dataset_name}")
    
    def register_member(self, worker_id):
        """Đăng ký worker vào cụm"""
        if worker_id not in self.members:
            self.members.append(worker_id)

    def receive_metrics(self, worker_id, metrics, model_state_dict):
        """
        Giai đoạn 1: Nhận báo cáo trạng thái từ Worker 
        (Thay thế cho việc nhận model ngay lập tức)
        metrics: {'bandwidth': float, 'loss': float}
        """
        self.member_metrics[worker_id] = {
            'bandwidth': metrics.get('bandwidth', 10.0), # Mbps
            'model': model_state_dict
        }

    def run_coco_optimization(self):
        """
        Giai đoạn 2 (CoCo Phase): Tối ưu hóa cấu trúc mạng và tỷ lệ nén.
        """
        members_list = list(self.member_metrics.keys())
        n = len(members_list)
        if n < 2:
            return {}, []

        # 1. Mapping ID <-> Index (0..n-1)
        id_to_idx = {mid: i for i, mid in enumerate(members_list)}
        idx_to_id = {i: mid for i, mid in enumerate(members_list)}

        # 2. Chuẩn bị dữ liệu đầu vào
        b_out = []
        valid_models = {}
        
        for mid in members_list:
            data = self.member_metrics[mid]
            b_out.append(data['bandwidth'])
            valid_models[mid] = data['model']
            
        b_in = b_out # Giả sử upload = download (đơn giản hóa)
        
        # Ước lượng kích thước model (MB)
        # Giả sử model có 1 triệu tham số -> 4MB
        param_count = sum(p.numel() for p in valid_models[members_list[0]].values())
        B_model_size = (param_count * 4) / (1024 * 1024) 

        # 3. Tính toán Ma trận khoảng cách D và D_max
        D_matrix = CoCo.calculate_distance_matrix(valid_models, id_to_idx)
        self.d_max_prev = CoCo.calculate_d_max(valid_models, self.d_max_prev, self.beta2)

        # 4. Khởi tạo Topology (Ring hoặc Fully Connected nếu chưa có)
        if self.current_topology is None or self.current_topology.shape[0] != n:
            # Khởi tạo Ring Topology để đảm bảo liên thông ban đầu
            self.current_topology = np.zeros((n, n))
            for i in range(n):
                self.current_topology[i, (i + 1) % n] = 1
                self.current_topology[(i + 1) % n, i] = 1
                self.current_topology[i, i] = 1 # Self-loop

        A_init = self.current_topology.copy()

        # 5. Giải bài toán tối ưu lần 1
        r_current_best = CoCo.solve_eq27(A_init, D_matrix, n, self.d_max_prev)
        t_current_best = CoCo.compute_total_time(A_init, r_current_best, b_out, b_in, B_model_size, n)
        
        print(f"[CoCo-CH{self.cluster_id}] Initial T = {t_current_best:.4f}s")

        # 6. Tìm và cắt bỏ các cạnh chậm (Iterative Optimization)
        # Thử lặp vài lần để tối ưu
        for _ in range(3): 
            # Chọn top N cạnh chậm nhất để thử cắt
            slowest_links = CoCo.select_slowest_links(A_init, r_current_best, b_out, b_in, D_matrix, n, n, B_model_size)
            
            A_new, r_new, t_new, improved = CoCo.ADJUSTCR(
                A_init, slowest_links, D_matrix, n, self.d_max_prev, B_model_size, b_out, b_in,
                t_current_best, r_current_best
            )
            
            if improved:
                A_init = A_new
                r_current_best = r_new
                t_current_best = t_new
            else:
                break # Không tối ưu thêm được nữa

        # 7. Cập nhật trạng thái
        self.current_topology = A_init
        r_final = np.clip(r_current_best, 0.1, 1.0) # Clip CR trong khoảng [0.1, 1.0]

        # 8. Đóng gói Instruction gửi về Worker
        instruction_map = {}
        u_max = np.max(np.sum(A_init, axis=1)) # Degree lớn nhất

        for i in range(n):
            worker_id = idx_to_id[i]
            
            # Tìm danh sách hàng xóm (ID)
            neighbor_indices = [j for j in range(n) if A_init[i, j] == 1 and i != j]
            neighbor_ids = [idx_to_id[idx] for idx in neighbor_indices]
            
            instruction_map[worker_id] = {
                'neighbors': neighbor_ids,
                'compression_ratio': float(r_final[i]),
                'u_max': int(u_max),
                'cluster_head_id': self.cluster_id
            }

        return instruction_map, A_init.tolist()


    def receive_update(self, worker_id, noisy_params):
        """
        Nhận mô hình (đã train & LDP) từ Worker gửi lên.
        Hàm này chỉ nhận và lưu, chưa xử lý logic (Logic xử lý nằm ở aggregate).
        """
        if noisy_params is None:
            print(f"[CH {self.cluster_id}] Worker {worker_id} sent None update.")
            return

        # Worker khi gửi lên thường để ở CPU để tiết kiệm bộ nhớ GPU khi truyền
        try:
            formatted_params = {
                k: v.to(Config.DEVICE) for k, v in noisy_params.items()
            }
            
            # Lưu vào danh sách chờ
            # (Bạn có thể lưu thêm worker_id nếu muốn log chi tiết ai bị loại ở bước BALANCE)
            self.pending_models.append(formatted_params)
            
            # Log (có thể comment lại nếu spam terminal quá nhiều)
            # print(f"[CH {self.cluster_id}] Received update from Worker {worker_id}")
            
        except Exception as e:
            print(f"[CH {self.cluster_id}] Error receiving update from {worker_id}: {e}")

    def balance_filtering(self, round_k):
        """
        Lọc mô hình độc hại sử dụng BALANCE Adaptive Threshold
        """
        if not self.pending_models:
            return []

        # 1. Chuẩn bị dữ liệu Global Model
        global_state = self.global_model.state_dict()
        global_vec = Balance.flatten_model(global_state)
        global_norm = torch.norm(global_vec).item()

        # 2. Tính khoảng cách từ Global Model đến TẤT CẢ các model nhận được
        # (Cần tính hết để tìm ra Median cho công thức)
        updates_with_distance = []
        distances = []

        for update in self.pending_models:
            # Tính khoảng cách Euclid thực sự
            dist = compute_euclidean_distance(global_state, update)
            distances.append(dist)
            updates_with_distance.append((dist, update))

        # 3. Tính ngưỡng thích nghi (Adaptive Threshold)
        threshold = Balance.calculate_adaptive_threshold(global_norm, distances, round_k)

        # 4. Lọc bỏ các model vượt quá ngưỡng
        valid_updates = []
        for dist, update in updates_with_distance:
            if dist <= threshold:
                valid_updates.append(update)
            else:
                print(f"[Refuse] Update rejected! Dist ({dist:.4f}) > Threshold ({threshold:.4f})")

        print(f"Cluster {self.cluster_id}: Accepted {len(valid_updates)}/{len(self.pending_models)} updates.")
        return valid_updates

    def aggregate(self, round_k):
        """
        Bước 5: Tổng hợp mô hình
        """
        # Gọi hàm lọc với tham số round_k
        updates = self.balance_filtering(round_k)
        
        # Reset bộ nhớ đệm
        self.pending_models = []

        if not updates:
            print(f"Cluster {self.cluster_id}: All updates rejected or empty.")
            # Trả về model cũ nếu không ai đạt chuẩn
            model_hash = compute_model_hash(self.global_model.state_dict())
            return self.global_model.state_dict(), model_hash
        
        # Tổng hợp FedAvg
        avg_state = federated_averaging(updates)
        self.global_model.load_state_dict(avg_state)
        
        # Tính Hash
        model_hash = compute_model_hash(avg_state)

        self.pending_models = []
        
        return self.global_model.state_dict(), model_hash