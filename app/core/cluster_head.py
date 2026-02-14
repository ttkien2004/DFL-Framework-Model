import torch
import numpy as np
from app.models.cnn import SimpleCNN, get_model
from config import Config
from app.utils.helpers import federated_averaging, compute_euclidean_distance, compute_model_hash, compute_model_norm
from app.core.coco_helpers import CoCo
from app.core.balance_helpers import Balance
from app.core.worker import WorkerNode
from app.blockchain.proposer import Proposer
from app.models.cnn import get_model
from app.utils.secret_sharing import SecretSharingUtils
# from app.blockchain.consensus import Blockchain

# Class này xử lý CoCo lọc BALANCE và tổng hợp
class ClusterHead(Proposer):
    def __init__(self, cluster_id, config, device, committee_info=None):
        super().__init__(cluster_id, config, device)
        self.members = []
        self.is_head = False

        # Khởi tạo model dựa trên Config
        model_name = config.get('model')
        # Aggregated model
        self.global_model = get_model(model_name, num_classes=self.num_classes).to(self.device)
        # try:
        #     self.global_model = get_model(Config.MODEL_NAME).to(Config.DEVICE)
        # except:
        #     self.global_model = SimpleCNN().to(Config.DEVICE)

        # Trạng thái Topology của cụm
        self.topology_matrix = None

        # Dữ liệu cho CoCo
        self.member_metrics = {}
        self.current_topology = None
        self.d_max_prev = 0.5 # Giá trị d_max khởi tạo
        self.beta2 = 0.1 # hệ số làm mềm d_max

        self.pending_models = [] # Bộ nhớ tạm để chứa model worker gửi lên

        # Lưu trữ public key của committee
        self.committee = []
        self.threshold = None
        self.num_committee = None

        self.rejected_workers = [] # Lưu trữ ID của các worker bị từ chối ở bước BALANCE

    def set_committee_info(self, committee_config):
        if committee_config is not None:
            if 'committee' not in committee_config:
                raise ValueError(f"Missing 'committee' in committee_config: {committee_config}")
            self.threshold = committee_config['t']
            self.num_committee = committee_config['n']
            self.committee = committee_config['committee']
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
            self.pending_models.append((worker_id, formatted_params))
            
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

        for worker_id, update in self.pending_models:
            # Tính khoảng cách Euclid thực sự
            dist = compute_euclidean_distance(global_state, update)
            import math
            if math.isnan(dist) or math.isinf(dist):
                # Nếu model nổ, khoảng cách coi như cực lớn hoặc 0 tuỳ logic hiển thị
                dist = 1000.0
            distances.append(dist)
            updates_with_distance.append((worker_id, dist, update))

        # 3. Tính ngưỡng thích nghi (Adaptive Threshold)
        threshold = Balance.calculate_adaptive_threshold(global_norm, distances, round_k)

        # 4. Lọc bỏ các model vượt quá ngưỡng
        rejected = []
        valid_updates = []
        for worker_id, dist, update in updates_with_distance:
            if dist <= threshold:
                valid_updates.append(update)
            else:
                rejected.append(worker_id)
                print(f"[Refuse] Update rejected! Dist ({dist:.4f}) > Threshold ({threshold:.4f})")

        self.rejected_workers = rejected
        print(f"Cluster {self.cluster_id}: Accepted {len(valid_updates)}/{len(self.pending_models)} updates.")
        return valid_updates
    
    # Cluster Head Phân mảnh cập nhật cụm cho từng Ủy ban
    def distribute_shares_to_committee(self, flat_weights, committee):
        """
        Hàm helper: Tạo mảnh và Mã hóa cho một Ủy ban cụ thể.
        Được dùng trong aggregate() và dùng lại khi View Change.
        """
        sorted_committee = sorted(committee, key=lambda x: x.id)
        committee_ids = sorted([n.id for n in sorted_committee])

        committee_keys = {n.id: n.get_public_key() for n in sorted_committee}
        print(f"[ClusterHead {self.cluster_id}] Generating shares with t={self.threshold}, n={len(committee_ids)}")
        print(f"[ClusterHead] Flat Weights Norm: {torch.norm(flat_weights).item():.4f}")
        # Tạo n mảnh
        raw_shares = SecretSharingUtils.generate_shares(
            flat_weights, 
            n=len(committee_ids), 
            t=self.threshold
        )
        encrypted_packets = {}
        for i, member_id in enumerate(committee_ids, start=1):
            share_vector = raw_shares[i]
            pub_key = committee_keys[member_id]

            encrypted_pkg = SecretSharingUtils.encrypt_share(share_vector, pub_key)

            encrypted_packets[member_id] = encrypted_pkg
        return encrypted_packets


    def aggregate(self, round_k):
        """
        Bước 5: Tổng hợp mô hình
        """
        # Gọi hàm lọc với tham số round_k
        # DEBUG LOGGING -----------------------------------------
        total_received = len(self.pending_models)
        print(f"[Debug Cluster {self.cluster_id}] Received {total_received} updates in pending_models.")
        # -------------------------------------------------------

        # Gọi hàm lọc với tham số round_k
        updates = self.balance_filtering(round_k)
        
        # DEBUG LOGGING -----------------------------------------
        accepted_count = len(updates)
        print(f"[Debug Cluster {self.cluster_id}] After filtering: {accepted_count}/{total_received} accepted.")
        # -------------------------------------------------------
        
        # Reset bộ nhớ đệm
        self.pending_models = []

        avg_state = None
        
        if not updates:
            print(f"Cluster {self.cluster_id}: All updates rejected or empty.")
            # Trả về model cũ nếu không ai đạt chuẩn
            model_hash = compute_model_hash(self.global_model.state_dict())
            # return self.global_model.state_dict(), model_hash
            # return {
            #     "metadata": None,           # Không có dữ liệu mới
            #     "encrypted_shares": None,   # Không có mảnh bí mật
            #     "model_hash": model_hash
            # }
            avg_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        else:
            # Tổng hợp FedAvg
            avg_state = federated_averaging(updates)
        
        self.global_model.load_state_dict(avg_state)
        
        # Tính norm
        model_norm = compute_model_norm(avg_state)
        # Tính Hash
        model_hash = compute_model_hash(avg_state)
        ## Phân mảnh bí mật (SHAMIR SECRET SHARING)
        flat_weights, metadata = SecretSharingUtils.flatten_weights(avg_state)

        # with open("avg_model.txt", "w", encoding="utf-8") as f:
        #     f.write(str(avg_state))
        # Tạo n mảnh bí mật
        # t: Ngưỡng, n: số thành viên ủy ban
        encrypted_packets = self.distribute_shares_to_committee(flat_weights=flat_weights, committee=self.committee)

        self.pending_models = []
        
        # return self.global_model.state_dict(), model_hash
        return {
            "metadata": metadata,
            "encrypted_shares": encrypted_packets,
            "model_hash": model_hash,
            "model_norm": model_norm,
            "flat_weights": flat_weights,
            "rejected_workers": self.rejected_workers
        }