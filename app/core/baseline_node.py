import copy
from app.core.worker import WorkerNode
from app.core.baseline_aggregation import AggregationAlgorithms
from app.utils.helpers import get_model_size_mb

class StandardDFLNode(WorkerNode):
    """
    Node chuẩn cho mạng DFL truyền thống (Baseline).
    Đặc điểm:
    - Không có Clustering động (DFCA).
    - Không có Blockchain Reputation.
    - Không có CoCo Compression (hoặc chỉ nén cố định).
    - Topology thường là cố định (Random Graph hoặc Ring).
    """
    def __init__(self, node_id, config, device):
        super().__init__(node_id, config, device)
        self.neighbors = [] # Danh sách hàng xóm cố định
        self.received_updates = {}

        import random
        self.b_out = random.uniform(10, 50)
        self.b_in = random.uniform(20, 100)
        self.compression_ratio = 1.0
        self.model_size_mb = 1.2

    def set_neighbors(self, neighbor_indices):
        """Thiết lập Topology cố định"""
        self.neighbors = neighbor_indices
    

    def aggregate(self, current_round_id=0):
        """
        Tổng hợp mô hình từ hàng xóm sử dụng thuật toán được cấu hình.
        """
        # 1. Thu thập updates
        valid_updates = list(self.received_updates.values())
        
        # Thêm chính mình
        my_update = self.model.state_dict()
        valid_updates.append(self.model.state_dict())
        
        # 2. Chọn thuật toán (FedAvg, Krum, Median...)
        algo_name = self.config.get('aggregation_algorithm', 'FED_AVG')
        
        # print(f"   [Node {self.id}] Aggregating {len(valid_updates)} models using {algo_name}")

        aggregated_model = None
        if algo_name == 'FED_AVG':
            aggregated_model = AggregationAlgorithms.fed_avg(valid_updates)
        elif algo_name == 'KRUM':
            aggregated_model = AggregationAlgorithms.krum(valid_updates)
        elif algo_name == 'MEDIAN':
            aggregated_model = AggregationAlgorithms.coordinate_wise_median(valid_updates)
        elif algo_name == 'TRIMMED_MEAN':
            aggregated_model = AggregationAlgorithms.trimmed_mean(valid_updates)
        elif algo_name == 'FL_TRUST':
            # Với DFL thường, node tin tưởng bản thân nó nhất
            aggregated_model = AggregationAlgorithms.fl_trust(valid_updates, server_update=my_update)
        elif algo_name == 'UBAR':
            # --- CHUẨN BỊ DỮ LIỆU CHO UBAR ---
            # Cần lấy 1 batch từ data_loader để tính loss
            try:
                # Tạo iterator tạm thời để lấy 1 batch
                data_iter = iter(self.data_loader)
                data_batch = next(data_iter)
            except Exception as e:
                print(f"[Node {self.id}] Cannot fetch batch for UBAR: {e}")
                # Fallback nếu lỗi data: dùng FedAvg
                data_batch = None
            
            if data_batch:
                # Gọi UBAR static method
                aggregated_model = AggregationAlgorithms.ubar(
                    own_state_dict=my_update,
                    neighbor_updates=valid_updates, # Danh sách weights hàng xóm
                    model_template=self.model,      # Dùng model của mình làm khuôn
                    data_batch=data_batch,
                    criterion=self.criterion,
                    device=self.device,
                    rho=self.config.get('ubar_rho', 0.5) # Config tỷ lệ tin cậy
                )
                
                # UBAR dùng self.model để thử weights hàng xóm
                # Nên sau khi chạy xong, self.model đang chứa weights lung tung.
                # Cần load lại ngay kết quả đúng.
                self.model.load_state_dict(aggregated_model)
            else:
                aggregated_model = AggregationAlgorithms.fed_avg(valid_updates)
        elif algo_name == "BALANCE":
            gamma = self.config.get("BALANCE_GAMMA", 0.3)
            lambda_val = self.config.get("BALANCE_LAMBDA",1.0)
            total_rounds = self.config.get("NUM_ROUNDS", 100)
            aggregated_model = AggregationAlgorithms.balance(
                updates=valid_updates,
                current_round=current_round_id+1,
                total_rounds=total_rounds,
                gamma=gamma,
                lambda_val=lambda_val
            )
        else:
            aggregated_model = AggregationAlgorithms.fed_avg(valid_updates)

        # 3. Cập nhật model
        self.model.load_state_dict(aggregated_model)
        
        # Reset buffer cho vòng sau
        self.received_updates = {}
        
        return aggregated_model
    
    def apply_coco_config(self, neighbors, compression_rate):
        self.neighbors = neighbors
        self.compression_ratio = compression_rate
    
    def gossip(self, all_workers_dict, is_coco_mode=False):
        payload = self.model.state_dict()
        sent_count = 0
        max_latency = 0.0

        # Tính kích thước gốc
        original_size_mb = get_model_size_mb(self.model)
        # Nếu chạy CoCo, áp dụng tỷ lệ nén r
        actual_size_mb = original_size_mb * (self.compression_rate if is_coco_mode else 1.0)

        for neighbor_id in self.neighbors:
            neighbor = all_workers_dict.get(neighbor_id)
            if neighbor:
                # Gửi model
                neighbor.received_updates[self.id] = copy.deepcopy(payload)
                sent_count += 1

                # Tính latency (chỉ quan trọng nếu chạy CoCo để đo đạc)
                if is_coco_mode:
                    # Time = (r * Size) / min(BW_upload, BW_download)
                    # Chuyển đổi Mbps -> MBps bằng cách chia 8
                    bw = min(self.b_out, neighbor.b_in)
                    if bw > 0:
                        latency = (self.compression_rate * self.model_size_mb) / (bw / 8.0)
                        if latency > max_latency:
                            max_latency = latency
        traffic_mb = sent_count * actual_size_mb
        return sent_count, max_latency, traffic_mb