import copy
from app.core.worker import WorkerNode
from app.core.baseline_aggregation import AggregationAlgorithms

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

    def set_neighbors(self, neighbor_indices):
        """Thiết lập Topology cố định"""
        self.neighbors = neighbor_indices

    def aggregate(self):
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
            gamma = self.config("BALANCE_GAMMA", 0.3)
            lambda_val = self.config.get("BALANCE_LAMBDA",1.0)
            total_rounds = self.config.get("NUM_ROUNDS", 100)
            aggregated_model = AggregationAlgorithms.balance(
                updates=valid_updates,
                current_round=self.current_round,
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