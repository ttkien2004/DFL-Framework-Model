import torch
import copy
import numpy as np
import math
import torch.nn.functional as F

class AggregationAlgorithms:
    """
    Thư viện chứa các giải thuật tổng hợp mô hình (Robust Aggregation Rules).
    Dùng để so sánh hiệu quả với mô hình đề xuất.
    """

    @staticmethod
    def fed_avg(updates):
        """
        FedAvg: Trung bình cộng đơn giản (Standard Baseline).
        Nhược điểm: Rất nhạy cảm với dữ liệu độc hại (Poisoning).
        """
        if not updates: return None
        
        # Lấy model đầu tiên làm khuôn mẫu
        avg_model = copy.deepcopy(updates[0])
        n = len(updates)
        
        for key in avg_model.keys():
            # Cộng dồn tham số
            layer_updates = [update[key].cpu() for update in updates]
            
            # Chỉ tính trung bình nếu là số thực (Float)
            if layer_updates[0].is_floating_point():
                import torch
                avg_model[key] = torch.stack(layer_updates).mean(dim=0).clone().detach()
            else:
                # Nếu là số nguyên (như num_batches_tracked), giữ nguyên
                avg_model[key] = layer_updates[0].clone().detach()
            
        return avg_model

    @staticmethod
    def coordinate_wise_median(updates):
        """
        Hiện thực giải thuật Median (Coordinate-wise Median).
        
        Luồng hoạt động tương ứng:
        1. Thu thập (Collection): Nhận vào list `updates`.
        2. Xử lý theo tọa độ (Coordinate-wise):
           - Với mỗi tham số k (key trong state_dict):
             + Trích xuất giá trị từ n updates.
             + Sắp xếp (Hàm torch.median tự thực hiện ngầm định việc tìm vị trí giữa).
        3. Tính trung vị (Median Calculation):
           - Lấy giá trị median của từng tọa độ.
        
        Args:
            updates (list[state_dict]): Danh sách các model updates (w_1, ..., w_n).
            
        Returns:
            state_dict: Vector tham số tổng hợp mới (w_med).
        """
        if not updates: return None
        
        median_model = copy.deepcopy(updates[0])
        
        for key in median_model.keys():
            # Stack thành tensor: [num_workers, param_shape...]
            tensors = [u[key] for u in updates]
            stacked = torch.stack(tensors, dim=0)
            
            # Lấy trung vị theo chiều dọc (dim=0)
            med_val, _ = torch.median(stacked, dim=0)
            median_model[key] = med_val
            
        return median_model

    @staticmethod
    def trimmed_mean(updates, beta=0.1):
        """
        Hiện thực giải thuật Trimmed Mean.
        
        Luồng hoạt động:
        1. Thu thập n updates.
        2. Xử lý theo từng tọa độ (Coordinate-wise).
        3. Sắp xếp giá trị tại mỗi tọa độ.
        4. Cắt tỉa (Trim) b giá trị lớn nhất và b giá trị nhỏ nhất.
        5. Tính trung bình phần còn lại.
        6. Cập nhật vào model đích.
        
        Args:
            updates (list[state_dict]): Danh sách các bản cập nhật (w_1, ..., w_n).
            beta (float): Tỷ lệ cắt tỉa (ví dụ 0.1 -> loại bỏ 10% mỗi đầu). 
                          Điều kiện: beta < 0.5.
                          
        Returns:
            state_dict: Vector tham số tổng hợp mới (w_tm).
        """
        if not updates: return None
        
        n = len(updates)
        cut = int(math.ceil(n * beta))
        
        # Nếu ít worker quá thì fallback về FedAvg
        if n - 2 * cut <= 0: 
            return AggregationAlgorithms.fed_avg(updates)

        trimmed_model = copy.deepcopy(updates[0])
        
        for key in trimmed_model.keys():
            stacked = torch.stack([up[key] for up in updates], dim=0)
            
            # Sắp xếp
            sorted_stack, _ = torch.sort(stacked, dim=0)
            
            # Cắt bỏ đầu đuôi (slicing)
            kept_values = sorted_stack[cut : n-cut]
            
            # Tính trung bình phần còn lại
            trimmed_model[key] = torch.mean(kept_values, dim=0)
            
        return trimmed_model

    @staticmethod
    def krum(updates, f=None):
        """
        Hiện thực giải thuật KRUM (Krum Aggregation).
        
        Luồng hoạt động:
        1. Thu thập n updates.
        2. Tính khoảng cách Euclidean bình phương giữa mọi cặp (pairwise distance).
        3. Với mỗi update i, chọn n-f-2 láng giềng gần nhất.
        4. Tính điểm s(i) = Tổng bình phương khoảng cách đến các láng giềng đó.
        5. Chọn update có điểm s(i) thấp nhất.
        
        Args:
            updates (list[state_dict]): Danh sách các model updates.
            f (int): Số lượng Byzantine workers giả định. Mặc định là n * 0.3.
            
        Returns:
            state_dict: Update tốt nhất được chọn (Best User Selection).
        """
        if not updates: return None
        n = len(updates)
        
        # f: Số lượng Byzantine worker giả định (mặc định < n/2)
        if f is None: f = int(n * 0.3) 
        
        # k: Số lượng hàng xóm để tính khoảng cách
        k = n - f - 2
        if k <= 0: k = n - 1

        # 1. Flatten updates ra vector 1 chiều để tính distance
        flattened_updates = []
        for up in updates:
            # Nối tất cả tham số lại thành 1 vector dài
            vec = torch.cat([p.view(-1).float() for p in up.values()])
            flattened_updates.append(vec)
        
        # Stack thành ma trận [n, total_params]
        matrix = torch.stack(flattened_updates)
        
        # 2. Tính ma trận khoảng cách đôi một (Pairwise Distance)
        dists = torch.cdist(matrix, matrix, p=2) # Euclidean Distance
        sq_dists = dists ** 2
        # 3. Tính Krum Score
        scores = []
        for i in range(n):
            # Lấy khoảng cách từ i đến các node khác, sắp xếp tăng dần
            d_i = sq_dists[i]
            sorted_dists, _ = torch.sort(d_i)

            # Chọn k láng giềng gần nhất (bỏ qua chính nó ở index 0)
            neighbors_dists = sorted_dists[1: k+1]
            s_i = torch.sum(neighbors_dists)
            scores.append(s_i)
            
        # 4. Chọn update có score nhỏ nhất
        best_idx = torch.argmin(torch.tensor(scores)).item()
        
        return copy.deepcopy(updates[best_idx])

    @staticmethod
    def fl_trust(updates, server_update):
        """
        Hiện thực giải thuật FLTrust (Federated Learning with Trust).
        
        Ý tưởng: Sử dụng một bản cập nhật tin cậy (server_update - g0) để đánh giá 
        và chuẩn hóa các bản cập nhật từ client/hàng xóm (updates - gi).
        
        Args:
            updates (list[state_dict]): Danh sách các bản cập nhật từ hàng xóm (g_i).
            server_update (state_dict): Bản cập nhật do chính node này tự train (g_0).
            
        Returns:
            state_dict: Vector tổng hợp g_global.
        """
        if not updates: return None
        if server_update is None: return AggregationAlgorithms.fed_avg(updates)
        
        # Flatten Server Update
        g0_vec = torch.cat([p.float().view(-1) for p in server_update.values()])
        # g0_vec = torch.cat(g0_params)
        g0_norm = torch.norm(g0_vec)

        if g0_norm == 0:
            return server_update
        
        weighted_sum_model = copy.deepcopy(updates[0])
        # Xóa dữ liệu cũ để bắt đầu cộng dồn
        for k in weighted_sum_model.keys():
            weighted_sum_model[k].zero_()

        total_trust_score = 0.0
        epsilon = 1e-9
        trusted_updates = []
        
        for g_i in updates:
            gi_vec = torch.cat([p.float().view(-1) for p in g_i.values()])
            gi_norm = torch.norm(gi_vec)
            if gi_norm == 0:
                continue
            
            # Cosine Similarity = (A . B) / (||A|| * ||B||)
            cos_sim = torch.dot(g0_vec, gi_vec) / (gi_norm * g0_norm + epsilon)
            
            # Trust Score = ReLU(Cosine)
            trust_score = F.relu(cos_sim).item()
            
            # Scaling Factor (FLTrust normalization rule)
            if trust_score <= 0:
                continue
            
            # Công thức chuẩn: g_bar = (||g0|| / ||gi||) * gi
            # Contribution = TS_i * g_bar_i
            #              = TS_i * (||g0|| / ||gi||) * g_i
            norm_factor = g0_norm / gi_norm
            # scaling = trust_score * (g0_norm / (gi_norm + epsilon))
            
            # Cộng dồn update đã được scale
            for k in weighted_sum_model.keys():
                scaled_tensor = g_i[k].float() * norm_factor
                weighted_sum_model[k] += trust_score * scaled_tensor
            total_trust_score += trust_score
                
        # Chia trung bình (Normalization step)
        # g_global = Sum(TS * g_bar) / Sum(TS)
        if total_trust_score == 0:
            return copy.deepcopy(server_update)
        
        for k in weighted_sum_model.keys():
            weighted_sum_model[k] = weighted_sum_model[k] / total_trust_score
            
        return weighted_sum_model
    
    def ubar(own_state_dict, neighbor_updates, model_template, data_batch, criterion, device, rho=0.5):
        """
        Hiện thực giải thuật UBAR (Under-Cover Byzantine Attack Resilient).
        
        Args:
            own_state_dict: Tham số mô hình hiện tại của nút i (x_{k,i}).
            neighbor_updates: List các tham số từ láng giềng ({x_{k,j}}).
            model_template: Một object mô hình (nn.Module) dùng để load weights và tính loss.
            data_batch: Một tuple (inputs, labels) lấy từ dữ liệu cục bộ.
            criterion: Hàm loss (thường là CrossEntropy).
            device: CPU/GPU.
            rho: Tỷ lệ nút lành tính dự kiến (0 < rho <= 1).
            
        Returns:
            state_dict: Vector tổng hợp R_{k,i} (đã lọc sạch).
        """
        if not neighbor_updates:
            return own_state_dict

        num_neighbors = len(neighbor_updates)
        
        # --- GIAI ĐOẠN 1: SÀNG LỌC DỰA TRÊN KHOẢNG CÁCH ---
        # Mục tiêu: Chọn ra N_s gồm floor(rho * |N|) láng giềng gần nhất.
        
        # 1. Flatten own model để tính khoảng cách
        own_vec = torch.cat([p.view(-1).float() for p in own_state_dict.values()])
        
        distances = []
        for idx, w_j in enumerate(neighbor_updates):
            # Flatten neighbor model
            neigh_vec = torch.cat([p.view(-1).float() for p in w_j.values()])
            # Tính Euclidean distance ||x_i - x_j||
            d_ij = torch.norm(own_vec - neigh_vec).item()
            distances.append((d_ij, idx))
            
        # 2. Sắp xếp theo khoảng cách tăng dần
        distances.sort(key=lambda x: x[0])
        
        # 3. Chọn tập ứng viên N_s (Stage 1 candidates)
        num_candidates = int(math.floor(rho * num_neighbors))
        # Đảm bảo ít nhất có 1 ứng viên
        if num_candidates < 1: num_candidates = 1
        
        candidate_indices = [x[1] for x in distances[:num_candidates]]
        
        # --- GIAI ĐOẠN 2: ĐÁNH GIÁ DỰA TRÊN HIỆU SUẤT ---
        # Mục tiêu: So sánh Loss trên dữ liệu cục bộ của mình.
        
        inputs, labels = data_batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 1. Tính Loss của chính mình (l_{k,i})
        # Load own weights vào model tạm để tính loss
        model_template.load_state_dict(own_state_dict)
        model_template.eval() # Chế độ eval để tắt Dropout/BatchNorm update
        
        with torch.no_grad():
            outputs = model_template(inputs)
            own_loss = criterion(outputs, labels).item()
            
        final_selection = [] # N^r_{k,i}
        
        # Lưu lại loss của các ứng viên để xử lý trường hợp ngoại lệ
        candidate_losses = [] 

        # 2. Duyệt qua các ứng viên từ Giai đoạn 1
        for idx in candidate_indices:
            w_j = neighbor_updates[idx]
            
            # Load weights của láng giềng vào model
            model_template.load_state_dict(w_j)
            
            with torch.no_grad():
                outputs = model_template(inputs)
                loss_j = criterion(outputs, labels).item()
            
            candidate_losses.append((loss_j, w_j))
            
            # So sánh: Nếu l_j <= l_i -> Chấp nhận
            if loss_j <= own_loss:
                final_selection.append(w_j)
                
        # --- XỬ LÝ NGOẠI LỆ & TỔNG HỢP ---
        
        # Trường hợp ngoại lệ: Nếu không có láng giềng nào tốt hơn mình (N^r rỗng)
        # Chọn láng giềng có loss nhỏ nhất trong số các ứng viên.
        if not final_selection:
            # Sắp xếp candidate_losses theo loss tăng dần
            candidate_losses.sort(key=lambda x: x[0])
            best_neighbor = candidate_losses[0][1]
            final_selection.append(best_neighbor)
            # print(f"UBAR: No better neighbor found. Picked min-loss neighbor.")

        # Tính trung bình (Average) các model trong final_selection
        # R_{k,i} = Mean(N^r)
        return AggregationAlgorithms.fed_avg(final_selection)
    
    @staticmethod
    def get_model_norm(state_dict):
        """Hàm phụ trợ: Tính L2 Norm của toàn bộ trọng số model"""
        # Gom tất cả các tensor thành 1 vector phẳng để tính norm
        flattened = torch.cat([p.view(-1).float() for p in state_dict.values()])
        return torch.norm(flattened).item()

    @staticmethod
    def baseline_threshold(w_norm, current_round, total_rounds, gamma, lambda_val):
        """
        Tính threshold/weight dựa trên công thức BALANCE Baseline của bạn.
        Formula: Gamma * exp(-Lambda * (t/T)) * ||w_i||
        """
        # Tránh chia cho 0
        T = max(1, total_rounds)
        Omega = current_round / T
        
        # Tính hệ số thời gian: exp(-Lambda * Omega)
        time_factor = math.exp(-lambda_val * Omega)
        
        return gamma * time_factor * w_norm

    @staticmethod
    def balance(updates, current_round, total_rounds, gamma=1.0, lambda_val=1.0):
        """
        Chiến lược BALANCE: Tổng hợp model dựa trên trọng số được tính toán
        từ Norm và thời gian.
        
        Args:
            updates: List các state_dict từ workers
            current_round: Vòng hiện tại (t)
            total_rounds: Tổng số vòng (T)
            gamma: Hệ số cân bằng (Gamma)
            lambda_val: Hệ số suy giảm (Lambda)
        """
        if not updates:
            return None

        # 1. Tính toán điểm số (score) cho từng update
        # Score này sẽ đóng vai trò là 'trọng số' (weight) khi cộng gộp
        scores = []
        for w in updates:
            w_norm = AggregationAlgorithms.get_model_norm(w)
            score = AggregationAlgorithms.baseline_threshold(
                w_norm, current_round, total_rounds, gamma, lambda_val
            )
            scores.append(score)

        # 2. Chuẩn hóa score để tổng các trọng số bằng 1 (Softmax hoặc Normalize)
        # Ở đây dùng Normalize đơn giản: weight_i = score_i / sum(scores)
        total_score = sum(scores)
        if total_score == 0:
            weights = [1.0 / len(updates)] * len(updates) # Tránh lỗi chia 0
        else:
            weights = [s / total_score for s in scores]

        # 3. Tổng hợp trọng số (Weighted Averaging)
        # w_global = sum(weight_i * w_i)
        first_model = updates[0]
        avg_update = {k: torch.zeros_like(v).float() for k, v in first_model.items()}

        for w, weight in zip(updates, weights):
            for k in avg_update:
                avg_update[k] += w[k].float() * weight

        # Chuyển về đúng kiểu dữ liệu gốc (nếu cần, ví dụ float16/double)
        # Thông thường giữ float32 là ổn
        return avg_update