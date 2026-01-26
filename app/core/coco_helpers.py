import torch
import numpy as np

class CoCo:
    # =========================================================================
    # CÁC HÀM PHỤ TRỢ CHO COCO (MATH HELPERS)
    # =========================================================================
    @staticmethod
    def flatten_state_dict(state_dict):
        """Chuyển toàn bộ weight của model thành 1 vector duy nhất"""
        return torch.cat([p.view(-1) for p in state_dict.values()])

    @staticmethod
    def calculate_distance_matrix(valid_models_dict, id_to_idx):
        """Tính ma trận khoảng cách Euclidean giữa các model"""
        n = len(valid_models_dict)
        D = np.zeros((n, n))
        flat_vectors = {}
        
        # Flatten trước để tính cho nhanh
        for wid, model in valid_models_dict.items():
            idx = id_to_idx[wid]
            flat_vectors[idx] = CoCo.flatten_state_dict(model).float().cpu()

        # Tính khoảng cách đôi một
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(flat_vectors[i] - flat_vectors[j]).item()
                D[i, j] = dist
                D[j, i] = dist
        return D

    @staticmethod
    def calculate_d_max(valid_models_dict, d_max_prev, beta2=0.1):
        """Cập nhật ràng buộc D_max dựa trên norm của các model"""
        n = len(valid_models_dict)
        if n == 0: return d_max_prev
        
        model_norms = []
        for model in valid_models_dict.values():
            vec = CoCo.flatten_state_dict(model).float()
            model_norms.append(torch.norm(vec).item())
        
        avg_norm = np.mean(model_norms)
        d_max = (1 - beta2) * d_max_prev + (beta2 / n) * sum(model_norms)
        # print(f"[CoCo-CH{self.cluster_id}] D_max = {d_max:.3f} (Prev={d_max_prev:.3f})")
        return d_max

    @staticmethod
    def is_connected(A: np.ndarray, n: int) -> bool:
        """Kiểm tra tính liên thông của đồ thị bằng BFS"""
        if n <= 1: return True
        visited = [False] * n
        queue = [0]
        visited[0] = True
        count = 1 # Đếm số node đã duyệt
        
        while queue:
            node = queue.pop(0)
            for neighbor in range(n):
                if A[node, neighbor] == 1 and not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
        return count == n

    @staticmethod
    def compute_total_time(A, r, b_out, b_in, B, n):
        """Tính tổng thời gian truyền tin (Equation trong báo cáo)"""
        t_cp = 0.1 # Thời gian xử lý CPU giả định
        max_round_time = 0.0
        
        for i in range(n):
            max_comm_time_i = 0.0
            for j in range(n):
                if A[i, j] > 0:
                    # Băng thông giữa i và j là min(upload i, download j)
                    b_ij = min(b_out[i], b_in[j]) 
                    
                    # Thời gian = (Tỷ lệ nén r[i] * Kích thước Model B) / Băng thông
                    if b_ij > 0:
                        transmission_time = (r[i] * B) / (b_ij / 8.0) # /8.0 để đổi Mbps -> MBps
                    else:
                        transmission_time = float('inf')
                        
                    if transmission_time > max_comm_time_i:
                        max_comm_time_i = transmission_time
            
            t_i = t_cp + max_comm_time_i
            if t_i > max_round_time:
                max_round_time = t_i
                
        return max_round_time

    @staticmethod
    def solve_eq27(A_prime: np.ndarray, D: np.ndarray, n: int, D_max: float):
        """Giải phương trình tìm tỷ lệ nén r tối ưu (Binary Search)"""
        m = n
        r = np.ones(n) * 0.5 

        def consensus_loss(r_test):
            total = 0.0
            # Công thức loss xấp xỉ sự sai khác mô hình
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # (1 - A[i,j]*r[j]) nghĩa là nếu có kết nối thì sai số giảm
                        total += (1 - A_prime[i, j] * r_test[j]) * D[i, j]
            return total / (m * m)

        # Binary search cho từng node
        for i in range(n):
            low, high = 0.1, 1.0
            best_r_i = high
            
            for _ in range(10): # Giảm số vòng lặp binary search xuống 10 để nhanh hơn
                mid = (low + high) / 2
                r_test = r.copy()
                r_test[i] = mid
                
                loss = consensus_loss(r_test)
                if loss <= D_max:
                    best_r_i = mid
                    high = mid # Thử nén nhiều hơn (r nhỏ hơn)
                else:
                    low = mid # Phải nén ít hơn (r lớn hơn)
            r[i] = best_r_i
        return r

    @staticmethod
    def select_slowest_links(A: np.ndarray, r: np.ndarray, b_out: list, b_in: list,
                             D: np.ndarray, s: int, n: int, B: float = 1.0) -> list:
        """Chọn ra các liên kết làm chậm hệ thống nhất để thử cắt bỏ"""
        candidates = []
        t_cp = 0.1 
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] > 0:
                    b_ij = min(b_out[i], b_in[j])
                    t_ij = t_cp + (r[i] * B) / (b_ij / 8.0) if b_ij > 0 else float('inf')
                    
                    b_ji = min(b_out[j], b_in[i])
                    t_ji = t_cp + (r[j] * B) / (b_ji / 8.0) if b_ji > 0 else float('inf')
                    
                    link_time = max(t_ij, t_ji)
                    candidates.append((link_time, i, j))
        
        candidates.sort(reverse=True) # Sắp xếp giảm dần theo thời gian
        return [(i, j) for _, i, j in candidates[:s]]

    @staticmethod
    def ADJUSTCR(A: np.ndarray, E: list, D: np.ndarray, n: int,
                 D_max: float, B: float, b_out: list, b_in: list,
                 t_old: float, r_old: np.ndarray) -> tuple:
        """
        Thuật toán tham lam: Thử cắt bỏ các cạnh chậm nhất.
        Nếu cắt xong mà vẫn thỏa mãn D_max và thời gian giảm -> Giữ lại thay đổi.
        """
        A_prime = A.copy()
        
        for i, j in E:
            if A_prime[i, j] == 0: continue
            
            # Thử cắt kết nối
            A_prime[i, j] = 0
            A_prime[j, i] = 0
            
            # Kiểm tra liên thông
            if not CoCo.is_connected(A_prime, n):
                # Nếu mất liên thông -> Hoàn tác
                A_prime[i, j] = 1
                A_prime[j, i] = 1
                continue
            
            # Tính lại r mới cho topology mới
            r_new = CoCo.solve_eq27(A_prime, D, n, D_max)
            
            # Tính thời gian mới
            t_new = CoCo.compute_total_time(A_prime, r_new, b_out, b_in, B, n)
            
            if t_new < t_old:
                # Nếu tốt hơn -> Chấp nhận thay đổi
                print(f"   [Opt] Cut link ({i},{j}) -> Time reduced: {t_old:.3f}s -> {t_new:.3f}s")
                return A_prime, r_new, t_new, True
            
            # Nếu không tốt hơn -> Hoàn tác
            A_prime[i, j] = 1
            A_prime[j, i] = 1
            
        return A, r_old, t_old, False