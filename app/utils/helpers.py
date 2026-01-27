# Các hàm phụ trợ (Math, Cryptography sim)
import torch
import numpy as np
import hashlib
import json
import copy
import math
# --- 1. CHUYỂN ĐỔI DỮ LIỆU (SERIALIZATION) ---

def model_to_json(state_dict):
    """
    Chuyển đổi trọng số model (Tensor) sang dạng List/JSON để gửi qua mạng.
    """
    json_dict = {}
    for key, value in state_dict.items():
        # Chuyển Tensor -> Numpy -> List
        json_dict[key] = value.cpu().numpy().tolist()
    return json_dict

def json_to_model(json_dict, device='cpu'):
    """
    Chuyển đổi từ JSON (List) ngược lại thành Tensor để load vào model.
    """
    state_dict = {}
    for key, value in json_dict.items():
        # Chuyển List -> Tensor
        state_dict[key] = torch.tensor(value).to(device)
    return state_dict

# --- 2. HASHING & BẢO MẬT ---

def compute_model_hash(state_dict):
    """
    Tạo mã băm SHA-256 duy nhất cho một phiên bản model.
    Dùng để định danh model trên Blockchain.
    """
    # 1. Chuyển về dạng list để có thể serialize
    clean_dict = model_to_json(state_dict)
    
    # 2. Sắp xếp key để đảm bảo tính nhất quán (Determinism)
    # Nếu không sort, {'a':1, 'b':2} sẽ khác {'b':2, 'a':1} -> Hash sai
    serialized = json.dumps(clean_dict, sort_keys=True).encode()
    
    # 3. Hash
    return hashlib.sha256(serialized).hexdigest()

# --- 3. TOÁN HỌC & AGGREGATION ---

def federated_averaging(models_list):
    """
    Thuật toán FedAvg: Tính trung bình cộng các tham số của nhiều model.
    Input: Danh sách các state_dict (dạng Tensor).
    """
    if not models_list:
        return None

    # Copy model đầu tiên làm khung
    avg_weights = copy.deepcopy(models_list[0])

    # Duyệt qua từng lớp (layer)
    for key in avg_weights.keys():
        # Lấy tham số của lớp này từ tất cả các model
        layer_updates = [model[key] for model in models_list]
        
        # Tính trung bình (stack lại rồi mean theo chiều 0)
        avg_weights[key] = torch.stack(layer_updates).mean(dim=0)
        
    return avg_weights

def compute_euclidean_distance(w1, w2):
    """
    Tính khoảng cách Euclid giữa 2 model (Dùng cho cơ chế BALANCE).
    Trả về: Một con số (float) thể hiện sự sai khác.
    """
    distance = 0.0
    for key in w1.keys():
        # Lấy hiệu 2 tensor, bình phương, rồi tổng lại
        diff = w1[key] - w2[key]
        distance += torch.sum(diff ** 2).item()
    
    return np.sqrt(distance)

def compute_model_norm(state_dict):
    """
    Tính L2 Norm (Độ lớn vector) của toàn bộ model.
    Dùng để so sánh "mềm" thay vì Hash SHA256.
    """
    total_norm_sq = 0.0
    for key, tensor in state_dict.items():
        # Tính bình phương L2 norm của từng layer và cộng dồn
        # Chuyển về float để tránh overflow
        norm = tensor.float().norm(2).item()
        total_norm_sq += norm ** 2
        
    return math.sqrt(total_norm_sq)