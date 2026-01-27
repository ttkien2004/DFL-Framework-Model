import torch
import numpy as np
import copy
import pickle
from app.utils.crypto import CryptoUtils

class SecretSharingUtils:
    @staticmethod
    def flatten_weights(state_dict):
        """
        Làm phẳng state_dict thành 1 vector duy nhất (1D Tensor).
        Trả về vector và metadata để khôi phục lại hình dạng sau này.
        """
        flattened = []
        metadata = {}
        offset = 0
        
        for key, tensor in state_dict.items():
            numel = tensor.numel()
            flattened.append(tensor.view(-1))
            metadata[key] = (tensor.shape, offset, offset + numel)
            offset += numel
            
        return torch.cat(flattened), metadata

    @staticmethod
    def unflatten_weights(vector, metadata):
        """
        Khôi phục vector 1D trở lại state_dict.
        """
        state_dict = {}
        for key, (shape, start, end) in metadata.items():
            state_dict[key] = vector[start:end].view(shape)
        return state_dict

    @staticmethod
    def generate_shares(secret_vector, n, t):
        """
        Tạo n mảnh từ vector bí mật với ngưỡng t sử dụng Shamir (Vectorized).
        P(x) = secret + a1*x + a2*x^2 + ... + a(t-1)*x^(t-1)
        
        Args:
            secret_vector: Tensor 1D chứa trọng số mô hình.
            n: Số lượng thành viên ủy ban (số mảnh cần tạo).
            t: Ngưỡng tối thiểu để tái tạo.
        Returns:
            shares: Dictionary {node_id: share_vector}
        """
        num_params = secret_vector.shape[0]
        device = secret_vector.device
        
        # 1. Tạo các hệ số ngẫu nhiên a1, ..., a(t-1)
        # Kích thước: (num_params, t-1)
        coeffs = torch.randn(num_params, t - 1, device=device)
        
        shares = {}
        
        # 2. Tính toán mảnh cho từng thành viên ủy ban (x = 1, ..., n)
        for i in range(1, n + 1):
            x = i
            # Tính phần đuôi đa thức: a1*x + a2*x^2 + ...
            # Sử dụng broadcasting để nhân x^power với coeffs
            x_powers = torch.tensor([x**power for power in range(1, t)], device=device, dtype=torch.float)
            
            # P(x) - secret = sum(ai * x^i)
            poly_tail = torch.matmul(coeffs, x_powers)
            
            # P(x) = secret + poly_tail
            share_val = secret_vector + poly_tail
            
            # Lưu mảnh thứ i
            shares[i] = share_val
            
        return shares

    @staticmethod
    def encrypt_share(share_vector, public_key):
        """
        Mô phỏng mã hóa bất đối xứng (RSA/ECC).
        Trong thực tế, do giới hạn kích thước RSA, ta thường mã hóa share bằng AES,
        sau đó mã hóa key AES bằng RSA Public Key.
        
        Ở đây giả lập bằng cách serialize và wrap lại.
        """
        # Giả lập: Serialize tensor thành bytes
        share_bytes = pickle.dumps(share_vector.cpu())
        
        # TODO: Thay thế dòng này bằng logic mã hóa thực tế:
        # encrypted_data = public_key.encrypt(share_bytes)
        encrypted_data = CryptoUtils.hybrid_encrypt(share_vector, public_key)
        
        # encrypted_data = {"data": share_bytes, "key_id": public_key} # Mock structure
        return encrypted_data
    
    @staticmethod
    def reconstruct_secret(shares, t):
        """
        Tái tạo bí mật từ các mảnh bằng nội suy Lagrange (Vectorized).
        Formula: L(0) = sum( y_j * product( x_m / (x_m - x_j) ) )
        
        Args:
            shares: Dictionary {x_j: y_j} (x_j là ID node, y_j là vector mảnh).
            t: Ngưỡng tối thiểu (dùng để kiểm tra).
        Returns:
            reconstructed_vector: Tensor 1D (w_CH).
        """
        # 1. Kiểm tra ngưỡng
        if len(shares) < t:
            raise ValueError(f"Not enough shares to reconstruct. Need {t}, got {len(shares)}.")
        
        # Lấy t mảnh bất kỳ để tái tạo (theo lý thuyết chỉ cần t mảnh)
        # Chuyển keys (x) và values (y) thành list
        indices = list(shares.keys())[:t]  # x_j
        y_vectors = [shares[i] for i in indices] # y_j
        
        device = y_vectors[0].device
        secret = torch.zeros_like(y_vectors[0], device=device)
        
        # 2. Thực hiện nội suy Lagrange tại x = 0
        for j in range(t):
            x_j = indices[j]
            y_j = y_vectors[j]
            
            # Tính basis polynomial l_j(0)
            numerator = 1.0
            denominator = 1.0
            
            for m in range(t):
                if m == j:
                    continue
                x_m = indices[m]
                
                # l_j(x) = product( (x - x_m) / (x_j - x_m) )
                # Tại x=0: product( (0 - x_m) / (x_j - x_m) ) = product( x_m / (x_m - x_j) )
                numerator *= x_m
                denominator *= (x_m - x_j)
            
            lagrange_coeff = numerator / denominator
            
            # Cộng dồn vào kết quả: secret += y_j * coeff
            secret += y_j * lagrange_coeff
            
        return secret