import torch
from app.models.cnn import SimpleCNN
from app.core.worker import WorkerNode
# from app.blockchain.data_loader import get_global_val_loader
from app.utils.crypto import CryptoUtils

from config import Config

class Validator(WorkerNode):
    def __init__(self, node_id, config, device):
        super().__init__(node_id=node_id, config=config, device=device)
        self.private_key, self.public_key = CryptoUtils.generate_rsa_keypair()

    def _recalibrate_bn(self, loader, num_batches=10):
        """
        Hàm phụ trợ: Chạy model ở chế độ train (nhưng không update weight)
        để cập nhật lại running_mean và running_var của Batch Normalization.
        """
        self.model.train() # Bắt buộc phải là train để update BN stats
        
        # Đóng băng gradient để không làm thay đổi trọng số model
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                if i >= num_batches: break
                x = x.to(self.device)
                self.model(x) # Chỉ cần forward pass, số liệu BN sẽ tự nhảy

    def validate_update(self, model_state_dict, val_loader):
        # 1. KIỂM TRA NHANH: Weights có bị NaN/Inf không?
        for name, param in model_state_dict.items():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"[Validator {self.id}] REJECT: Model contains NaN/Inf weights!")
                return 0.0, False

        # 2. BACKUP Model hiện tại
        original_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        # Đẩy lên GPU
        self.model.to(self.device)
        try:
            # 3. Load model cần chấm điểm
            self.model.load_state_dict(model_state_dict)
            
            # --- FIX: TÁI CÂN CHỈNH BATCH NORM ---
            # Chạy khoảng 10-20 batches dữ liệu để model "làm quen" và sửa lại stats
            # Lưu ý: val_loader ở đây nên có shuffle=True để thống kê tốt hơn
            # self._recalibrate_bn(val_loader, num_batches=20)
            # -------------------------------------

            # 4. Đánh giá (Eval mode)
            self.model.eval()

            correct = 0
            total = 0

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    
                    out = self.model(x)
                    
                    if torch.isnan(out).any():
                        print(f"[Validator {self.id}] Prediction is NaN -> Accuracy = 0")
                        return 0.0, False

                    _, pred = torch.max(out, 1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)

            if total == 0: return 0.0, False
            
            acc = 100.0 * correct / total
            threshold = getattr(Config, 'ACC_THRESHOLD', 0.5)
            
            # Logic vote: Nếu Acc > ngưỡng quy định thì Vote True
            # Lưu ý: threshold trong file config có thể là 0.5 (50%) hoặc 50.0. Hãy kiểm tra kỹ.
            # Nếu threshold là số float nhỏ (0.5), hãy so sánh: acc/100 >= threshold
            # Ở đây giả sử Config để 50.0
            vote = acc >= threshold
            
            print(f"[Validator {self.id}] Validated: Acc={acc:.2f}% | Vote={vote}")
            return acc, vote

        except Exception as e:
            print(f"[Validator {self.id}] Error during validation: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, False

        finally:
            # 5. HOÀN TRẢ lại model cũ
            self.model.load_state_dict(original_state)
            self.model = self.model.to('cpu')
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_public_key(self):
        return self.public_key
    
    def get_private_key(self):
        return self.private_key
    
    # -- Thu thập và giải mã bản cập nhật cluster head gửi
    def decrypt_share(self,encrypted_pkg):
        if encrypted_pkg is None: return None
        try:
            return CryptoUtils.hybrid_decrypt(encrypted_pkg, self.private_key)
        except Exception as e:
            print(f"[Validator {self.id}] Decryption error: {e}")
            return None
        
    def _collect_and_decrypt_shares(self, encrypted_shares_map):
        """
        Giai đoạn (a): Thu thập và giải mã mảnh.
        Giả lập việc các thành viên ủy ban online và giải mã phần của họ.
        """
        collected_shares = {}
        
        # Duyệt qua các thành viên ủy ban
        for member_id in self.committee_ids:
            if member_id in encrypted_shares_map:
                # 1. Lấy gói tin mã hóa
                encrypted_pkg = encrypted_shares_map[member_id]
                
                # 2. Giả lập giải mã (Dùng Private Key tương ứng)
                # Trong thực tế: decrypted = rsa_decrypt(encrypted_pkg, priv_key)
                priv_key = self.committee_private_keys.get(member_id)
                if priv_key:
                    share_vector = CryptoUtils.hybrid_decrypt(encrypted_pkg, priv_key)

                    if share_vector is not None:
                        collected_shares[member_id] = share_vector
                        # return share_vector
                    else:
                        print(f"[Consensus] Failed to decrypt share from member {member_id}")
        
        return collected_shares