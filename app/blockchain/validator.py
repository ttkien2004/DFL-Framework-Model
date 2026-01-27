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

    def validate_update(self, model_state_dict,val_loader):
        # 1. Khởi tạo model
        # model = SimpleCNN().to(Config.DEVICE)

        # # 2. Load trọng số
        self.model.load_state_dict(model_state_dict)

        # # 3. Eval mode
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                _, pred = torch.max(out, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = 100.0 * correct / total

        vote = acc >= getattr(Config, 'ACC_THRESHOLD', 0.5)

        return acc, vote

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