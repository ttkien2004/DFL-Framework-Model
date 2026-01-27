from app.blockchain.validator import Validator
from app.utils.secret_sharing import SecretSharingUtils

class Proposer(Validator):
    """
    Proposer: Có khả năng đóng gói giao dịch và tạo Block mới.
    (Kế thừa Validator vì Proposer cũng cần tự validate trước khi gửi)
    """
    def __init__(self, node_id, config, device):
        super().__init__(node_id, config, device)

    def create_block(self, transactions, blockchain, cluster_id=-1):
        """Logic tạo block mới"""
        print(f"[Proposer {self.id}] Creating new block...")
        # (Gọi logic tạo block của blockchain tại đây hoặc trả về data để Engine tạo)
        return blockchain.create_new_block(transactions, cluster_id)
    
    def reconstruct_model(self, collected_shares, metadata, threshold):
        """
        Giai đoạn (b): Tái cấu trúc bản rõ.
        """
        # 1. Kiểm tra ngưỡng (Threshold Check)
        k = len(collected_shares)
        if k < threshold:
            print(f"[Consensus] Reconstruction FAILED. Not enough shares ({k} < {threshold}).")
            return None
        
        print(f"[Consensus] Reconstructing model from {k} shares (Threshold: {threshold})...")
        
        try:
            # 2. Nội suy Lagrange
            flat_vector = SecretSharingUtils.reconstruct_secret(collected_shares, threshold)
            
            # 3. Unflatten về state_dict
            reconstructed_state_dict = SecretSharingUtils.unflatten_weights(flat_vector, metadata)
            return reconstructed_state_dict
            
        except Exception as e:
            print(f"[Consensus] Reconstruction Error: {e}")
            return None