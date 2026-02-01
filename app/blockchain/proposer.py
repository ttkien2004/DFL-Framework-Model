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
        
    # Dùng cho cơ chế VIEW-CHANGE
    def execute_view_change(self, old_committee, active_validators, blockchain, available_workers):
        """
        (c) Quy trình Thay thế Ủy ban & (d) Trừng phạt.
        Input:
            - old_committee: Danh sách Ủy ban cũ (gặp sự cố).
            - active_validators: Danh sách những người ĐÃ gửi mảnh (không lỗi).
            - blockchain: Để ghi nhận fault.
            - available_workers: Danh sách candidate để chọn người thay thế.
        Output:
            - new_committee: Danh sách ủy ban mới.
        """
        print("\n[View Change] !!! COMMITTEE FAILURE DETECTED !!! Initiating View Change...")

        # 1. Cơ chế trừng phạt (Identify & Penalize)
        # Những người trong old_committee nhưng KHÔNG nằm trong active_validators là người lỗi
        active_ids = {v.id for v in active_validators}
        faulty_nodes = [node for node in old_committee if node.id not in active_ids]

        for node in faulty_nodes:
            print(f" -> Identifying faulty node: {node.id} (Timeout/Offline)")
            blockchain.penalize_node(node.id)

        # 2. Kích hoạt Ủy ban dự phòng
        new_committee = [v for v in old_committee if v.id in active_ids]

        needed = len(old_committee) - len(new_committee)

        if needed > 0:
            old_ids = {n.id for n in old_committee}
            candidates = [
                w for w in available_workers
                if w.id not in old_ids and w.id != self.id
            ]
            # Sắp xép thep Reputation giảm dần
            candidates.sort(
                key=lambda x: blockchain.reputation_scores.get(x.id, 0), 
                reverse=True
            )
            replacements = candidates[:needed]
            new_committee.extend(replacements)
            # print(f" -> Replacing {len(faulty_nodes)} nodes with: {[n.id for n in replacements]}")
        return new_committee
