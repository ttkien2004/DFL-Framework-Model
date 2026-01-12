# Logic của Cơ chế đồng thuận ủy ban
import time
import hashlib
import json
from config import Config
from app.blockchain.block import Block

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.committee = [] # Danh sách node uy tín
        # Bảng điểm uy tín (Lưu trữ trên chuỗi - Ledger State)
        # Key: Node ID, Value: Reputation Score
        self.reputation_scores = {}
    
    def _initialize_reputation(self):
        """Khởi tạo điểm cho các node tham gia"""
        # Giả sử có 5 worker và 2 CH
        all_nodes = [f"ClusterHead_{i}" for i in range(Config.NUM_CLUSTERS)] + self.committee
        for node in all_nodes:
            self.reputation_scores[node] = Config.INITIAL_REPUTATION

    def create_genesis_block(self):
        return Block(0, "0", "Genesis Model", time.time())

    def get_latest_block(self):
        return self.chain[-1]

    def propose_update(self, cluster_updates):
        """Bước 6: Cơ chế Đồng thuận (Consensus)"""
        # Giả lập Committee voting
        votes = 0
        for member in self.committee:
            # Logic kiểm tra model trên tập validation của member
            votes += 1 

        # Tính tỷ lệ phiếu bầu cần thiết
        required_votes = len(self.committee) * Config.CONSENSUS_THRESHOLD
        
        if votes > required_votes:
            self.add_block(cluster_updates)
            return True
        return False
    
    # --- SMART CONTRACT LOGIC ---
    def execute_smart_contract(self, proposer_id, validation_accuracy, is_approved, voters):
        """
        Hàm này đóng vai trò là Smart Contract:
        Tự động tính toán và cập nhật điểm dựa trên kết quả đồng thuận.
        """
        print(f"\n---Executing Smart Contract for {proposer_id} ---")
        
        current_score = self.reputation_scores.get(proposer_id, Config.INITIAL_REPUTATION)
        
        if is_approved:
            # 1. Tính thưởng cho Proposer (Cluster Head)
            # Công thức: Điểm cũ + Thưởng cơ bản + (Độ chính xác * Hệ số)
            acc_bonus = validation_accuracy * Config.ACCURACY_BONUS_FACTOR
            new_score = current_score + Config.REWARD_SUCCESSFUL_BLOCK + acc_bonus
            
            print(f"Block Approved! {proposer_id} gained points.")
            print(f"   Reward: {Config.REWARD_SUCCESSFUL_BLOCK}, Acc Bonus: {acc_bonus:.2f}")

            # 2. Thưởng cho các thành viên Ủy ban (Voters) đã làm việc
            for voter in voters:
                v_score = self.reputation_scores.get(voter, Config.INITIAL_REPUTATION)
                self.reputation_scores[voter] = v_score + Config.REWARD_COMMITTEE_VOTE
                
        else:
            # Phạt Proposer vì gửi model kém chất lượng
            new_score = current_score + Config.PENALTY_REJECTED_BLOCK
            print(f"Block Rejected! {proposer_id} penalized.")

        # 3. Áp dụng Decay (Giảm nhẹ điểm của tất cả để tránh lạm phát điểm)
        # (Tùy chọn: chỉ giảm những người không hoạt động, ở đây ta đơn giản hóa)
        new_score = new_score * Config.DECAY_FACTOR
        
        # Cập nhật vào bảng điểm
        self.reputation_scores[proposer_id] = round(new_score, 2)
        
        print(f"New Reputation Score: {self.reputation_scores[proposer_id]}")
        print("--------------------------------------------------\n")

    def propose_update(self, cluster_id, aggregated_model_hash, accuracy):
        """
        Bước 6: Cơ chế Đồng thuận + Gọi Smart Contract
        """
        proposer_id = f"ClusterHead_{cluster_id}"
        votes = 0
        valid_voters = []

        # 1. Giả lập Ủy ban đánh giá (Committee Voting)
        for member in self.committee:
            # Logic: Nếu độ chính xác > ngưỡng chấp nhận thì Vote Yes
            # (Giả sử member dùng tập validation riêng để check)
            if accuracy > 20.0: # Ngưỡng tối thiểu chấp nhận (hardcoded for sim)
                votes += 1
                valid_voters.append(member)
        
        # 2. Kiểm tra Đồng thuận
        required_votes = len(self.committee) * Config.CONSENSUS_THRESHOLD
        is_approved = votes >= required_votes
        
        # 3. KÍCH HOẠT SMART CONTRACT (Tính điểm)
        self.execute_smart_contract(proposer_id, accuracy, is_approved, valid_voters)
        
        # 4. Nếu đạt đồng thuận thì ghi Block
        if is_approved:
            block_data = {
                "cluster_id": cluster_id,
                "model_hash": aggregated_model_hash,
                "accuracy": accuracy,
                "reputation_snapshot": self.reputation_scores.copy() # Lưu vết điểm số vào Block
            }
            self.add_block(block_data)
            return True
        
        return False

    def add_block(self, data):
        prev_block = self.get_latest_block()
        new_block = Block(len(self.chain), prev_block.hash, data, time.time())
        self.chain.append(new_block)
        print(f"Block #{new_block.index} added to Ledger.")