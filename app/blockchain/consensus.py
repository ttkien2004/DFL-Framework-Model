# Logic của Cơ chế đồng thuận ủy ban
import time
import hashlib
import json
from wsgiref.validate import validator
from config import Config
from app.blockchain.block import Block

from app.blockchain.validator import Validator
from app.blockchain.data_loader import get_global_val_loader


class Blockchain:
    def __init__(self, committee, all_nodes):
        self.chain = [self.create_genesis_block()]
        self.committee = committee
        self.reputation_scores = {}
        self.fault = {}
        
        self._initialize_reputation(all_nodes)
        self._initialize_faults(all_nodes)
    
    def _initialize_reputation(self, all_nodes):
        """Khởi tạo điểm cho các node tham gia"""
        for node in all_nodes:
            self.reputation_scores[node] = Config.INITIAL_REPUTATION
    
    def _initialize_faults(self, all_nodes):
        for node in all_nodes:
            self.fault[node] = 0

    def create_genesis_block(self):
        return Block(0, "0", "Genesis Model", time.time())

    def get_latest_block(self):
        return self.chain[-1]

    # def propose_update(self, cluster_updates):
    #     """Bước 6: Cơ chế Đồng thuận (Consensus)"""
    #     # Giả lập Committee voting
    #     votes = 0
    #     for member in self.committee:
    #         # Logic kiểm tra model trên tập validation của member
    #         votes += 1 

    #     # Tính tỷ lệ phiếu bầu cần thiết
    #     required_votes = len(self.committee) * Config.CONSENSUS_THRESHOLD
        
    #     if votes > required_votes:
    #         self.add_block(cluster_updates)
    #         return True
    #     return False
    
    # --- SMART CONTRACT LOGIC ---
    def execute_smart_contract(
        self,
        proposer_id,
        accuracy,
        is_good_update,
        votes,
        cluster_members
    ):
        print("\n--- SMART CONTRACT V2 ---")

        # === 1. Thưởng/Phạt cho CLUSTER HEAD (PROPOSER) ===
        if is_good_update:
            self.reputation_scores[proposer_id] += accuracy
        else:
            self.reputation_scores[proposer_id] -= Config.PENALTY

        # === 2. Thưởng/Phạt cho THÀNH VIÊN CỤM (CLUSTER MEMBERS) ===
        for member in cluster_members:
            if member == proposer_id:
                continue  # tránh cộng trùng CH

            if is_good_update:
                self.reputation_scores[member] += accuracy
            else:
                self.reputation_scores[member] -= Config.PENALTY

        # === 3. Thưởng/Phạt cho VALIDATORS (COMMITTEE) ===
        # Chia đều điểm thưởng cho các validator
        Scom = 1 / max(1, Config.NUM_CLUSTERS)
        for validator, vote in votes.items():
            if vote is None:
                continue  # không vote → không thưởng phạt
            # vote đúng
            if vote == is_good_update:
                self.reputation_scores[validator] += Scom * accuracy
                print(f"Validator {validator} rewarded +{Scom * accuracy}")

            # vote sai
            else:
                # vote cho model xấu -> phạt nặng
                if vote and not is_good_update:
                    # self.fault[validator] += 1
                    self.reputation_scores[proposer_id] -= Config.PENALTY
                    # print(f"Validator {validator} heavy penalty (Fault +1)")

                # từ chối model tốt -> phạt nhẹ
                elif not vote and is_good_update:
                    # self.fault[validator] += 0.5
                    self.reputation_scores[proposer_id] -= Config.PENALTY / 2
                    # print(f"Validator {validator} light penalty (Fault +0.5)")

        print("--- END SMART CONTRACT ---\n")


    # def execute_smart_contract(self, proposer_id, validation_accuracy, is_approved, voters):
    #     """
    #     Hàm này đóng vai role là Smart Contract:
    #     Tự động tính toán và cập nhật điểm dựa trên kết quả đồng thuận.
    #     """
    #     print(f"\n---Executing Smart Contract for {proposer_id} ---")
        
    #     current_score = self.reputation_scores.get(proposer_id, Config.INITIAL_REPUTATION)
        
    #     if is_approved:
    #         # 1. Tính thưởng cho Proposer (Cluster Head)
    #         # Công thức: Điểm cũ + Thưởng cơ bản + (Độ chính xác * Hệ số)
    #         acc_bonus = validation_accuracy * Config.ACCURACY_BONUS_FACTOR
    #         new_score = current_score + Config.REWARD_SUCCESSFUL_BLOCK + acc_bonus
            
    #         print(f"Block Approved! {proposer_id} gained points.")
    #         print(f"   Reward: {Config.REWARD_SUCCESSFUL_BLOCK}, Acc Bonus: {acc_bonus:.2f}")

    #         # 2. Thưởng cho các thành viên Ủy ban (Voters) đã làm việc
    #         for voter in voters:
    #             v_score = self.reputation_scores.get(voter, Config.INITIAL_REPUTATION)
    #             self.reputation_scores[voter] = v_score + Config.REWARD_COMMITTEE_VOTE
                
    #     else:
    #         # Phạt Proposer vì gửi model kém chất lượng
    #         new_score = current_score + Config.PENALTY_REJECTED_BLOCK
    #         print(f"Block Rejected! {proposer_id} penalized.")

    #     # 3. Áp dụng Decay (Giảm nhẹ điểm của tất cả để tránh lạm phát điểm)
    #     # (Tùy chọn: chỉ giảm những người không hoạt động, ở đây ta đơn giản hóa)
    #     new_score = new_score * Config.DECAY_FACTOR
        
    #     # Cập nhật vào bảng điểm
    #     self.reputation_scores[proposer_id] = round(new_score, 2)
        
    #     print(f"New Reputation Score: {self.reputation_scores[proposer_id]}")
    #     print("--------------------------------------------------\n")

    def propose_update(self, proposer_id, aggregated_model, cluster_members):
        votes = {}
        scores = {}

        # val_loader = get_global_val_loader()
        # validator = Validator(val_loader)

        
        validators = {
            v: Validator(v)
            for v in self.committee
        }

        print("\n--- COMMITTEE VALIDATION ---")

        for vid, validator in validators.items():
            acc = validator.evaluate(aggregated_model)
            vote = acc >= Config.ACC_THRESHOLD
            votes[vid] = vote
            scores[vid] = acc 
            print(f"Validator {vid}: acc={acc:.2f}% → vote={vote}")

        approved_votes = sum(votes.values())
        required_votes = len(votes) * Config.CONSENSUS_THRESHOLD
        is_approved = approved_votes >= required_votes

        final_acc = sum(scores.values()) / len(scores)
        is_good_update = final_acc >= Config.ACC_THRESHOLD
        self.execute_smart_contract(
            proposer_id=proposer_id,
            accuracy=final_acc,
            is_good_update=is_good_update,
            votes=votes,
            cluster_members=cluster_members
        )
        if is_approved:
            self.add_block({
                "proposer": proposer_id,
                "accuracy": final_acc,
                "votes": votes
            })
            return True
        return False




    def add_block(self, data):
        prev_block = self.get_latest_block()
        new_block = Block(len(self.chain), prev_block.hash, data, time.time())
        self.chain.append(new_block)
        print(f"Block #{new_block.index} added to Ledger.")