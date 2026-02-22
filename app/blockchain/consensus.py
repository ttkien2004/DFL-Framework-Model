# Logic của Cơ chế đồng thuận ủy ban
import time
import hashlib
import json
from wsgiref.validate import validator
from config import Config
from app.blockchain.block import Block
from app.utils.secret_sharing import SecretSharingUtils
from app.utils.crypto import CryptoUtils

from app.blockchain.validator import Validator
from app.blockchain.data_loader import get_global_val_loader
import torch
import os

class Blockchain:
    def __init__(self,storage_dir="./chain_storage"):
        self.chain = [self.create_genesis_block()]
        self.committee = []
        self.reputation_scores = {}
        self.fault = {}
        self.storage_dir = storage_dir
        # all_nodes = committee_config.get('workers', [])
        os.makedirs(storage_dir, exist_ok=True)
        # self._initialize_reputation(all_nodes)
        # self._initialize_faults(all_nodes)
        self.current_round_k_models = {}
    
    def initialize_reputation(self, all_nodes):
        """Khởi tạo điểm cho các node tham gia"""
        for node in all_nodes:
            node_id = node.id if hasattr(node, 'id') else node

            self.reputation_scores[node_id] = Config.INITIAL_REPUTATION
    
    def initialize_faults(self, all_nodes):
        for node in all_nodes:
            node_id = node.id if hasattr(node, 'id') else node
            self.fault[node_id] = 0

    # Các hàm hỗ trợ lưu và tải k-models
    def _save_model_offchain(self, model_state_dict, cluster_id, model_hash):
        """
        Lưu model xuống đĩa và trả về đường dẫn.
        """
        filename = f"cluster_{cluster_id}_ver_{model_hash[:8]}.pth"
        path = os.path.join(self.storage_dir, filename)
        
        torch.save(model_state_dict, path)
        print(f"[Storage] Model saved to {path}")
        return path
    
    def load_model_from_path(self, file_path):
        if not os.path.exists(file_path):
            print(f"[Storage] File not found: {file_path}")
            return None
        try:
            return torch.load(file_path,map_location=torch.device('cpu'))
        except Exception as e:
            print(f"[Storage] Error loading model: {e}")
            return None
        
    def update_global_models_registry(self, cluster_models_map):
        """
        Transaction cập nhật danh sách K-model cho vòng mới.
        Input: {0: "QmHash0...", 1: "QmHash1..."}
        """
        self.current_round_k_models = cluster_models_map
        print(f"[Smart Contract] Registry Updated for Next Round: {cluster_models_map}")
    
    def get_latest_k_model_hashes(self):
        return self.current_round_k_models

    def create_genesis_block(self):
        return Block(0, "0", [], time.time())

    def get_latest_block(self):
        return self.chain[-1]
    
    def get_latest_k_models(self, num_clusters):
        """
        Quét ngược Blockchain để tìm model mới nhất cho từng Cluster ID.
        Trả về dict: {cluster_id: model_state}
        """
        latest_models = {}
        found_clusters = set()
        for block in reversed(self.chain):
            if block.index == 0: continue
            data = block.data
            cid = data.get('cluster_id')
            if cid is not None and cid not in found_clusters:
                path = data.get('storage_uri')
                if path:
                    model_state = self.load_model_from_path(path)
                    if model_state:
                        latest_models[cid] = model_state
                        found_clusters.add(cid)
            if len(found_clusters) == num_clusters:
                break
        return latest_models
    #####################################################
    
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
            v.id: Validator(node_id=v.id,config=v.config,device=v.device)
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
        print("Prev Block is None?", prev_block, flush=True)
        if prev_block is None:
            self.chain = [self.create_genesis_block()]
            prev_block = self.get_last_block()
        
        new_block = Block(len(self.chain), prev_block.hash, data, time.time())
        self.chain.append(new_block)
        print(f"Block #{new_block.index} added to Ledger.")

    def is_valid_new_block(self, new_block, previous_block):
        """
        Kiểm tra tính hợp lệ của Block mới trước khi thêm vào chuỗi.
        """
        # 1. Kiểm tra Index (Phải tăng tịnh tiến 1 đơn vị)
        if previous_block.index + 1 != new_block.index:
            print(f"[Blockchain Error] Invalid Index: Expected {previous_block.index + 1}, got {new_block.index}")
            return False

        # 2. Kiểm tra Previous Hash (Mắt xích quan trọng nhất)
        if previous_block.hash != new_block.previous_hash:
            print(f"[Blockchain Error] Invalid Previous Hash link!")
            return False

        # 3. Kiểm tra tính toàn vẹn (Re-hash)
        # Tính lại hash của new_block xem có khớp với hash đang lưu trong nó không
        # Điều này đảm bảo dữ liệu (data/timestamp) không bị ai đó sửa đổi trộm sau khi Block được tạo
        recalculated_hash = new_block.compute_hash()
        
        if new_block.hash != recalculated_hash:
            print(f"[Blockchain Error] Invalid Block Hash (Data tampered)!")
            return False

        # (Tùy chọn) 4. Kiểm tra Timestamp
        # Block mới không được sinh ra trước Block cũ (ngăn chặn tấn công thời gian)
        if new_block.timestamp < previous_block.timestamp:
            print(f"[Blockchain Error] Backdated timestamp! New block is older than previous block.")
            return False

        return True
    
    # Dùng cho cơ chế VIEW-CHANGE
    def penalize_node(self, node_id, penalty=1):
        """
        Trừng phạt node bằng cách tăng chỉ số fault.
        """
        # Đảm bảo node_id tồn tại trong dict
        if node_id not in self.fault:
            self.fault[node_id] = 0
            
        self.fault[node_id] += Config.PENALTY
        print(f"[Blockchain] Node {node_id} penalized! Fault count: {self.fault[node_id]}")