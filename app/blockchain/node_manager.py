import random
import math
from config import Config
from app.core.worker import WorkerNode


class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.worker = WorkerNode(node_id, None)

class NodeManager:
    def __init__(self, blockchain):
        self.blockchain = blockchain 
        self.nodes = []
        self.proposer = None
        self.committee = []
        self.workers = []

    def create_nodes(self):
        """Tạo 10 node node1 → node10"""
        self.nodes = [
            Node(f"node{i}")
            for i in range(1, 11)
        ]


    def assign_roles(self):
        """Chọn proposer, committee, worker theo reputation"""
        proposer, committee, workers = self.choose_roles_smart()

        self.proposer = proposer
        self.committee = committee
        self.workers = workers

    def choose_roles_smart(self):
        # 1. Lọc node có ích
        positive_nodes = [n for n in self.nodes if self.blockchain.reputation_scores[n.node_id] > 0]
        target_pool = positive_nodes if positive_nodes else self.nodes
        # scores = [n.rep_score for n in target_pool]
        scores = [
            self.blockchain.reputation_scores[n.node_id]
            for n in target_pool
        ]


        avg = sum(scores) / len(scores) if scores else 0
        variance = sum((x - avg) ** 2 for x in scores) / len(scores) if scores else 0
        std_dev = math.sqrt(variance)

        candidate_threshold = avg + std_dev
        follower_threshold = avg

        candidates, followers, others = [], [], []

        for n in self.nodes:
            
            if self.blockchain.reputation_scores[n.node_id] >= candidate_threshold:
                candidates.append(n)
            elif self.blockchain.reputation_scores[n.node_id] >= follower_threshold:
                followers.append(n)
            else:
                others.append(n)

        # Fallback nếu không có candidate
        if not candidates:
            candidates = followers if followers else target_pool

        # Proposer
        proposer = random.choice(candidates)

        # Committee pool
        all_qualified = list(dict.fromkeys(candidates + followers))
        committee_pool = [n for n in all_qualified if n != proposer]

        if len(committee_pool) < Config.COMMITTEE_SIZE - 1:
            needed = Config.COMMITTEE_SIZE - 1 - len(committee_pool)
            sorted_others = sorted(
                [n for n in others if self.blockchain.reputation_scores[n.node_id] > 0],
                key=lambda x: self.blockchain.reputation_scores[x.node_id],
                reverse=True
            )

            committee_pool.extend(sorted_others[:needed])

        k = min(len(committee_pool), Config.COMMITTEE_SIZE - 1)
        committee = [proposer] + random.sample(committee_pool, k)

        # Workers = phần còn lại
        excluded = {n.node_id for n in committee}
        
        workers = [n for n in self.nodes if n.node_id not in excluded]
        return proposer, committee, workers

    def print_nodes(self):
        print("\n===== ROUND ROLE ASSIGNMENT =====")

        print("\nProposer:")
        rep = self.blockchain.reputation_scores[self.proposer.node_id]
        print(f" - {self.proposer.node_id} (rep={rep})")

        print("\nCommittee:")
        for n in self.committee:
            rep = self.blockchain.reputation_scores[n.node_id]
            print(f" - {n.node_id} (rep={rep})")

        print("\nWorkers:")
        for n in self.workers:
            rep = self.blockchain.reputation_scores[n.node_id]
            print(f" - {n.node_id} (rep={rep})")

        print("================================\n")

