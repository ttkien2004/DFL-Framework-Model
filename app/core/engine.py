# app/core/engine.py
import time
import random
import torch
from config import Config
from app.scenarios.factory import ScenarioFactory
from app.core.baseline_node import StandardDFLNode
from app.core.cluster_head import ClusterHead
from app.scenarios.scenario_4_sec import ScenarioExperiment4
from app.scenarios.scenario_1_baseline import ScenarioExperiment1
from app.utils.data_loader import get_global_test_loader
from app.utils.helpers import compute_model_norm
from app.blockchain.consensus import Blockchain
import copy
from collections import defaultdict
import math

class SimulationEngine:
    def __init__(self):
        """
        Khởi tạo Engine với các thành phần cốt lõi của hệ thống.
        """
        self.workers = []
        self.blockchain = None
        self.system_mode = 'PROPOSED'
        
        # Lưu trạng thái nội bộ
        self.current_dataset = Config.DATASET_NAME
        
        self.test_loader = None
        self.t_threshold = getattr(Config, 'T_THRESHOLD', 2)
        self.consensus_threshold = getattr(Config, 'CONSENSUS_THRESHOLD', 0.66) # 2/3 đồng ý
        
    
    def _init_state(self):
        """
        Hàm duy nhất chịu trách nhiệm khởi tạo/reset biến.
        Giúp code không bị lặp lại.
        """
        self.workers = []
        self.blockchain = None
        
        # Dùng defaultdict: Tự động tạo list rỗng cho bất kỳ key metric mới nào
        # Bạn không cần phải liệt kê "asr", "tpr", "avg_acc"... thủ công nữa
        # self.metrics_history = defaultdict(list)
        self.common_metrics = {}
        self.specific_metrics = {}
        
        # Các biến khác
        self.test_loader = None
        self.blockchain = None
    
    def reset_system(self):
        """
        Reset hệ thống chỉ bằng 1 dòng gọi lại _init_state
        """
        print("RESETTING SYSTEM (Clean State)...")
        self._init_state()
        
        # Giải phóng bộ nhớ GPU nếu dùng CUDA
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def initialize_system(self, req_data):
        """
        Khởi tạo hệ thống dựa trên mode.
        """
        # self.reset_system()
        self.system_mode = req_data.get('system_mode', 'PROPOSED')
        num_workers = req_data.get('num_workers', 10)
        config = req_data # Truyền full config
        dataset_name = req_data.get('dataset', Config.DATASET_NAME)
        batch_size = req_data.get('batch_size', Config.BATCH_SIZE)
        self.test_loader = get_global_test_loader(dataset_name, batch_size)
        worker_config = req_data 
        
        self.workers = []
        print(f"Initializing System in [{self.system_mode}] mode with {num_workers} workers.")

        for i in range(num_workers):
            if self.system_mode == 'BASELINE':
                # Tạo Node DFL thường
                worker = StandardDFLNode(i, config, Config.DEVICE)
            else:
                # Tạo Node Đề xuất (ClusterHead/Smart Worker)
                worker = ClusterHead(i, config, Config.DEVICE)
            
            self.workers.append(worker)

        # Nếu là BASELINE, cần thiết lập Topology tĩnh ngay từ đầu (VD: Random Graph)
        if self.system_mode == 'BASELINE':
            self._setup_static_topology(num_workers)
        else:
            # Chạy bầu cử lần đầu
            # Khởi tạo blockchain
            committee_config = {
                'all_nodes': self.workers,

            }
            self.blockchain = Blockchain()

            # Khởi tạo điểm ban đầu và lỗi cho tất cả nodes
            self.blockchain.initialize_faults(self.workers)
            self.blockchain.initialize_reputation(self.workers)

            self.proposer_node, self.committee_nodes, self.worker_nodes = self._elect_committee(current_round=0)
            committee_ids = [n.id for n in self.committee_nodes]

    def _setup_static_topology(self, num_workers, connectivity=0.5):
        """
        Tạo đồ thị ngẫu nhiên cho DFL thường (Topology cố định).
        """
        import random
        for w in self.workers:
            # Chọn ngẫu nhiên hàng xóm (trừ chính mình)
            candidates = [i for i in range(num_workers) if i != w.id]
            num_neighbors = max(2, int(num_workers * connectivity)) # Ít nhất 2 hàng xóm
            neighbors = random.sample(candidates, num_neighbors)
            w.set_neighbors(neighbors)
        print("Static Topology established for Baseline DFL.")

    def set_blockchain_ref(self, blockchain_ref):
        """
        Nhận tham chiếu Blockchain từ Main để các worker có thể tương tác.
        """
        self.blockchain = blockchain_ref

    def run_round(self, round_id, req_data):
        # self._setup_scenario(req_data)
        scenario_id = str(req_data.get('scenario', '1'))
        current_scenario = ScenarioExperiment4(self.workers,req_data)
        current_scenario.setup_security(self.workers)

        # 2. Khởi tạo và Cấu hình Scenario
        if scenario_id == '1':
            current_scenario = ScenarioExperiment1(self.workers, req_data)
            
            # QUAN TRỌNG: Chỉ chia lại dữ liệu (Dirichlet) ở vòng 0 hoặc khi có cờ reset
            # Nếu chạy mỗi vòng sẽ làm mất tính ổn định của Local Training
            should_reset_data = req_data.get('reset', False) or (round_id == 0)
            
            if should_reset_data:
                dataset_name = req_data.get('dataset', 'mnist')
                current_scenario.setup_data(self.workers, dataset_name)
                current_scenario.setup_network(self.workers)
            
            # Scenario 1 luôn là Clean (Security = NONE)
            current_scenario.setup_security(self.workers)
            attack_config = {}
        elif scenario_id == '4':
            cnt_mal = 0
            for node in self.workers:
                if node.is_malicious: cnt_mal += 1
            self.blocked_nodes = set()
            current_scenario = ScenarioExperiment4(self.workers, req_data)
            # Scenario 4 tập trung vào tấn công, cần setup mỗi vòng (hoặc tùy logic tấn công động)
            current_scenario.setup_security(self.workers)

            attack_config = {
                'attack_type': req_data.get('attack_type', 'NONE'),
                'source_class': req_data.get('source_class'),
                'target_class': req_data.get('target_class')
            }
            
            # Gọi hàm mới đã refactor            
        else:
            print(f"Unknown scenario {scenario_id}, defaulting to Security Scenario (4)")
            current_scenario = ScenarioExperiment4(self.workers, req_data)
            current_scenario.setup_security(self.workers)
        
        print(f"=== ROUND {round_id} START ({self.system_mode}) ===")

        execution_result = {}
        
        if self.system_mode == 'BASELINE':
            execution_result = self._run_round_baseline(round_id)
        else:
            execution_result = self._run_round_proposed(round_id)

        security_metrics = self._calculate_advanced_metrics(attack_config)
        # print("Exe result", execution_result.keys(), flush=True)

        final_result = {
            **execution_result,  # Bung toàn bộ kết quả vận hành (Logs, Topology...)
            **security_metrics,  # Bung toàn bộ chỉ số bảo mật (ASR, TER...)
            "round": round_id,
            "status": "success"
        }
        return final_result

    def _run_round_baseline(self, round_id):
        # Pha 1: Training & Attack (Tự động kích hoạt Attack nếu Scenario đã setup)
        print("   [Phase 1] Local Training...")
        for w in self.workers:
            w.train()

        # Pha 2: Gossip (Truyền tin)
        print("   [Phase 2] Gossiping...")
        count_gossips = 0
        for w in self.workers:
            # Gửi model cho hàng xóm
            payload = w.model.state_dict()
            for neighbor_id in w.neighbors:
                neighbor = self.workers[neighbor_id]
                neighbor.received_updates[w.id] = copy.deepcopy(payload)
                count_gossips += 1
        print(f"   -> Total Gossips Sent: {count_gossips}")
        # Pha 3: Aggregation
        print("   [Phase 3] Aggregation...")
        accuracies = []
        
        for w in self.workers:
            w.aggregate() # Gọi hàm aggregate của StandardDFLNode
            # Đánh giá nhanh (Optional)
            # acc = w.evaluate(self.test_loader)
            # accuracies.append(acc)
            
        # avg_acc = sum(accuracies) / len(accuracies)
        # print(f"[BASELINE] Round {round_id} finished. Avg Acc: {avg_acc:.4f}")
        return {
            "status": "aggregated",
            "topology": "Full Mesh / P2P",
            "hello": "WTF"
        }
    
    def _run_round_proposed(self, round_id):
        """
        Thực thi toàn bộ quy trình 1 vòng (Round) gồm 5 pha.
        """
        start_time = time.time()
        # scenario_id = req_data.get('scenario_id', 1)
        
        # print(f"\n--- ENGINE: STARTING ROUND {round_id} | SCENARIO {scenario_id} ---")

        # # 1. SETUP SCENARIO
        # # ----------------------------------------------------------------------
        # scenario_runner = ScenarioFactory.get_runner(scenario_id, req_data)
        # scenario_runner.apply(self.workers)
        
        # # Cập nhật dataset nếu thay đổi
        # req_dataset = req_data.get('dataset', 'cifar10')
        # self.current_dataset = req_dataset

        # 2. EXECUTE 5 PHASES
        # ----------------------------------------------------------------------
        proposer, committee, active_workers = self._elect_committee(round_id)
        # Proposer sẽ chịu trách nhiệm tạo block ở Phase Consensus
        self.current_proposer = proposer
        
        # Committee sẽ chịu trách nhiệm verify và giải mã (Secret Sharing)
        self.current_committee = committee
        # Phase 1: Clustering
        clusters = self._phase_clustering(workers_pools=active_workers,round_id=round_id)

        # Phase 2: Training & LDP
        worker_updates_cache = self._phase_training_ldp(clusters)

        # Phase 3: CoCo Optimization
        instruction_maps, all_topologies = self._phase_coco_optimization(clusters, worker_updates_cache)

        # Phase 4: Gossip & Aggregation
        round_results, real_accuracies = self._phase_aggregation(clusters=clusters, instruction_maps=instruction_maps, worker_updates_cache=worker_updates_cache, round_id=round_id)

        # Phase 5: Consensus
        consensus_log = self._phase_consensus(round_results)

        # 3. FINALIZE & RETURN METRICS
        # ----------------------------------------------------------------------
        execution_time = time.time() - start_time
        avg_accuracy = sum(real_accuracies) / len(real_accuracies) if real_accuracies else 0
        
        print(f"Round {round_id} finished inside Engine in {execution_time:.2f}s")

        return {
            "execution_time": execution_time,
            "avg_accuracy": avg_accuracy,
            # "topology": all_topologies,
            "logs": consensus_log,
            # "scenario": f"Scenario {scenario_id}",
            "reputation": self.blockchain.reputation_scores
        }

    # --- CÁC PHƯƠNG THỨC NỘI BỘ (PRIVATE METHODS) CHO TỪNG PHA ---

    def _phase_clustering(self, workers_pools, round_id):
        """
        Giai đoạn 1: Phân cụm động dựa trên Loss (Dynamic Clustering)
        """
        print(f"[Phase 1] Clustering (Round {round_id})...")
        
        num_clusters = getattr(Config, 'NUM_CLUSTERS', 5)
        cluster_models = {} # Dictionary chứa {cluster_id: state_dict}

        # --- BƯỚC 1: CHUẨN BỊ CENTROIDS (TÂM CỤM) ---
        if round_id == 0:
            # Round 0: Chưa có Blockchain, lấy model của K worker đầu tiên làm tâm cụm khởi tạo
            print(" -> Round 0: Initializing random centroids from first K workers.")
            for i in range(num_clusters):
                # Copy trọng số để tránh tham chiếu
                import copy
                cluster_models[i] = copy.deepcopy(self.workers[i].model.state_dict())
        else:
            # Round > 0: Tải centroids từ Blockchain
            cluster_models = self.blockchain.get_latest_centroids()
            
            # Fallback: Nếu Blockchain chưa có đủ (ví dụ lỗi), dùng lại chiến lược Round 0 hoặc giữ nguyên
            if not cluster_models:
                print(" -> Warning: No centroids found on Blockchain. Using local random init.")
                for i in range(num_clusters):
                    cluster_models[i] = copy.deepcopy(self.workers[i].model.state_dict())

        # --- BƯỚC 2: WORKER CHỌN CỤM (JOIN) ---
        # Reset trạng thái cũ
        clusters = {i: [] for i in range(num_clusters)}
        
        for w in workers_pools:
            w.is_head = False
            w.members = []
            
            # Worker tính loss trên từng centroid và chọn cái tốt nhất
            # Hàm join_cluster sẽ set w.cluster_id
            w.join_cluster(cluster_models)
            
            # Gom nhóm để bầu chọn Head sau này
            if w.cluster_id in clusters:
                clusters[w.cluster_id].append(w)
            else:
                # Trường hợp worker chọn cụm chưa có trong danh sách (ít xảy ra)
                clusters[w.cluster_id] = [w]

        # --- BƯỚC 3: BẦU CHỌN CLUSTER HEAD ---
        # Trong mỗi cụm, ta cần 1 người làm Head để tổng hợp
        # Chiến lược: Chọn người có Loss thấp nhất làm Head (Best Performance)
        # Hoặc đơn giản: Chọn người đầu tiên trong danh sách
        
        active_clusters_count = 0
        
        for cid, members in clusters.items():
            if not members:
                print(f" -> Cluster {cid} is empty!")
                continue
                
            active_clusters_count += 1
            
            # Chọn Head: Ở đây tôi chọn người có Loss thấp nhất trong cụm
            # (Giả sử join_cluster đã lưu loss vào biến w.current_loss, nếu chưa thì chọn random)
            # Cách đơn giản nhất: Chọn thành viên đầu tiên làm Head
            head_node = members[0] 
            
            head_node.is_head = True
            head_node.cluster_head_id = head_node.id # Chính nó là Head
            
            # Đăng ký các thành viên khác vào Head này
            for member in members:
                member.cluster_head_id = head_node.id
                if member.id != head_node.id:
                    # Nếu class Worker có hàm register_member
                    if hasattr(head_node, 'register_member'):
                        head_node.register_member(member.id)
                    else:
                        head_node.members.append(member.id)
            
            # Log thông tin
            print(f" -> Cluster {cid}: Head={head_node.id}, Size={len(members)}")

        print(f"-> Formed {active_clusters_count} active clusters.")
        
        # Trả về cấu trúc clusters để dùng cho các bước sau (Training/Aggregation)
        return clusters

    def _phase_training_ldp(self, clusters):
        print("[Phase 2] Training...")
        worker_updates_cache = {}
        
        # Duyệt qua các cụm để lấy danh sách worker active
        active_workers = []
        for cid, members in clusters.items():
            active_workers.extend(members)
            
        for w in active_workers:
            # Lưu ý: w.model đã được load weights của cụm ở Phase 1 (trong hàm join_cluster)
            
            # 1. Local Training
            w_new = w.train()
            
            # 2. Local Differential Privacy (Nếu có)
            w_dp = w.apply_ldp(w_new)
            
            # Lưu vào cache để lát nữa Head thu thập
            worker_updates_cache[w.id] = w_dp
            
        print(f" -> Trained {len(worker_updates_cache)} workers.")
        return worker_updates_cache

    def _phase_coco_optimization(self, clusters, worker_updates_cache):
        print("[Phase 3] CoCo Optimization...")
        instruction_maps = {}
        all_topologies = {}
        
        # B1: Report Metrics
        for cid, members in clusters.items():
            if not members: continue

            head_node = next((m for m in members if m.is_head), None)
            if not head_node:
                print(f" -> Cluster {cid} has no head! Skipping.")
                continue
            receivers_map = {head_node.id: []}
            topo_matrix = []
            for member in members:
                if member.id != head_node.id:
                    metrics = {
                        'bandwidth': random.uniform(10, 100), 
                        'latency': random.uniform(20, 200),
                        'cpu_load': random.uniform(10, 80)
                    }
                    head_node.receive_metrics(member.id, metrics, worker_updates_cache[member.id])
                    # receivers_map[head_node.id].append(member.id)

            topology_map, topology_viz = head_node.run_coco_optimization()
            instruction_maps[cid] = topology_map
            all_topologies[cid] = topology_viz

        # node_map = {node.id: node for node in self.workers}
        # for w in self.workers:
        #     # Hiện tại worker chưa có bandwidth
        #     metrics = {'bandwidth': random.uniform(20, 100), 'cpu_load': random.uniform(10, 80)}
        #     if w.cluster_head_id is not None:
        #         head_node = node_map.get(w.cluster_head_id)
        #         if head_node:
        #             head_node.receive_metrics(w.id, metrics, worker_updates_cache[w.id])
        
        # B2: Run CoCo
        # for ch in self.workers:
        #     head_node = node_map.get(w.cluster_head_id)

        #     instr, topo_viz = head_node.run_coco_optimization()
        #     instruction_maps.update(instr)
        #     all_topologies[head_node.cluster_id] = topo_viz
        print(f"[Debug CoCo] Instruction Maps: {instruction_maps}", flush=True)
        return instruction_maps, all_topologies

    def _phase_aggregation(self, clusters, instruction_maps, worker_updates_cache, round_id):
        print("[Phase 4] Aggregation...")
        worker_lookup = {w.id: w  
                         for cluster in clusters.values()
                         for w in cluster}
        cluster_heads = {node.cluster_id: node 
                         for cluster in clusters.values()
                         for node in cluster
                         if node.is_head}
        final_updates_for_ch = {}

        if not cluster_heads:
            print("Warning: No Cluster Heads found! Check _phase_clustering.")
            return [], []
        # Gossip Logic (DFCA)
        for ch_id, ch_instr in instruction_maps.items():
            for w_id, instr in ch_instr.items():
                receiver = worker_lookup.get(w_id)
                if not receiver: continue
                
                for neighbor_id in instr.get('neighbors', []):
                    if neighbor_id in worker_updates_cache:
                        sender_params = worker_updates_cache[neighbor_id]
                        sender_cluster = worker_lookup[neighbor_id].cluster_id
                        receiver.apply_dfca_gossip_update(sender_cluster, sender_params)
            
                final_updates_for_ch[w_id] = {k: v.cpu().clone() for k, v in receiver.model.state_dict().items()}

        # Send to CH
        for w_id, model_state in final_updates_for_ch.items():
            ch_id = worker_lookup[w_id].cluster_id
            cluster_heads[ch_id].receive_update(w_id, model_state)

        # Aggregate
        results = []
        accuracies = []
        for ch in cluster_heads.values():
            ch.set_committee_info(committee_config={
                't': self.t_threshold,
                'n': len(self.current_committee),
                "public_keys": {com.id: com.get_public_key() 
                                for com in self.current_committee}
            })
            aggregate_res = ch.aggregate(round_k=round_id)

            # Giả lập validate accuracy
            # simulated_acc = min(95.0, 15.0 + round_id * 2.5 + random.uniform(-2, 3))
            accuracies.append(0.0)
            results.append({"cluster_id": ch.cluster_id,"accuracy": 0.0, **aggregate_res})
            
        return results, accuracies

    # def _phase_aggregation(self, clusters, instruction_maps, worker_updates_cache, round_id):
    #     print("[Phase 4] Aggregation & Secret Sharing...")
    #     round_results = []
    #     real_accuracies = {} # {cluster_id: accuracy}
        
    #     for cid, members in clusters.items():
    #         if cid not in instruction_maps:
    #             continue
                
    #         # 1. Xác định Head
    #         head_node = next((m for m in members if m.is_head), None)
    #         if not head_node: continue
            
    #         # 2. Head thu thập models từ các member (Mô phỏng truyền tin)
    #         # instruction_maps[cid] có dạng {head_id: [member_id_1, member_id_2...]}
    #         senders_list = instruction_maps[cid].get(head_node.id, [])
            
    #         # Reset hàng đợi của Head trước khi nhận
    #         head_node.pending_models = []
            
    #         # Nạp update từ cache vào Head
    #         for sender_id in senders_list:
    #             if sender_id in worker_updates_cache:
    #                 w_update = worker_updates_cache[sender_id]
    #                 head_node.receive_update(noisy_params=w_update,worker_id=sender_id) # Hàm này append vào pending_models
            
    #         # Head cũng tự train
    #         if head_node.id in worker_updates_cache:
    #             head_node.receive_update(noisy_params=worker_updates_cache[head_node.id],worker_id=head_node.id)

    #         # 3. Thực hiện Aggregate (Đã bao gồm Filtering + Secret Sharing)
    #         # Hàm này trả về Dict chứa {metadata, encrypted_shares, model_hash, ...}

    #         head_node.set_committee_info(committee_config={
    #             't': self.t_threshold,
    #             'n': len(self.current_committee),
    #             "public_keys": {com.id: com.get_public_key() 
    #                             for com in self.current_committee}
    #         })
    #         agg_result = head_node.aggregate(round_k=round_id)
            
    #         # Gán thêm cluster_id để Blockchain biết của ai
    #         agg_result['cluster_id'] = cid
    #         agg_result['cluster_members'] = [m.id for m in members]
            
    #         round_results.append(agg_result)
            
    #     return round_results, real_accuracies

    def _phase_consensus(self, results):
        print("[Phase 5] Blockchain Consensus...")
        logs = []
        proposer = self.current_proposer
        committee = self.current_committee
        threshold = self.t_threshold
        consensus_threshold = self.consensus_threshold
        
        sorted_committee_ids = sorted([v.id for v in committee])
        id_to_index = {uid: i for i, uid in enumerate(sorted_committee_ids, start=1)}
        print(f"[Debug] Index Mapping for Reconstruction: {id_to_index}")
        for res in results:
            cluster_id = res['cluster_id']
            encrypted_shares_map = res.get('encrypted_shares')
            cluster_members = res.get('cluster_members', [])            

            if not encrypted_shares_map:
                continue            

            # --- BƯỚC 1: Validator (Committee) giải mã ---
            decrypted_shares = {}
            
            for validator in committee:
                # Kiểm tra xem có gói tin cho validator này không
                if validator.id in encrypted_shares_map:
                    enc_pkg = encrypted_shares_map[validator.id]
                    
                    # Validator tự dùng key của mình để giải mã
                    share = validator.decrypt_share(enc_pkg)
                    
                    if share is not None:
                        # decrypted_shares[validator.id] = share
                        correct_x = id_to_index.get(validator.id)
                        
                        if correct_x:
                            decrypted_shares[correct_x] = share
                        else:
                            print(f"Error: Validator {validator.id} not in sorted mapping!")
            
            # --- BƯỚC 2: Proposer tái tạo Model ---
            metadata = res['metadata']
            reconstructed_model = proposer.reconstruct_model(collected_shares=decrypted_shares, metadata=metadata,threshold=threshold)

            with open("rec_model.txt", "w", encoding="utf-8") as f:
                f.write(str(reconstructed_model))
                
            if reconstructed_model is None:
                logs.append(f"Cluster {cluster_id}: Consensus Failed (Reconstruction Error)")
                continue

            # --- Verify (Soft Check Norm/Hash) ---
            proposed_norm = res.get('model_norm')
            rec_norm = compute_model_norm(reconstructed_model)
            
            if abs(proposed_norm - rec_norm) > 1e-3:
                logs.append(f"The difference between two models are too much")
                continue
            else:
                logs.append(f"Two models are the same!")

            # Bước 3: COmmittee Validation
            votes, scores = self._run_committee_validation(committee=committee, model_state=reconstructed_model,val_loader=self.test_loader)

            total_votes = len(votes)
            approved_votes = sum(votes.values())
            is_approved = (approved_votes / total_votes) >= consensus_threshold

            # Tính điểm Acc trung bình của cả hội đồng
            final_acc = sum(scores.values()) / total_votes if total_votes > 0 else 0
            print(f" -> Consensus Result: {approved_votes}/{total_votes} votes. Approved? {is_approved}")

            # Bước 4: Gọi Smart Contract
            self.blockchain.execute_smart_contract(
                proposer_id=proposer.id,
                cluster_members=cluster_members,
                votes=votes,
                accuracy=final_acc,
                is_good_update=is_approved
            )

            if not is_approved:
                logs.append(f"Cluster {cluster_id}: Rejected (Vote Failed {approved_votes}/{total_votes})")
                continue

            # Bước 5: Tạo BLOCK mới
            storage_path = self.blockchain._save_model_offchain(reconstructed_model, cluster_id, res['model_hash'])

            last_block = self.blockchain.chain[-1]
            block_data = {
                "accuracy": final_acc,          # Dùng accuracy do ủy ban chấm
                "model_hash": res['model_hash'],
                "storage_uri": storage_path,
                "votes": votes,
                "previous_block": last_block
            }
            
            new_block = self.blockchain.add_block(
                block_data
            )
            success = self.blockchain.is_valid_new_block(new_block, last_block)
            status = "Accepted" if success else "Rejected (Blockchain Add Error)"
            logs.append(f"Cluster {cluster_id}: {status}")

        return logs
    
    def _run_committee_validation(self, committee, model_state, val_loader):
        """
        Chạy vòng lặp lấy ý kiến đánh giá của từng Validator        
        """
        votes = {}
        scores = {}
        for validator in committee:
            acc,vote = validator.validate_update(model_state, val_loader)
            votes[validator.id] = vote
            scores[validator.id] = acc
            print(f" -> Validator {validator.id}: Acc={acc:.4f} | Vote={vote}")

        return votes, scores
    
    # Các hàm tính toán chỉ số đánh giá cho từng kịch bản tấn công
    def _calculate_robustness_metrics(self):
        """
        Tính toán các chỉ số bảo mật chuyên sâu cho kịch bản tấn công.
        Chỉ tính trên các nút LÀNH TÍNH (Benign Nodes).
        """
        benign_workers = [w for w in self.workers if not w.is_malicious]
        
        if not benign_workers:
            return {}

        accuracies = []
        error_rates = []
        mses = []

        print(f"Evaluating {len(benign_workers)} benign workers...")
        
        for w in benign_workers:
            # Dùng test_loader toàn cục để đánh giá khách quan
            metrics = w.evaluate_detailed(self.test_loader)
            
            accuracies.append(metrics['accuracy'])
            error_rates.append(metrics['error_rate'])
            mses.append(metrics['mse'])

        # 1. Max.TER (Maximum Testing Error Rate)
        max_ter = max(error_rates)

        # 2. Average Test Accuracy
        avg_acc = sum(accuracies) / len(accuracies)

        # 3. Max.MSE (Maximum Mean Squared Error)
        max_mse = max(mses)
        
        # Trả về dictionary kết quả
        return {
            "benign_max_ter": max_ter,
            "benign_avg_acc": avg_acc,
            "benign_max_mse": max_mse
        }
    
    def _calculate_advanced_metrics(self, attack_config):
        """
        Hàm chính: Điều phối việc tính toán dựa trên loại tấn công.
        """
        attack_type = attack_config.get('attack_type', 'NONE')
        benign_workers = [w for w in self.workers if not w.is_malicious]
        
        if not benign_workers: return {}
        
        print(f"Calculating metrics for [{attack_type}] on {len(benign_workers)} benign nodes...")

        # 1. Tính các Metrics riêng biệt (Specific Metrics)
        self.specific_metrics = {}
        error_rates = [] # Cần thu thập để tính Max.TER
        accuracies = []  # Cần thu thập để tính Avg Acc

        if attack_type == 'LABEL_FLIPPING':
            src = attack_config.get('source_class')
            tgt = attack_config.get('target_class')
            
            # Container tạm
            asrs, src_recalls, tgt_precisions = [], [], []
            
            for w in benign_workers:
                m = w.evaluate_label_flipping(self.test_loader, src, tgt)
                
                accuracies.append(m['accuracy'])
                error_rates.append(m['error_rate'])
                asrs.append(m['asr'])
                src_recalls.append(m['src_recall'])
                tgt_precisions.append(m['tgt_precision'])
            
            # Tổng hợp
            self.specific_metrics = {
                "asr": sum(asrs) / len(asrs),
                "src_recall": sum(src_recalls) / len(src_recalls),
                "tgt_precision": sum(tgt_precisions) / len(tgt_precisions)
            }

        elif attack_type in ['GAUSSIAN', 'MODEL_POISONING']:
            # Với Gaussian, ta quan tâm MSE và Loss hơn là ASR
            mses = []
            losses = []
            
            for w in benign_workers:
                m = w.evaluate_gaussian_metrics(self.test_loader)
                
                accuracies.append(m['accuracy'])
                error_rates.append(m['error_rate'])
                mses.append(m['mse'])
                losses.append(m['loss'])
                
            self.specific_metrics = {
                "avg_mse": sum(mses) / len(mses),
                "avg_loss": sum(losses) / len(losses)
            }
        elif attack_type == 'BACKDOOR':
            tgt = attack_config.get('target_class')
            asrs = []
            
            print(f"   Using Target Class {tgt} for Backdoor Eval")

            for w in benign_workers:
                # Gọi hàm mới viết
                m = w.evaluate_backdoor(self.test_loader, target_class=tgt)
                
                accuracies.append(m['accuracy']) # Clean Accuracy
                error_rates.append(m['error_rate'])
                asrs.append(m['asr'])            # Backdoor Success Rate
            
            self.specific_metrics = {
                "asr": sum(asrs) / len(asrs),
                "clean_acc": sum(accuracies) / len(accuracies)
            }
        else:
            # Fallback cho trường hợp chạy bình thường (NONE)
            for w in benign_workers:
                m = w.evaluate_gaussian_metrics(self.test_loader) # Dùng hàm cơ bản
                accuracies.append(m['accuracy'])
                error_rates.append(m['error_rate'])

        # 2. Tính các Metrics chung (Common Metrics)
        self.common_metrics = self._calculate_common_metrics(benign_workers, error_rates, accuracies)

        # 3. Gộp kết quả (Merge dicts)
        return {**self.specific_metrics, **self.common_metrics}

    def _calculate_common_metrics(self, benign_workers, error_rates, accuracies):
        """
        Tính toán các chỉ số hệ thống dùng chung cho mọi kịch bản.
        Bao gồm: Max.TER, Consensus Error, TPR/FPR, Comm Cost.
        """
        # A. Max.TER & Avg Acc
        max_ter = max(error_rates) if error_rates else 0
        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0

        # B. Consensus Error (Độ phân tán của mô hình lành tính)
        benign_weights_flat = []
        for w in benign_workers:
            vec = torch.cat([p.view(-1).float() for p in w.model.state_dict().values()])
            benign_weights_flat.append(vec)
            
        if benign_weights_flat:
            stack = torch.stack(benign_weights_flat)
            mean_vec = torch.mean(stack, dim=0)
            dists = torch.norm(stack - mean_vec, dim=1)
            consensus_error = torch.mean(dists).item()
        else:
            consensus_error = 0.0

        # C. Detection Metrics
        tpr, fpr = self._calculate_detection_rate()

        # D. Comm Cost (Ước tính)
        model_size_mb = 1.2 
        comm_cost = len(self.workers) * model_size_mb

        return {
            "max_ter": max_ter,
            "avg_acc": avg_acc,
            "consensus_error": consensus_error,
            "tpr": tpr,
            "fpr": fpr,
            "comm_cost": comm_cost
        }
    def _calculate_detection_rate(self):
        """
        Tính TPR (True Positive Rate) và FPR (False Positive Rate)
        Dựa trên danh sách node bị hệ thống chặn/cô lập.
        """
        # Ground Truth
        actual_malicious = set([w.id for w in self.workers if w.is_malicious])
        actual_benign = set([w.id for w in self.workers if not w.is_malicious])
        
        # System Output (Predicted Malicious)
        # Logic này phụ thuộc vào thuật toán của bạn. 
        # Ví dụ: Trong DFCA, những node KHÔNG được chọn vào aggregation list cuối cùng.
        # Hoặc những node có trust score thấp.
        # Ở đây tôi giả định bạn có biến self.blocked_nodes lưu danh sách bị chặn trong vòng này.
        predicted_malicious = getattr(self, 'blocked_nodes', set())
        
        if not actual_malicious: return 1.0, 0.0 # Không có tấn công

        # TPR = TP / (TP + FN) -> Tỷ lệ node xấu bị bắt
        tp = len(predicted_malicious.intersection(actual_malicious))
        tpr = tp / len(actual_malicious) if actual_malicious else 0
        
        # FPR = FP / (FP + TN) -> Tỷ lệ node tốt bị bắt nhầm
        fp = len(predicted_malicious.intersection(actual_benign))
        fpr = fp / len(actual_benign) if actual_benign else 0
        
        return tpr, fpr
    
    def _elect_committee(self, current_round):
        """
        Bầu chọn Proposer và Committee dựa trên Reputation (Smart Role Selection).
        Logic dựa trên độ lệch chuẩn (Standard Deviation) của điểm uy tín.
        """
        print(f"[Election] Electing roles for Round {current_round}...")
        # 1. Lấy điểm uy tín từ Blockchain
        # Giả sử blockchain.reputation_scores là dict {node_id: score}        
        rep_scores = self.blockchain.reputation_scores
        
        # 2. Lọc các node "Positive" (Có ích)
        positive_nodes = [
            n for n in self.workers 
            if rep_scores.get(n.id, 0) > Config.MIN_REPUTATION
        ]
        
        # Nếu chưa ai có uy tín (vòng đầu), lấy tất cả làm pool
        target_pool = positive_nodes if positive_nodes else self.workers
        
        # 3. Tính toán thống kê (Avg, StdDev)
        scores = [rep_scores.get(n.id, 0) for n in target_pool]
        
        if scores:
            avg = sum(scores) / len(scores)
            variance = sum((x - avg) ** 2 for x in scores) / len(scores)
            std_dev = math.sqrt(variance)
        else:
            avg, std_dev = 0, 0
            
        candidate_threshold = avg + std_dev # Ngưỡng cao cho Proposer
        follower_threshold = avg            # Ngưỡng trung bình cho Committee

        # 4. Phân loại node
        candidates = [] # Nhóm tiềm năng làm Proposer
        followers = []  # Nhóm tiềm năng làm Committee
        others = []     # Nhóm còn lại (Worker thường)

        for n in self.workers:
            score = rep_scores.get(n.id, 0)
            if score >= candidate_threshold:
                candidates.append(n)
            elif score >= follower_threshold:
                followers.append(n)
            else:
                others.append(n)

        # Fallback: Nếu không có candidate (do điểm bằng nhau hết), lấy nhóm follower
        if not candidates:
            candidates = followers if followers else target_pool

        # 5. Chọn Proposer (Random trong nhóm ưu tú nhất)
        proposer = random.choice(candidates)

        # 6. Chọn Committee (Validator Pool)
        # Gom cả candidates (trừ proposer) và followers
        all_qualified = list(dict.fromkeys(candidates + followers))
        committee_pool = [n for n in all_qualified if n.id != proposer.id]

        # Đảm bảo đủ số lượng Committee (lấy thêm từ others nếu thiếu)
        target_committee_size = getattr(Config, 'COMMITTEE_SIZE', 5) - 1 # Trừ 1 slot của Proposer
        
        if len(committee_pool) < target_committee_size:
            needed = target_committee_size - len(committee_pool)
            # Sắp xếp others theo uy tín giảm dần để lấy người khá nhất trong nhóm tệ
            sorted_others = sorted(
                [n for n in others if rep_scores.get(n.id, 0) > 0],
                key=lambda x: rep_scores.get(x.id, 0),
                reverse=True
            )
            committee_pool.extend(sorted_others[:needed])

        # Random chọn trong pool để chốt danh sách
        k = min(len(committee_pool), target_committee_size)
        committee = random.sample(committee_pool, k)

        # 7. Worker là phần còn lại
        # Gom Proposer và Committee thành set để loại trừ
        excluded_ids = {proposer.id} | {n.id for n in committee}
        workers = [n for n in self.workers if n.id not in excluded_ids]

        print(f" -> Proposer: {proposer.id}")
        print(f" -> Committee ({len(committee)}): {[n.id for n in committee]}")
        print(f" -> Workers ({len(workers)}): Count only")

        return proposer, committee, workers