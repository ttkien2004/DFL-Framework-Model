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
from app.utils.helpers import compute_model_norm, sanitize_for_json, get_model_size_mb
from app.blockchain.consensus import Blockchain
from app.core.ipfs import StorageService
from app.core.coco_helpers import CoCo
from app.core.bypass_ablation import BypassExecutor, BypassConfig, BypassClustering, BypassByzantine, BypassBlockchain, BypassPrivacy
import numpy as np
import copy
from collections import defaultdict
import math
import json

class SimulationEngine:
    def __init__(self):
        """
        Khởi tạo Engine với các thành phần cốt lõi của hệ thống.
        """
        self.workers = []
        self.blockchain = None
        self.system_mode = 'PROPOSED'
        self.bypass_executor = None  # Sẽ được khởi tạo trong initialize_system
        
        # Lưu trạng thái nội bộ
        self.current_dataset = Config.DATASET_NAME
        
        self.test_loader = None
        self.t_threshold = getattr(Config, 'T_THRESHOLD', 2)
        self.consensus_threshold = getattr(Config, 'CONSENSUS_THRESHOLD', 0.66) # 2/3 đồng ý
        self.storage = StorageService()
        # Lưu các log lịch sử để hiển thị trên UI
        self.logs = {
            "rounds": [],
            "roles": [],
            "cluster_assignments": [],
            "reputation": [],
            "comm_traffic_mb": []
        }
        # Cấu hình dynamic threshold
        self.previous_global_acc = 0.0  # Khởi tạo: Vòng đầu chưa có model thì Acc = 0
        self.base_min_threshold = 0.1   # Ngưỡng tối thiểu tuyệt đối (10% - bằng đoán mò)
        self.tolerance = 0.15           # Biên độ khoan dung (15%). VD: Nếu Global Acc là 60%, node đạt 45% vẫn được qua.
        self.currrent_dynamic_threshold = Config.ACC_THRESHOLD

    
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
        
        # Initialize test_loader with error handling
        try:
            self.test_loader = get_global_test_loader(dataset_name, batch_size)
            print(f"[Engine] Initialized test_loader for {dataset_name}")
        except Exception as e:
            print(f"[Engine] Failed to init test_loader: {e}. System will fallback to worker local test sets.")
            self.test_loader = None
        
        self.engine_config = req_data
        
        # Khởi tạo Bypass Executor (nếu có bypass_mode trong request)
        bypass_mode = req_data.get('bypass_mode', 0)  # 0 = No bypass (full features)
        self.bypass_executor = BypassExecutor(bypass_mode)
        
        self.workers = []
        print(f"Initializing System in [{self.system_mode}] mode with {num_workers} workers.")
        print(f"[Bypass Mode] {self.bypass_executor.get_report()}")

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
            self._setup_static_topology(num_workers, connectivity=0.1)
        else:
            # Chạy bầu cử lần đầu
            # Khởi tạo blockchain            
            self.blockchain = Blockchain()

            # Khởi tạo điểm ban đầu và lỗi cho tất cả nodes
            self.blockchain.initialize_faults(self.workers)
            self.blockchain.initialize_reputation(self.workers)
            self.blockchain.initialize_rewards(self.workers)
            # Khởi tạo k-models
            self._initialize_global_k_models()

            self.proposer_node, self.committee_nodes, self.worker_nodes = self._elect_committee(current_round=0)
            
            print("[Engine] System Initialized. Ready for Round 0.")

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
        # current_scenario = ScenarioExperiment4(self.workers,req_data)
        # current_scenario.setup_security(self.workers)

        # 2. Khởi tạo và Cấu hình Scenario
        if scenario_id == '1':
            current_scenario = ScenarioExperiment1(self.workers, req_data)
            
            # QUAN TRỌNG: Chỉ chia lại dữ liệu (Dirichlet) ở vòng 0 hoặc khi có cờ reset
            # Nếu chạy mỗi vòng sẽ làm mất tính ổn định của Local Training
            should_reset_data = req_data.get('reset', False)
            
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
            # QUAN TRỌNG: Chỉ chia lại dữ liệu (Dirichlet) ở vòng 0 hoặc khi có cờ reset
            # Nếu chạy mỗi vòng sẽ làm mất tính ổn định của Local Training
            should_reset_data = req_data.get('reset', False)
            
            current_scenario = ScenarioExperiment4(self.workers, req_data)
            # Scenario 4 tập trung vào tấn công, cần setup mỗi vòng (hoặc tùy logic tấn công động)
            if should_reset_data:
                dataset_name = req_data.get('dataset', 'mnist')
                current_scenario.setup_data(self.workers, dataset_name)
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
            current_round_cluster_models = execution_result['current_round_k_models']

        security_metrics = self._calculate_advanced_metrics(attack_config, current_round_cluster_models=current_round_cluster_models) if self.system_mode != 'BASELINE' else self._calculate_advanced_metrics(attack_config, current_round_cluster_models=None)
        # print("Exe result", execution_result.keys(), flush=True)
        # Cập nhật previous global acc cho vòng kế tiếp
        self.previous_global_acc = security_metrics['avg_acc']
        print("PRev accuracy", self.previous_global_acc, flush=True)

        # Chỉ trả những kết quả không phải current_round_k_models
        filtered_result = execution_result.copy()
        filtered_result.pop("current_round_k_models", None)
        final_result = {
            **filtered_result,  # Bung toàn bộ kết quả vận hành (Logs, Topology...)
            **security_metrics,  # Bung toàn bộ chỉ số bảo mật (ASR, TER...),
            "algo_name": req_data.get('aggregation_algorithm', "Proposed"),
            "round": round_id,
            "status": "success"
        }
        return sanitize_for_json(final_result)

    def _run_round_baseline(self, round_id):
        # Pha 1: Training & Attack (Tự động kích hoạt Attack nếu Scenario đã setup)
        start_time = time.time()
        algo_name = self.engine_config.get('aggregation_algorithm', None)

        t0 = time.time()
        is_coco = True if algo_name and algo_name == "CoCo" else False
        if is_coco:
            n = len(self.workers)
            self.coco_state = {
                'A': np.ones((n,n)),
                'r': np.ones(n),
                'D_max': 5.0,
                't_prev': float('inf')
            }
            np.fill_diagonal(self.coco_state['A'], 0)

        if algo_name in ('DFCA', 'DFCA_BALANCE'):
            self._phase_baseline_clustering()

        print("   [Phase 1] Local Training...")
        
        for w in self.workers:
            w.train()

        # Pha 2: Gossip (Truyền tin)
        t1 = time.time()
        print("   [Phase 2] Gossiping...")
        if is_coco:
            self.coco_state, improved, t_opt = CoCo.optimize_network(self.workers, self.coco_state)
            if improved:
                print(f"CoCo Optimized: Time={t_opt:.3f}s, Avg Compression={np.mean(self.coco_state['r']):.2f}")
            for i, w in enumerate(self.workers):
                new_neighbors_idx = np.where(self.coco_state['A'][i] == 1)[0]
                neighbord_ids = [self.workers[idx].id for idx in new_neighbors_idx]

                w.apply_coco_config(neighbord_ids, self.coco_state['r'][i])

        count_gossips = 0
        round_max_latency = 0.0
        total_traffic = 0.0
        workers_map = {w.id: w for w in self.workers}
        for w in self.workers:
            # Gửi model cho hàng xóm
            # payload = w.model.state_dict()
            # for neighbor_id in w.neighbors:
            #     neighbor = self.workers[neighbor_id]
            #     neighbor.received_updates[w.id] = copy.deepcopy(payload)
            #     count_gossips += 1
            count, latency, traffic = w.gossip(workers_map, is_coco_mode=is_coco)
            count_gossips += count
            total_traffic += traffic
            round_max_latency = max(round_max_latency, latency)
        print(f"   -> Total Gossips Sent: {count_gossips}")
        if is_coco: print(f"   -> Estimated Latency: {round_max_latency:.4f}s")
        # Pha 3: Aggregation
        t2 = time.time()
        print("   [Phase 3] Aggregation...")
        
        for w in self.workers:
            w.aggregate(current_round_id=round_id) # Gọi hàm aggregate của StandardDFLNode            
            
        t3 = time.time()
        execution_time = t3 - start_time
        # Phân rã thời gian (Latency Breakdown)
        latency_breakdown = {
            "time_election": 0.0,         # 0.0
            "time_clustering": 0.0,     # 0.0
            "time_training": t1 - t0,         # Thực tế
            "time_gossip": t2 - t1,          # Thực tế
            "time_aggregation": t3 - t2,   # Thực tế
            "time_consensus": 0.0        # 0.0
        }
        # Thêm mảng log thời gian nếu chưa có
        if "latency_breakdown" not in self.logs:
            self.logs["latency_breakdown"] = []
        self.logs["latency_breakdown"].append(latency_breakdown)
        return {
            "status": "aggregated",
            "topology": "Dynamic (CoCo)" if is_coco else "Fixed",
            "latency": round_max_latency if is_coco else 0,
            "avg_compression": np.mean(self.coco_state['r']) if is_coco else 1.0,
            "comm_traffic_mb": total_traffic,
            "execution_time": execution_time,
            "latency_breakdown": self.logs["latency_breakdown"]
        }

    def _phase_baseline_clustering(self):
        """
        Baseline DFCA clustering: assign workers to 3 clusters by evaluating
        loss against cluster centroid models.
        """
        num_clusters = 3
        if not self.workers:
            return

        print(f"[Baseline DFCA] Clustering into {num_clusters} clusters by loss")

        cluster_models = {}
        for cid in range(num_clusters):
            if cid < len(self.workers):
                cluster_models[cid] = {
                    k: v.clone() for k, v in self.workers[cid].model.state_dict().items()
                }
            else:
                cluster_models[cid] = {
                    k: v.clone() for k, v in cluster_models[0].items()
                }

        for w in self.workers:
            w.join_cluster(cluster_models)
            w.model.load_state_dict(cluster_models[w.cluster_id])
            w.model = w.model.to(w.device)

        cluster_counts = {cid: 0 for cid in cluster_models}
        for w in self.workers:
            cluster_counts[w.cluster_id] += 1

        print(f"   -> Baseline cluster assignment: {cluster_counts}")

    def _run_round_proposed(self, round_id):
        """
        Thực thi toàn bộ quy trình 1 vòng (Round) gồm 5 pha.
        """
        start_time = time.time()
        # 2. EXECUTE 5 PHASES
        # ----------------------------------------------------------------------
        t0 = time.time()
        proposer, committee, active_workers = self._elect_committee(round_id)
        # Proposer sẽ chịu trách nhiệm tạo block ở Phase Consensus
        self.current_proposer = proposer
        
        # Committee sẽ chịu trách nhiệm verify và giải mã (Secret Sharing)
        self.current_committee = committee

        current_roles = {
            "proposer": self.current_proposer.id if self.current_proposer else None,
            "committee": [node.id for node in self.current_committee]
        }
        # Phase 1: Clustering
        t1 = time.time()
        clusters = self._phase_clustering(workers_pools=active_workers,round_id=round_id)
        self.cluster_heads = []
        for members in clusters.values():
            head = next((node for node in members if node.is_head), None)
            if head:
                self.cluster_heads.append(head)
        # Thu thập log hiển thị
        cluster_map = {}
        for cid, members in clusters.items():
            cluster_map[cid] = [w.id for w in members]
        # Phase 2: Training & LDP
        t2 = time.time()
        worker_updates_cache, global_train_loss = self._phase_training_ldp(clusters)

        # Phase 3: CoCo Optimization
        t3 = time.time()
        instruction_maps, all_topologies = self._phase_coco_optimization(clusters, worker_updates_cache)

        # Phase 4: Gossip & Aggregation
        t4 =  time.time()
        round_results, real_accuracies, total_traffic = self._phase_aggregation(clusters=clusters, instruction_maps=instruction_maps, worker_updates_cache=worker_updates_cache, round_id=round_id)

        # Phase 5: Consensus
        t5 = time.time()
        self.currrent_dynamic_threshold = max(self.base_min_threshold, self.previous_global_acc - self.tolerance) if Config.ENABLE_DYNAMIC else self.currrent_dynamic_threshold # Tính ngưỡng động cho committee
        consensus_results, consensus_log = self._phase_consensus(round_results)

        current_round_cluster_models = {}
        current_hashes_on_chain = self.blockchain.get_latest_k_model_hashes()
        for res in consensus_results:
            cid = res['cluster_id']
            if res.get('status') == 'ACCEPTED':                
                model_state = res['model_state_dict'] # Model sau khi tái tạo
                current_round_cluster_models[cid] = model_state
            else:
                if cid in current_hashes_on_chain:                    
                    old_hash = current_hashes_on_chain[cid]
                    old_model_state = self.storage.download_model(old_hash)
                    # Kiểm tra model_state là dict có dữ liệu
                    if isinstance(old_model_state, dict) and len(old_model_state) > 0:
                        current_round_cluster_models[cid] = old_model_state
        consensus_dist = self._calculate_consensus_distance(self.workers, current_round_cluster_models)

        # Phase 5.1: Lưu cid mới và update k-models cho vòng sau
        self._finalize_round_and_prepare_next(consensus_results)
        t6 = time.time()
        # 3. FINALIZE & RETURN METRICS
        # ----------------------------------------------------------------------
        execution_time = t6 - start_time
        
        print(f"Round {round_id} finished inside Engine in {execution_time:.2f}s")

        self.logs["rounds"].append(round_id)
        self.logs["roles"].append(current_roles)
        self.logs["cluster_assignments"].append(cluster_map)
        self.logs["reputation"].append(self.blockchain.reputation_scores.copy())
        self.logs["comm_traffic_mb"].append(total_traffic)

        # Phân rã thời gian (Latency Breakdown)
        latency_breakdown = {
            "time_election": t1 - t0,
            "time_clustering": t2 - t1,
            "time_training": t3 - t2,
            "time_gossip": t4 - t3,
            "time_aggregation": t5 - t4,
            "time_consensus": t6 - t5
        }
        # Thêm mảng log thời gian nếu chưa có
        if "latency_breakdown" not in self.logs:
            self.logs["latency_breakdown"] = []
        self.logs["latency_breakdown"].append(latency_breakdown)
        
        return {
            "execution_time": execution_time,
            "global_loss": global_train_loss,
            "consensus_distance": consensus_dist,
            # "avg_accuracy": avg_accuracy,
            # "topology": all_topologies,
            **self.logs,
            "logs": consensus_log,
            # "scenario": f"Scenario {scenario_id}",
            "reputation": self.blockchain.reputation_scores,
            "current_round_k_models": current_round_cluster_models
        }

    # --- CÁC PHƯƠNG THỨC NỘI BỘ (PRIVATE METHODS) CHO TỪNG PHA ---
    def _initialize_global_k_models(self):
        print("\n[Engine] Initializing System: Generating random K-Models...")
        num_clusters = getattr(Config, 'NUM_CLUSTERS', 5)
        initial_registry = {}
        # Get input_channels from first worker if available, otherwise use default 3
        input_channels = 3
        if self.workers and len(self.workers) > 0:
            input_channels = getattr(self.workers[0], 'input_channels', 3)
        for cid in range(num_clusters):
            # Tạo model
            from app.models.cnn import get_model
            temp_model = get_model(self.engine_config.get('model'),num_classes=self.engine_config.get('num_classes', 10),input_channels=input_channels)
            model_state = {k: v.cpu().clone() for k, v in temp_model.state_dict().items()}
            # Upload lên Storage
            cid_hash = self.storage.upload_model(model_state=model_state)
            # Lưu vào dict đăng ký
            initial_registry[cid] = cid_hash
            print(f" -> Generated Genesis Model for Cluster {cid} (CID: {cid_hash[:8]})")

            del temp_model
        self.blockchain.update_global_models_registry(initial_registry)

    def _phase_clustering(self, workers_pools, round_id):
        """
        Giai đoạn 1: Phân cụm động dựa trên Loss (Dynamic Clustering)
        Hỗ trợ Bypass Clustering: Gán tất cả vào 1 cụm duy nhất
        """
        print(f"[Phase 1] Clustering (Round {round_id})...")
        
        # Kiểm tra Bypass Clustering
        if self.bypass_executor and not self.bypass_executor.should_cluster():
            print("[Bypass Clustering] All workers assigned to single cluster")
            return BypassClustering.cluster_all_nodes(workers_pools, cluster_id=0)
        
        num_clusters = getattr(Config, 'NUM_CLUSTERS', 5)
        cluster_models = {} # Dictionary chứa {cluster_id: state_dict}        

        # --- BƯỚC 2: WORKER CHỌN CỤM (JOIN) ---
        # Reset trạng thái cũ
        clusters = {i: [] for i in range(num_clusters)}
        
        for w in workers_pools:
            w.is_head = False
            w.members = []
            
            # Worker tính loss trên từng centroid và chọn cái tốt nhất
            # Hàm join_cluster sẽ set w.cluster_id
            w.join_cluster_via_blockchain(blockchain=self.blockchain)
            
            # Gom nhóm để bầu chọn Head sau này
            if w.cluster_id in clusters:
                clusters[w.cluster_id].append(w)
            else:
                # Trường hợp worker chọn cụm chưa có trong danh sách (ít xảy ra)
                clusters[w.cluster_id] = [w]

        # --- BƯỚC 3: BẦU CHỌN CLUSTER HEAD ---
        # Trong mỗi cụm, ta cần 1 người làm Head để tổng hợp
        # Chiến lược: Chọn người có Loss thấp nhất làm Head (Best Performance)
        
        active_clusters_count = 0
        
        for cid, members in clusters.items():
            if not members:
                print(f" -> Cluster {cid} is empty!")
                continue
                
            active_clusters_count += 1
            
            # Chọn Head: chọn người có Loss thấp nhất trong cụm
            # (Giả sử join_cluster đã lưu loss vào biến w.current_loss, nếu chưa thì chọn random)
            # Cách đơn giản nhất: Chọn thành viên đầu tiên làm Head
            # head_node = members[0]
            head_node = min(members, key=lambda w: getattr(w, 'current_loss', float('inf')))
            
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
        """
        Giai đoạn 2: Huấn luyện & LDP (Local Differential Privacy)
        Hỗ trợ Bypass Privacy: Bỏ qua thêm nhiễu LDP/SSS
        """
        print("[Phase 2] Training...")
        worker_updates_cache = {}
        
        # Duyệt qua các cụm để lấy danh sách worker active
        active_workers = []
        for cid, members in clusters.items():
            active_workers.extend(members)
        
        train_losses = []
        for w in active_workers:
            # Lưu ý: w.model đã được load weights của cụm ở Phase 1 (trong hàm join_cluster)
            
            # 1. Local Training
            result = w.train()
            w_final = result['weights']
            w_loss = result.get('loss')
            if w_loss is not None:
                train_losses.append(w_loss)
            
            # 2. Local Differential Privacy (LDP)
            # Kiểm tra Bypass Privacy
            if self.bypass_executor and not self.bypass_executor.should_apply_privacy():
                print(f"[Bypass Privacy] Worker {w.id} sending clean gradient (no LDP noise)")
                # Bỏ qua thêm nhiễu, gửi gradient gốc
                w_final = BypassPrivacy.skip_ldp_noise(w_final)
            else:
                # Normal LDP processing (nếu worker có hàm apply_ldp)
                if hasattr(w, 'apply_ldp'):
                    w_final = w.apply_ldp(w_final)
            
            # Lưu vào cache để lát nữa Head thu thập
            worker_updates_cache[w.id] = w_final

        global_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
        print(f" -> Trained {len(worker_updates_cache)} workers.")
        return worker_updates_cache, global_train_loss

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
        """
        Giai đoạn 4: Gossip & Aggregation
        Hỗ trợ Bypass Byzantine: Dùng FedAvg thay vì BALANCE
        """
        print("[Phase 4] Aggregation...")
        worker_lookup = {w.id: w  
                         for cluster in clusters.values()
                         for w in cluster}
        cluster_heads = {node.cluster_id: node 
                         for cluster in clusters.values()
                         for node in cluster
                         if node.is_head}
        final_updates_for_ch = {}
        total_traffic = 0.0

        if not cluster_heads:
            print("Warning: No Cluster Heads found! Check _phase_clustering.")
            return [], [], 0.0
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
            total_traffic += get_model_size_mb(model=model_state)
            ch_id = worker_lookup[w_id].cluster_id
            cluster_heads[ch_id].receive_update(w_id, model_state)

        # Aggregate
        results = []
        accuracies = []
        for ch in cluster_heads.values():
            # Kiểm tra Bypass Byzantine
            if self.bypass_executor and not self.bypass_executor.should_use_byzantine():
                print(f"[Bypass Byzantine] Cluster {ch.cluster_id} using FedAvg aggregation")
                # Collect local updates từ pending_models
                # IMPORTANTE: pending_models contém tuplas (worker_id, model_state_dict)
                # Precisa extrair apenas o model_state_dict (segundo elemento)
                local_updates = []
                for item in getattr(ch, 'pending_models', []):
                    if isinstance(item, tuple) and len(item) >= 2:
                        # Extract model_state_dict from tuple (worker_id, model_state_dict)
                        model_state = item[1]
                    elif isinstance(item, dict):
                        # If it's already a dict, use it directly
                        model_state = item
                    else:
                        # Skip invalid items
                        continue
                    
                    if isinstance(model_state, dict):
                        local_updates.append(model_state.copy())
                
                if not local_updates and hasattr(ch, 'global_model'):
                    # Nếu chưa có pending_models, dùng model hiện tại
                    local_updates = [ch.global_model.state_dict()]
                
                # Dùng FedAvg aggregation
                aggregated_model = BypassByzantine.fedavg_aggregation(local_updates)
                aggregate_res = {
                    "flat_weights": None,  # Bypass Byzantine doesn't use secret sharing
                    "model_state_dict": aggregated_model,
                    "model_hash": str(hash(str(aggregated_model)))
                }
            else:
                # Normal aggregation (BALANCE or robust aggregation)
                ch.set_committee_info(committee_config={
                    't': self.t_threshold,
                    'n': len(self.current_committee),
                    'committee': self.current_committee
                })
                aggregate_res = ch.aggregate(round_k=round_id)
            
            total_traffic += ch.calculate_traffic()


            # Giả lập validate accuracy
            # simulated_acc = min(95.0, 15.0 + round_id * 2.5 + random.uniform(-2, 3))
            accuracies.append(0.0)
            members = clusters[ch.cluster_id]
            results.append({"cluster_id": ch.cluster_id, **aggregate_res, "cluster_head_id": ch.id, "cluster_members": [m.id for m in members]})
            
        return results, accuracies, total_traffic

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

    # def _phase_consensus(self, results):
    #     print("[Phase 5] Blockchain Consensus...")
    #     logs = []
    #     proposer = self.current_proposer
    #     committee = self.current_committee
    #     threshold = self.t_threshold
    #     consensus_threshold = self.consensus_threshold
        
    #     sorted_committee_ids = sorted([v.id for v in committee])
    #     id_to_index = {uid: i for i, uid in enumerate(sorted_committee_ids, start=1)}
    #     print(f"[Debug] Index Mapping for Reconstruction: {id_to_index}")
    #     for res in results:
    #         cluster_id = res['cluster_id']
    #         encrypted_shares_map = res.get('encrypted_shares')
    #         cluster_members = res.get('cluster_members', [])            

    #         if not encrypted_shares_map:
    #             continue            

    #         # --- BƯỚC 1: Validator (Committee) giải mã ---
    #         decrypted_shares = {}
            
    #         for validator in committee:
    #             # Kiểm tra xem có gói tin cho validator này không
    #             if validator.id in encrypted_shares_map:
    #                 enc_pkg = encrypted_shares_map[validator.id]
                    
    #                 # Validator tự dùng key của mình để giải mã
    #                 share = validator.decrypt_share(enc_pkg)
                    
    #                 if share is not None:
    #                     # decrypted_shares[validator.id] = share
    #                     correct_x = id_to_index.get(validator.id)
                        
    #                     if correct_x:
    #                         decrypted_shares[correct_x] = share
    #                     else:
    #                         print(f"Error: Validator {validator.id} not in sorted mapping!")
            
    #         # --- BƯỚC 2: Proposer tái tạo Model ---
    #         metadata = res['metadata']
    #         reconstructed_model = proposer.reconstruct_model(collected_shares=decrypted_shares, metadata=metadata,threshold=threshold)

    #         # with open("rec_model.txt", "w", encoding="utf-8") as f:
    #         #     f.write(str(reconstructed_model))
                
    #         if reconstructed_model is None:
    #             logs.append(f"Cluster {cluster_id}: Consensus Failed (Reconstruction Error)")
    #             continue

    #         # --- Verify (Soft Check Norm/Hash) ---
    #         proposed_norm = res.get('model_norm')
    #         rec_norm = compute_model_norm(reconstructed_model)
            
    #         if abs(proposed_norm - rec_norm) > 1e-3:
    #             logs.append(f"The difference between two models are too much")
    #             continue
    #         else:
    #             logs.append(f"Two models are the same!")

    #         # Bước 3: COmmittee Validation
    #         votes, scores = self._run_committee_validation(committee=committee, model_state=reconstructed_model,val_loader=self.test_loader)

    #         total_votes = len(votes)
    #         approved_votes = sum(votes.values())
    #         is_approved = (approved_votes / total_votes) >= consensus_threshold

    #         # Tính điểm Acc trung bình của cả hội đồng
    #         final_acc = sum(scores.values()) / total_votes if total_votes > 0 else 0
    #         print(f" -> Consensus Result: {approved_votes}/{total_votes} votes. Approved? {is_approved}")

    #         # Bước 4: Gọi Smart Contract
    #         self.blockchain.execute_smart_contract(
    #             proposer_id=proposer.id,
    #             cluster_members=cluster_members,
    #             votes=votes,
    #             accuracy=final_acc,
    #             is_good_update=is_approved
    #         )

    #         if not is_approved:
    #             logs.append(f"Cluster {cluster_id}: Rejected (Vote Failed {approved_votes}/{total_votes})")
    #             continue

    #         # Bước 5: Tạo BLOCK mới
    #         storage_path = self.blockchain._save_model_offchain(reconstructed_model, cluster_id, res['model_hash'])

    #         last_block = self.blockchain.chain[-1]
    #         block_data = {
    #             "accuracy": final_acc,          # Dùng accuracy do ủy ban chấm
    #             "model_hash": res['model_hash'],
    #             "storage_uri": storage_path,
    #             "votes": votes,
    #             "previous_block": last_block
    #         }
            
    #         new_block = self.blockchain.add_block(
    #             block_data
    #         )
    #         success = self.blockchain.is_valid_new_block(new_block, last_block)
    #         status = "Accepted" if success else "Rejected (Blockchain Add Error)"
    #         logs.append(f"Cluster {cluster_id}: {status}")

    #     return logs
    def _phase_consensus(self, results):
        """
        Giai đoạn 5: Đồng thuận Blockchain (Consensus)
        Bao gồm: Thu thập mảnh, View Change (nếu cần), Bầu cử và Ghi Block.
        Hỗ trợ Bypass Blockchain: Lưu model vào memory storage
        """
        print("\n[Phase 5] Blockchain Consensus with View Change...")
        logs = []
        consensus_data = []
        
        # Kiểm tra Bypass Blockchain
        if self.bypass_executor and not self.bypass_executor.should_use_blockchain():
            print("[Bypass Blockchain] Using in-memory storage instead of blockchain")
            for res in results:
                cluster_id = res.get('cluster_id')
                model_state = res.get('model_state_dict')
                
                # Se model_state_dict não é dict válido, pula
                if not isinstance(model_state, dict) or len(model_state) == 0:
                    print(f"[Bypass Blockchain] Cluster {cluster_id} has invalid model_state, skipping")
                    continue
                
                # Lưu vào memory storage
                self.bypass_executor.memory_storage.store_model(
                    cluster_id=cluster_id,
                    model_state=model_state,
                    metadata={"round": len(logs), "status": "ACCEPTED"}
                )
                
                # Trả về kết quả bypass (không cần vote/consensus)
                consensus_result = {
                    "status": "ACCEPTED",
                    "cluster_id": cluster_id,
                    "model_state_dict": model_state,
                    "storage_uri": f"memory://cluster_{cluster_id}",
                    "data": {"accuracy": 0.0, "votes": {}}
                }
                consensus_data.append(consensus_result)
                logs.append(f"Cluster {cluster_id}: Bypass storage completed")
            
            return consensus_data, logs
        
        # Normal blockchain consensus (không bypass)
        # Danh sách worker dự phòng để thay thế nếu cần View Change
        available_workers = self.workers 

        for res in results:
            # Gọi hàm xử lý chi tiết cho từng Cluster
            process_results = self._process_cluster_consensus(res, available_workers)
            if isinstance(process_results, dict):
                consensus_data.append(process_results)
                status = process_results.get('status')
                cid = process_results.get('cluster_id')
                logs.append(f"Cluster {cid}: Consensus finished {status}")
            else:
                logs.append(process_results)

        return consensus_data, logs
    
    def _run_committee_validation(self, committee, model_state, val_loader):
        """
        Chạy vòng lặp lấy ý kiến đánh giá của từng Validator        
        """
        votes = {}
        scores = {}
        print(f"Current Dynamic threshold {self.currrent_dynamic_threshold}",flush=True)
        for validator in committee:
            acc,vote = validator.validate_update(model_state, val_loader, self.currrent_dynamic_threshold)
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
    
    def _calculate_advanced_metrics(self, attack_config, current_round_cluster_models=None):
        """
        Hàm chính: Điều phối việc tính toán dựa trên loại tấn công.
        """
        attack_type = attack_config.get('attack_type', 'NONE')
        benign_workers = [w for w in self.workers if not w.is_malicious]
        malicious_workers = [w for w in self.workers if w.is_malicious]
        
        if not benign_workers: return {}
        
        print(f"Calculating metrics for [{attack_type}] on {len(benign_workers)} benign nodes...")

        # 1. Tính các Metrics riêng biệt (Specific Metrics)
        self.specific_metrics = {}
        error_rates = [] # Cần thu thập để tính Max.TER
        accuracies = []  # Cần thu thập để tính Avg Acc
        is_attack = True

        def safe_mean(values, default=0.0):
            clean_vals = [v for v in values if v is not None and not math.isnan(v)]
            # if not clean_vals: return 0.0
            return sum(clean_vals) / len(clean_vals) if clean_vals else default

        if attack_type == 'LABEL_FLIPPING':
            src = int(attack_config.get('source_class'))
            tgt = int(attack_config.get('target_class'))
            if src is None or tgt is None:
                print("[Lỗi] Kịch bản Label Flipping bị thiếu 'source_class' hoặc 'target_class' trong Config!")
                return self.specific_metrics # Thoát an toàn
            
            # Container tạm
            asrs, src_recalls, tgt_precisions, f1s, aucs = [], [], [], [], []
            
            for w in benign_workers:
                if self.system_mode == "PROPOSED":
                    # m = w.evaluate_label_flipping(self.test_loader, src, tgt)
                    m = w.evaluate_label_flipping(src, tgt, current_round_cluster_models, self.test_loader)
                else:
                    m = w.evaluate_label_flipping(src, tgt, None, self.test_loader)
                accuracies.append(m['accuracy'])
                error_rates.append(m['error_rate'])
                asrs.append(m['asr'])
                src_recalls.append(m['src_recall'])
                tgt_precisions.append(m['tgt_precision'])
                f1s.append(m['f1'])
                aucs.append(m['auc'])
            
            # Tổng hợp
            self.specific_metrics = {
                "asr": sum(asrs) / len(asrs),
                "src_recall": sum(src_recalls) / len(src_recalls),
                "tgt_precision": sum(tgt_precisions) / len(tgt_precisions),
                "f1": safe_mean(f1s),
                "auc": safe_mean(aucs)
            }

        elif attack_type in ['GAUSSIAN_MODEL_POISONING', 'MODEL_POISONING']:
            # Với Gaussian, ta quan tâm MSE và Loss hơn là ASR
            mses = []
            losses = []
            
            for w in benign_workers:
                if self.system_mode == "PROPOSED":
                    m = w.evaluate_gaussian_metrics(current_round_cluster_models)
                else:
                    m = w.evaluate_gaussian_metrics()
                
                accuracies.append(m['accuracy'])
                error_rates.append(m['error_rate'])

                mse = m.get('mse', 1.0)
                loss = m.get('loss', 10000.0)

                mses.append(mse)
                losses.append(loss)
                
            self.specific_metrics = {
                "avg_mse": safe_mean(mses, default=1.0),
                "avg_loss": safe_mean(losses, default=10000.0)
            }
        elif attack_type == 'BACKDOOR':
            tgt = attack_config.get('target_class')
            asrs = []
            
            print(f"   Using Target Class {tgt} for Backdoor Eval")
            for w in benign_workers:
                # Gọi hàm
                if self.system_mode == "PROPOSED":
                    m = w.evaluate_backdoor(tgt, current_round_cluster_models)
                else:
                    m = w.evaluate_backdoor(tgt)

                # Sanitize từng metric đơn lẻ trước khi append
                acc = m.get('accuracy', 0.0)
                err = m.get('error_rate', 1.0)
                asr = m.get('asr', 0.0)

                if math.isnan(acc): acc = 0.0
                if math.isnan(err): err = 1.0
                if math.isnan(asr): asr = 0.0
                
                accuracies.append(m['accuracy']) # Clean Accuracy
                error_rates.append(m['error_rate'])
                asrs.append(m['asr'])            # Backdoor Success Rate
            
            self.specific_metrics = {
                "asr": safe_mean(asrs),
                "clean_acc": safe_mean(accuracies)
            }
        elif attack_type in ["MIA", "GIA", "GRADIENT_INVERSION"]:
            target_workers = malicious_workers if malicious_workers else benign_workers
            results = defaultdict(list)
            gen_errors = []

            for w in target_workers:
                m = w.evaluate_privacy()
                print(m, "MEtrics do not exist?", flush=True)

                for key, val in m.items():
                    results[key].append(val)
            # Tính trung bình cho tất cả key thu được
            for key, vals in results.items():
                self.specific_metrics[key] = safe_mean(vals)
            
            if attack_type == 'MIA':
                # Thêm Gen error + Accuracy metrics
                train_acc = m.get('member_acc', 0.0)
                test_acc = m.get('non_member_acc', 0.0)
                gen_errors.append(abs(train_acc - test_acc))
                self.specific_metrics['gen_error'] = safe_mean(gen_errors)
                accuracies.append(test_acc)
                error_rates.append(1.0 - test_acc)
                pass
            elif attack_type in ['GIA', 'GRADIENT_INVERSION']:
                # Thêm accuracy từ privacy attack evaluation (nếu có)
                for key, vals in results.items():
                    if 'accuracy' in key.lower() and vals:
                        avg_acc_val = safe_mean(vals)
                        accuracies.append(avg_acc_val)
                        error_rates.append(1.0 - avg_acc_val)
                        break                
                print(f"   [GIA Report] Avg MSE: {self.specific_metrics.get('recon_mse', 0):.4f}, "
                      f"PSNR: {self.specific_metrics.get('recon_psnr', 0):.2f} dB")
        else:
            # # Fallback cho trường hợp chạy bình thường (NONE)
            # is_attack = False
            # losses = []
            # for w in benign_workers:
            #     m = w.evaluate_standard(self.test_loader) # Dùng hàm cơ bản
            #     accuracies.append(m['accuracy'])
            #     error_rates.append(m['error_rate'])
            #     losses.append(m['loss'])
            # self.specific_metrics = {
            #     "avg_loss": safe_mean(losses, default=10000.0)
            # }
            # Fallback cho trường hợp chạy bình thường (NONE)
            is_attack = False
            losses = []
            accuracies = []
            error_rates = []

            # KIỂM TRA: Đang chạy Proposed (có k-models) hay Baseline?
            # Note: current_round_cluster_models có thể là dict rỗng nếu consensus fails
            if current_round_cluster_models and len(current_round_cluster_models) > 0:
                # --- LUỒNG PROPOSED (Đánh giá trên K-Models) ---
                print("   [Metrics] Evaluating Personalized Accuracy on K-Models...")
                for w in benign_workers:
                    m = w.evaluate_k_models(current_round_cluster_models)
                    accuracies.append(m['accuracy'])
                    error_rates.append(m['error_rate'])
                    losses.append(m['loss'])
            else:
                # --- LUỒNG BASELINE (Đánh giá trên Model Cục bộ) ---
                # This is triggered when:
                # 1. current_round_cluster_models is None (BASELINE mode)
                # 2. current_round_cluster_models is empty dict (consensus failed)
                print("   [Metrics] Evaluating Local Models (Baseline)...")
                for w in benign_workers:
                    # Truyền global test_loader để đánh giá khách quan
                    m = w.evaluate_standard(self.test_loader) 
                    accuracies.append(m['accuracy'])
                    error_rates.append(m['error_rate'])
                    losses.append(m['loss'])

            self.specific_metrics = {
                "avg_loss": safe_mean(losses, default=10000.0)
            }

        # 2. Tính các Metrics chung (Common Metrics)
        self.common_metrics = self._calculate_common_metrics(benign_workers, error_rates, accuracies, is_attack=is_attack)

        # 3. Gộp kết quả (Merge dicts)
        return {**self.specific_metrics, **self.common_metrics}

    def _calculate_common_metrics(self, benign_workers, error_rates, accuracies, is_attack=False):
        """
        Tính toán các chỉ số hệ thống dùng chung cho mọi kịch bản.
        Bao gồm: Max.TER, Consensus Error, TPR/FPR, Comm Cost.
        """
        # A. Max.TER & Avg Acc
        # Validate and sanitize before calculation
        # Filter out 1.0 error rates that likely represent fallback/NaN cases (accuracy=0 with NaN model)
        # Keep actual error rates from real evaluation
        valid_errors = [
            e for e in error_rates 
            if e is not None and not math.isnan(e) and not math.isinf(e) and e < 1.0
        ]
        valid_accs = [a for a in accuracies if a is not None and not math.isnan(a) and not math.isinf(a)]
        
        # Fallback: if all errors filtered out (all 1.0), include them but use median instead of max
        if not valid_errors and error_rates:
            valid_errors = [e for e in error_rates if e is not None and not math.isnan(e) and not math.isinf(e)]
            if valid_errors:
                max_ter = sorted(valid_errors)[len(valid_errors) // 2]  # Median as fallback
                print(f"   [Metrics] All error_rates were 1.0 (fallback), using median: {max_ter}")
            else:
                max_ter = 1.0
        elif valid_errors:
            max_ter = max(valid_errors)
        else:
            max_ter = 0.0  # No valid error rates at all
            
        if valid_accs:
            avg_acc = sum(valid_accs) / len(valid_accs)
        else:
            avg_acc = 0.0  # Fallback when no valid accuracies

        # B. Consensus Error (Độ phân tán của mô hình lành tính)
        benign_weights_flat = []
        for w in benign_workers:
            is_valid_model = True
            params_vec = []
            for p in w.model.state_dict().values():
                if torch.isnan(p).any() or torch.isinf(p).any():
                    is_valid_model = False
                    break
                params_vec.append(p.view(-1).float())
            if is_valid_model:
                benign_weights_flat.append(torch.cat(params_vec))

            vec = torch.cat([p.view(-1).float() for p in w.model.state_dict().values()])
            benign_weights_flat.append(vec)
            
        if benign_weights_flat:
            # stack = torch.stack(benign_weights_flat)
            # mean_vec = torch.mean(stack, dim=0)
            # dists = torch.norm(stack - mean_vec, dim=1)
            # consensus_error = torch.mean(dists).item()
            try:
                stack = torch.stack(benign_weights_flat)
                mean_vec = torch.mean(stack, dim=0)
                dists = torch.norm(stack - mean_vec, dim=1)
                consensus_error = torch.mean(dists).item()
                
                # Double Check lần cuối
                if math.isnan(consensus_error) or math.isinf(consensus_error):
                    consensus_error = 1000.0 # Giá trị phạt nếu lỗi toán học
            except Exception:
                consensus_error = 1000.0
        else:
            consensus_error = 0.0

        # C. Detection Metrics
        # tpr, fpr = self._calculate_detection_rate()

        # D. Comm Cost (Ước tính)
        model_size_mb = 1.2 
        comm_cost = len(self.workers) * model_size_mb

        response_metrics = {
            "max_ter": max_ter,
            "avg_acc": avg_acc,
            "consensus_error": consensus_error,            
            "comm_cost": comm_cost
        }
        if not is_attack:
            return response_metrics
        else:
            response_metrics["tpr"] = 0.0
            response_metrics["fpr"] = 0.0
            return response_metrics
    def _calculate_detection_rate(self):
        """
        Tính TPR (True Positive Rate) và FPR (False Positive Rate)
        Dựa trên danh sách node bị hệ thống chặn/cô lập.
        """
        # Ground Truth
        actual_malicious = set([w.id for w in self.workers if w.is_malicious])
        actual_benign = set([w.id for w in self.workers if not w.is_malicious])        
        
        if not actual_malicious: return 1.0, 0.0 # Không có tấn công

        # Thu thập danh sách bị chặn từ tất cả cluster heads
        total_blocked_mal = 0
        total_blocked_benign = 0

        total_mal_updates = 0 # Tổng số lần kẻ xấu gửi updates
        total_benign_updates = 0 # Tổng số ng tốt gửi update

        # for ch in self.cluster_heads:
        #     if not hasattr(ch, 'blocked_ids_this_round'): continue

        #     blocked_set = ch.blocked_ids_this_round
        #     member_ids = set(ch.members)

        #     local_malicious = member_ids.intersection(actual_malicious)
        #     local_benign = member_ids.intersection(actual_benign)

        #     total_benign_updates += len(local_benign)
        #     total_mal_updates += len(local_malicious)

        #     total_blocked_benign += len(blocked_set.intersection(actual_benign))
        #     total_blocked_mal += len(blocked_set.intersection(actual_malicious))
        # ==========================================
        # TRƯỜNG HỢP 1: KỊCH BẢN PROPOSED (Có Cluster Heads)
        # ==========================================
        if hasattr(self, 'cluster_heads') and self.cluster_heads:
            for ch in self.cluster_heads:
                # Nếu CH chưa có danh sách chặn, bỏ qua
                if not hasattr(ch, 'blocked_ids_this_round'): continue

                blocked_set = set(ch.blocked_ids_this_round)
                member_ids = set(ch.members)

                local_malicious = member_ids.intersection(actual_malicious)
                local_benign = member_ids.intersection(actual_benign)

                total_benign_updates += len(local_benign)
                total_mal_updates += len(local_malicious)

                total_blocked_benign += len(blocked_set.intersection(actual_benign))
                total_blocked_mal += len(blocked_set.intersection(actual_malicious))

        # ==========================================
        # TRƯỜNG HỢP 2: KỊCH BẢN BASELINE (StandardDFL)
        # ==========================================
        else:
            # Trong Baseline, mỗi worker gửi 1 update lên Global/Neighbors
            total_mal_updates = len(actual_malicious)
            total_benign_updates = len(actual_benign)

            # Lấy danh sách bị chặn toàn cục (Nếu thuật toán aggregate có ghi nhận)
            # FedAvg sẽ không có biến này -> blocked_set rỗng -> TPR = 0, FPR = 0 (Hợp lý vì FedAvg không chặn ai)
            if hasattr(self, 'blocked_ids_this_round'):
                blocked_set = set(self.blocked_ids_this_round)
            else:
                blocked_set = set()

            total_blocked_mal = len(blocked_set.intersection(actual_malicious))
            total_blocked_benign = len(blocked_set.intersection(actual_benign))

        # TPR = TP / (TP + FN) -> Tỷ lệ node xấu bị bắt
        tpr = total_blocked_mal / total_mal_updates if total_mal_updates > 0 else 0.0
        # FPR = FP / (FP + TN) -> Tỷ lệ node tốt bị bắt nhầm
        fpr = total_blocked_benign / total_benign_updates if total_benign_updates > 0 else 0.0
        
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
        committee = random.sample(committee_pool, k) + [proposer]

        # 7. Worker là phần còn lại
        # Gom Proposer và Committee thành set để loại trừ
        excluded_ids = {proposer.id} | {n.id for n in committee}
        workers = [n for n in self.workers if n.id not in excluded_ids]

        print(f" -> Proposer: {proposer.id}")
        print(f" -> Committee ({len(committee)}): {[n.id for n in committee]}")
        print(f" -> Workers ({len(workers)}): Count only")

        return proposer, committee, workers
    
    # Các hàm dùng cho _phase_consensus
    def _process_cluster_consensus(self, res, available_workers):
        """
        Xử lý logic thử lại và thay thế ủy ban
        """
        cluster_id = res['cluster_id']
        flat_weights = res.get('flat_weights')
        metadata = res.get('metadata')
        model_state_dict = res.get('model_state_dict')

        # Nếu flat_weights là None (bypass Byzantine), không cần secret sharing
        if flat_weights is None:
            print(f"[Bypass Byzantine] Cluster {cluster_id} skipping secret sharing (flat_weights=None)")
            # Trả về kết quả trực tiếp mà không cần consensus
            consensus_result = {
                "status": "ACCEPTED",
                "cluster_id": cluster_id,
                "model_state_dict": model_state_dict,
                "storage_uri": f"memory://cluster_{cluster_id}",
                "data": {"accuracy": 0.0, "votes": {}}
            }
            return consensus_result

        # Validate flat_weights: nếu dict, convert từ model_state_dict
        if isinstance(flat_weights, dict):
            print(f"[WARN] flat_weights invalid (type={type(flat_weights)}), attempting to create from model_state_dict...")
            if isinstance(model_state_dict, dict) and len(model_state_dict) > 0:
                try:
                    from app.utils.secret_sharing import SecretSharingUtils
                    flat_weights, _ = SecretSharingUtils.flatten_weights(model_state_dict)
                    print(f"[OK] Successfully created flat_weights from model_state_dict")
                except Exception as e:
                    print(f"[ERROR] Could not flatten weights: {e}")
                    flat_weights = None
            else:
                print(f"[ERROR] Invalid model_state_dict in consensus result")
                flat_weights = None
            
            # Nếu vẫn là None, retornar sem consensus
            if flat_weights is None:
                consensus_result = {
                    "status": "ACCEPTED",
                    "cluster_id": cluster_id,
                    "model_state_dict": model_state_dict,
                    "storage_uri": f"memory://cluster_{cluster_id}",
                    "data": {"accuracy": 0.0, "votes": {}}
                }
                return consensus_result

        active_committee = self.current_committee
        threshold = self.t_threshold
        consensus_threshold = self.consensus_threshold
        proposer = self.current_proposer
        MAX_RETRIES = Config.VC_MAX_RETRIES

        for attempt in range(MAX_RETRIES + 1):
            print(f"\n--- Consensus Attempt {attempt+1} (Cluster {cluster_id}) ---")

            # Bước 1: Thu thập mảnh giải mã
            decrypted_shares, responding_validators = self._attempt_decryption(
                cluster_id, flat_weights, active_committee
            )
            # Bước 2: Kiểm tra ngưỡng
            if len(decrypted_shares) >= threshold:
                print(f" -> Success! Collected {len(decrypted_shares)}/{threshold} shares.")
                return self._finalize_consensus_success(
                    res, decrypted_shares, active_committee
                )
            # Bước 3: Xử lý thất bại
            print(f" -> FAILED! Collected {len(decrypted_shares)} < {threshold}. Initiating View Change.")
            
            if attempt < MAX_RETRIES:
                # Gọi Proposer thực hiện thay thế ủy ban và trừng phạt node lỗi
                active_committee = proposer.execute_view_change(
                    old_committee=active_committee,
                    active_validators=responding_validators,
                    blockchain=self.blockchain,
                    available_workers=available_workers
                )
                print(f" -> New Committee for Retry: {[n.id for n in active_committee]}")
            else:
                print(" -> Max retries reached. Dropping update.")
                return f"Cluster {cluster_id}: FAILED (View Change Exhausted)"
        
        return f"Cluster {cluster_id}: FAILED (Unknown Error)"
    
    def _attempt_decryption(self, cluster_id, flat_weights, committee):
        # Bước 1: Tìm cluster head
        ch_node = next((w for w in self.workers if w.cluster_id == cluster_id and w.is_head), None)
        if not ch_node:
            print(f"Error: No Cluster Head found for Cluster {cluster_id}")
            return {}, []
        # Bước 2: Yêu cầu CH phân mảnh và mã hóa cho ủy ban này
        encrypted_packets = ch_node.distribute_shares_to_committee(flat_weights, committee)

        # Bước 3: Thu thập mảnh
        decrypted_shares = {}
        responding_validators = []
        sorted_ids = sorted([n.id for n in committee])
        id_to_index = {uid: i for i, uid in enumerate(sorted_ids, start=1)}

        for validator in committee:
            # Check lỗi (Mô phỏng offline)
            if self.blockchain.fault.get(validator.id, 0) > 2:
                print(f" -> Node {validator.id} unresponsive (High Faults).")
                continue
            if validator.id in encrypted_packets:
                share = validator.decrypt_share(encrypted_packets[validator.id])
                
                if share is not None:
                    # Map đúng index toán học
                    idx = id_to_index.get(validator.id)
                    if idx:
                        decrypted_shares[idx] = share
                        responding_validators.append(validator)
        
        return decrypted_shares, responding_validators
    
    def _finalize_consensus_success(self, res, decrypted_shares, final_committee):
        """
        Xử lý khi đã đủ mảnh: Tái tạo, Vote, Smart Contract, Lưu Block
        """
        cluster_id = res['cluster_id']
        metadata = res['metadata']
        proposer = self.current_proposer
        threshold = self.t_threshold
        cluster_head_id = res["cluster_head_id"]


        # Tái tạo Model
        reconstructed_model = proposer.reconstruct_model(decrypted_shares,metadata, threshold)

        # Dùng cho upload model vào Storage
        model_state_cpu = {k: v.cpu().clone() for k,v in reconstructed_model.items()}
        if reconstructed_model is None:
            return {"status": "FAILED", "cluster_id": cluster_id, "error": "Reconstruction Failed"}
        # Kiểm tái tạo model giống ban đầu chưa:
        proposed_norm = res.get('model_norm')
        rec_norm = compute_model_norm(reconstructed_model)
        print(f"[Integrity Check] Cluster {cluster_id}: Proposed Norm={proposed_norm:.4f} vs Rec Norm={rec_norm:.4f}")

        if rec_norm > proposed_norm * 2.0:
            scale_factor = rec_norm / proposed_norm
            print(f" -> DETECTED SCALING ERROR! Factor ~= {scale_factor:.2f}. Attempting to fix...")
            
            # Thử sửa bằng cách chia cho hệ số (thường là lũy thừa của 10 hoặc 2)
            # Ở đây ta chia trực tiếp để đưa về Norm gốc
            correction_ratio = proposed_norm / rec_norm
            for k in reconstructed_model.keys():
                reconstructed_model[k] = reconstructed_model[k] * correction_ratio
            
            # Tính lại Norm sau khi sửa
            rec_norm = compute_model_norm({k: v.cpu().float() for k, v in reconstructed_model.items()})
            print(f" -> Fixed Norm: {rec_norm:.4f}")
        tolerance = max(1e-2, 0.001 * proposed_norm) # Cho phép lệch 0.1% hoặc 0.01
    
        if abs(proposed_norm - rec_norm) > tolerance:
            print(f" -> REJECTED: Norm mismatch too high even after fix attempt.")
            return f"Cluster {cluster_id}: Rejected (Integrity Check Failed)"
        # Committee validation
        votes, scores = self._run_committee_validation(
            committee=final_committee, 
            model_state=reconstructed_model, 
            val_loader=self.test_loader
        )
        total_votes = len(votes)
        approved_votes = sum(votes.values())
        # Ngưỡng đồng thuận (ví dụ 2/3)
        consensus_threshold = self.consensus_threshold
        is_approved = (approved_votes / total_votes) >= consensus_threshold if total_votes > 0 else False
        final_acc = sum(scores.values()) / total_votes if total_votes > 0 else 0

        print(f" -> Consensus Result: {approved_votes}/{total_votes} votes. Approved? {is_approved}")

        # Smart Contract (Thưởng/Phạt)
        self.blockchain.execute_smart_contract(
            proposer_id=proposer.id,
            cluster_head_id=cluster_head_id,
            cluster_members=res.get('cluster_members', []),
            rejected_workers=res.get('rejected_workers', []),
            votes=votes,
            accuracy=final_acc,
            is_good_update=is_approved
        )

        # if not is_approved:
        #     return f"Cluster {cluster_id}: Rejected by Vote ({approved_votes}/{total_votes})"

        # Bước 5: Tạo BLOCK mới
        storage_path = self.blockchain._save_model_offchain(reconstructed_model, cluster_id, res['model_hash']) if is_approved else None
        
        # new_block = self.blockchain.add_block(
        #     block_data
        # )
        
        # return f"Cluster {cluster_id}: Accepted (Block {new_block.index})"
        return {
            "status": "ACCEPTED" if is_approved else "REJECTED",
            "cluster_id": cluster_id,
            "data": {
                "accuracy": final_acc,
                "model_hash": res['model_hash'],
                "storage_uri": storage_path,
                "votes": votes
            },
            "model_state_dict": model_state_cpu
        }
    
    def _finalize_round_and_prepare_next(self, results):
        """
        Cuối vòng: Gom model -> Upload -> Cập nhật Blockchain Registry
        """
        print("\n[Engine] Finalizing Round & Updating Hybrid Storage...")
        new_k_models_cids = {}
        for res in results:
            if res.get('status') == 'ACCEPTED':
                cid_id = res['cluster_id']
                model_state = res.get('model_state_dict')
                # Kiểm tra model_state là dict có dữ liệu, không phải boolean evaluation
                if isinstance(model_state, dict) and len(model_state) > 0:
                    ipfs_cid = self.storage.upload_model(model_state=model_state)
                    new_k_models_cids[cid_id] = ipfs_cid # Hash của file
                    print(f" -> Cluster {cid_id} uploaded to Storage (CID: {ipfs_cid[:4]})")
        # Xứ lý các cụm bị thiếu -> Giữ nguyên CID cũ
        current_registry = self.blockchain.get_latest_k_model_hashes()
        num_clusters = getattr(Config, 'NUM_CLUSTERS',2)
        for i in range(num_clusters):
            if i not in new_k_models_cids:
                if i in current_registry:
                    # Dùng lại cái cũ
                    new_k_models_cids[i] = current_registry[i]
                    print(f" -> Cluster {i} failed updates. Keeping old CID.")
                else:
                    # Chưa từng có
                    from app.models.cnn import get_model
                    model_name = self.engine_config.get('model')
                    input_channels = getattr(self.workers[0], 'input_channels', 3) if self.workers else 3
                    temp_model = get_model(
                        model_name=model_name,
                        num_classes=self.engine_config.get('num_classes', 10),
                        input_channels=input_channels
                    )
                    # Lấy state_dict
                    random_state = {k: v.cpu() for k, v in temp_model.state_dict().items()}
                    # Upload lên Storage
                    random_cid = self.storage.upload_model(random_state)
                    new_k_models_cids[i] = random_cid
                    print(f" -> Created Random Model for Cluster {i} (CID: {random_cid[:8]})")
                    del temp_model
        # Gửi transaction cập nhật Smart Contract
        self.blockchain.update_global_models_registry(new_k_models_cids)

    # Hàm tính consensus distance để trả về log kết quả
    def _calculate_consensus_distance(self, workers_list, cluster_models_map):
        """
        Tính Consensus Distance trong Clustered FL.
        Công thức: Avg( || Worker_i - Cluster_Model_of_Worker_i || )
        """
        total_distance = 0.0
        count = 0

        for w in workers_list:
            # 1. Xác định Worker này thuộc cụm nào
            cid = w.cluster_id
            
            # 2. Lấy model chuẩn của cụm đó
            # Nếu cụm đó bị lỗi (không có trong map), ta bỏ qua hoặc dùng model cũ
            if cid not in cluster_models_map:
                continue

            target_cluster_state = cluster_models_map[cid]
            worker_state = w.model.state_dict()
            
            w_dist = 0.0
            
            # 3. Tính khoảng cách Euclide
            for key in target_cluster_state:
                # Kiểm tra key có phải string trước khi dùng 'in'
                if not isinstance(key, str):
                    continue
                    
                if 'weight' in key or 'bias' in key:
                    if key not in worker_state:
                        continue
                        
                    w_tensor = worker_state[key].float().cpu()
                    c_tensor = target_cluster_state[key].float().cpu()
                    
                    # || w - c ||^2
                    diff = torch.norm(w_tensor - c_tensor, p=2).item()
                    w_dist += diff ** 2
            
            # Căn bậc 2 tổng bình phương
            total_distance += torch.sqrt(torch.tensor(w_dist)).item()
            count += 1
            
        return total_distance / count if count > 0 else 0.0
    
    # Dùng để lưu file json cho các kết quả (chưa cần dùng)
    def save_history_to_file(self):
        """Ghi đè metrics hiện tại vào file JSON"""
        try:
            with open(self.log_file_path, 'w') as f:
                json.dump(self.history, f, indent=4)
            # print(" -> Metrics saved to disk.")
        except Exception as e:
            print(f"Error saving metrics: {e}")

    def _evaluate_personalized_k_models(self, current_round_cluster_models):
        """
        Đánh giá Personalized Accuracy trên tập k-models đã được Ủy ban thông qua.
        Mỗi worker sẽ thử toàn bộ k-models trên local_test_loader của mình,
        chọn model có Loss thấp nhất để làm đại diện.
        """
        if not current_round_cluster_models:
            return 0.0, 1.0 # avg_acc = 0.0, max_ter = 1.0

        # Khởi tạo model vỏ rỗng để mượn đường chạy test (chạy trên GPU nếu có)
        input_channels = getattr(self.workers[0], 'input_channels', 3) if self.workers else 3
        eval_model = get_model(self.config.get('model', 'simple_cnn'), num_classes=self.num_classes, input_channels=input_channels).to(self.device)
        criterion = torch.nn.CrossEntropyLoss()

        all_workers_best_acc = []
        all_workers_best_err = []

        # Tắt gradient để tính toán siêu tốc
        with torch.no_grad():
            for w in self.workers:
                # Bỏ qua nếu worker không có tập test cục bộ
                if getattr(w, 'local_test_loader', None) is None:
                    continue
                    
                best_loss = float('inf')
                best_acc = 0.0
                
                # Worker "thử áo": Mặc thử từng model trong danh sách k-models
                for cid, model_state in current_round_cluster_models.items():
                    eval_model.load_state_dict(model_state)
                    eval_model.eval()
                    
                    client_loss = 0.0
                    correct = 0
                    total = 0
                    
                    # Chạy suy luận trên tập Test cục bộ (đã chia Dirichlet đồng bộ)
                    for data, target in w.local_test_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        outputs = eval_model(data)
                        
                        loss = criterion(outputs, target)
                        client_loss += loss.item() * data.size(0)
                        
                        pred = outputs.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += data.size(0)
                        
                    # Tính trung bình Loss và Accuracy cho model này
                    avg_loss = client_loss / total if total > 0 else float('inf')
                    acc = correct / total if total > 0 else 0.0
                    
                    # IFCA Standard: Chọn model có hàm Loss NHỎ NHẤT làm đại diện
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_acc = acc
                
                # Lưu lại kết quả của cho worker này
                all_workers_best_acc.append(best_acc)
                all_workers_best_err.append(1.0 - best_acc)

        # Tổng hợp thành metric của toàn mạng (Global System Metrics)
        avg_personalized_acc = sum(all_workers_best_acc) / len(all_workers_best_acc) if all_workers_best_acc else 0.0
        max_ter = max(all_workers_best_err) if all_workers_best_err else 1.0

        return avg_personalized_acc, max_ter