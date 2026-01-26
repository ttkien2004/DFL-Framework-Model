# app/core/engine.py
import time
import random
import torch
from config import Config
from app.scenarios.factory import ScenarioFactory
from app.core.baseline_node import StandardDFLNode
from app.core.cluster_head import ClusterHead
from app.scenarios.scenario_4_sec import ScenarioExperiment4
from app.utils.data_loader import get_global_test_loader
import copy
from collections import defaultdict

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
        current_scenario = ScenarioExperiment4(self.workers,req_data)
        current_scenario.setup_security(self.workers)

        cnt_mal = 0
        for node in self.workers:
            if node.is_malicious: cnt_mal += 1
        self.blocked_nodes = set()
        print(f"=== ROUND {round_id} START ({self.system_mode}) ===")

        execution_result = {}
        if self.system_mode == 'BASELINE':
            execution_result = self._run_round_baseline(round_id)
        else:
            execution_result = self._run_round_proposed(round_id)

        attack_config = {
            'attack_type': req_data.get('attack_type', 'NONE'),
            'source_class': req_data.get('source_class'),
            'target_class': req_data.get('target_class')
        }
        
        # Gọi hàm mới đã refactor
        security_metrics = self._calculate_advanced_metrics(attack_config)

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
        
        # Phase 1: Clustering
        self._phase_clustering()

        # Phase 2: Training & LDP
        worker_updates_cache = self._phase_training_ldp()

        # Phase 3: CoCo Optimization
        instruction_maps, all_topologies = self._phase_coco_optimization(worker_updates_cache)

        # Phase 4: Gossip & Aggregation
        round_results, real_accuracies = self._phase_aggregation(instruction_maps, worker_updates_cache, round_id)

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
            "topology": all_topologies,
            "logs": consensus_log,
            # "scenario": f"Scenario {scenario_id}",
            "reputation": self.blockchain.reputation_scores
        }

    # --- CÁC PHƯƠNG THỨC NỘI BỘ (PRIVATE METHODS) CHO TỪNG PHA ---

    def _phase_clustering(self):
        print("[Phase 1] Clustering...")
        
        num_clusters = getattr(Config, 'NUM_CLUSTERS', 5)
        
        # 1. Reset trạng thái của tất cả worker
        for w in self.workers:
            w.members = []      # Xóa danh sách thành viên cũ (nếu là Head cũ)
            w.cluster_head_id = -1
            w.cluster_id = -1

        # 2. Chỉ định Cluster Heads (Lấy num_clusters node đầu tiên)
        # Lưu ý: self.workers trong PROPOSED mode đều là object ClusterHead
        print("NUM Clusters Hello", num_clusters)
        active_heads = self.workers[:num_clusters]
        
        for idx, head in enumerate(active_heads):
            head.is_head = True
            head.cluster_head_id = idx
            head.cluster_id = idx
            # Head tự quản lý chính mình
            # head.register_member(head.id) 

        # 3. Gán các Member còn lại vào các Head
        member_nodes = self.workers[num_clusters:]
        
        for w in member_nodes:
            # Chọn ngẫu nhiên 1 Head để tham gia
            chosen_head = random.choice(active_heads)
            
            w.cluster_head_id = chosen_head.id
            w.cluster_id = chosen_head.cluster_id
            
            # Đăng ký member vào Head (gọi hàm của class ClusterHead)
            if hasattr(chosen_head, 'register_member'):
                chosen_head.register_member(w.id)
            else:
                # Fallback nếu chưa có hàm register
                chosen_head.members.append(w.id)
                
        print(f"-> Formed {num_clusters} clusters.")

    def _phase_training_ldp(self):
        print("[Phase 2] Training...")
        cache = {}
        for w in self.workers:
            trained_params = w.train()
            noisy_params = w.apply_ldp(trained_params)
            cache[w.id] = noisy_params
        return cache

    def _phase_coco_optimization(self, worker_updates_cache):
        print("[Phase 3] CoCo Optimization...")
        instruction_maps = {}
        all_topologies = {}
        
        # B1: Report Metrics
        node_map = {node.id: node for node in self.workers}
        for w in self.workers:
            # Hiện tại worker chưa có bandwidth
            metrics = {'bandwidth': random.uniform(20, 100), 'cpu_load': random.uniform(10, 80)}
            if w.cluster_head_id is not None:
                head_node = node_map.get(w.cluster_head_id)
                if head_node:
                    head_node.receive_metrics(w.id, metrics, worker_updates_cache[w.id])
        
        # B2: Run CoCo
        for ch in self.workers:
            head_node = node_map.get(w.cluster_head_id)

            instr, topo_viz = head_node.run_coco_optimization()
            instruction_maps.update(instr)
            all_topologies[head_node.cluster_id] = topo_viz
            
        return instruction_maps, all_topologies

    def _phase_aggregation(self, instruction_maps, worker_updates_cache, round_id):
        print("[Phase 4] Aggregation...")
        worker_lookup = {w.id: w for w in self.workers}
        cluster_heads = {node.cluster_id: node for node in self.workers if node.is_head}
        final_updates_for_ch = {}

        if not cluster_heads:
            print("Warning: No Cluster Heads found! Check _phase_clustering.")
            return [], []
        # Gossip Logic (DFCA)
        for w_id, instr in instruction_maps.items():
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
            _, model_hash = ch.aggregate(round_k=round_id)
            # Giả lập validate accuracy
            simulated_acc = min(95.0, 15.0 + round_id * 2.5 + random.uniform(-2, 3))
            accuracies.append(0.0)
            results.append({"cluster_id": ch.cluster_id, "hash": model_hash, "accuracy": 0.0})
            
        return results, accuracies

    def _phase_consensus(self, results):
        print("[Phase 5] Blockchain Consensus...")
        logs = []
        for res in results:
            success = self.blockchain.propose_update(res['cluster_id'], res['hash'], res['accuracy'])
            status = "Accepted" if success else "Rejected"
            logs.append(f"Cluster {res['cluster_id']}: {status}")
        return logs
    
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