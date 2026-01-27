# File chạy chính
from platform import node
from platform import node
from flask import Flask, jsonify, request, render_template
from app.core.worker import WorkerNode
from app.core.cluster_head import ClusterHead
from app.blockchain.consensus import Blockchain
import random # Thêm tạm thời để tính loss
from config import Config
from app.utils.helpers import model_to_json
from app.core.engine import SimulationEngine

from app.utils.history import update_history_dynamic
from collections import defaultdict
import time

from app.blockchain.node_manager import NodeManager


app = Flask(__name__)

# Khởi tạo hệ thống giả lập
# workers = [WorkerNode(i, None) for i in range(Config.NUM_WORKERS)]
# cluster_heads = [ClusterHead(i) for i in range(Config.NUM_CLUSTERS)]
# blockchain = Blockchain()
engine = SimulationEngine()


# BIẾN TOÀN CỤC LƯU LỊCH SỬ
history = {
    "rounds": [],
    "accuracy": [],
    "loss": [],
    "blockchain_height": [],
    "system_mode": []
}

# Biến toàn cục theo dõi dataset
current_system_dataset = Config.DATASET_NAME

@app.route('/')
def home():
    # return "BCFL CoCo Cluster Simulation API is Running!"
    return render_template('simulation.html')

@app.route('/chain', methods=['GET'])
def get_chain():
    """API xem trạng thái Blockchain"""
    chain_data = [block.to_dict() for block in blockchain.chain]
    return jsonify({
        "length": len(chain_data),
        "chain": chain_data,
        "reputation": blockchain.reputation_scores
    })

@app.route('/run_round', methods=['POST'])
def run_simulation_round():
    req_data = request.get_json(silent=True) or {}
    # current_round = len(blockchain.chain)

    requested_mode = req_data.get('system_mode', 'PROPOSED').upper()
    should_reset = req_data.get('reset', False)
    if should_reset or not engine.workers or engine.system_mode != requested_mode:
        print(f"[System Switch] Switching from {engine.system_mode} to {requested_mode}...")
        
        # Gọi hàm khởi tạo lại toàn bộ worker & topology
        engine.initialize_system(req_data)
        global history, blockchain
        # history = {"rounds": [], "system_mode": [], "blockchain_height": []}
        history = defaultdict(list)
        current_round = 0
        blockchain = engine.blockchain
    else:
        # Nếu cùng mode, tiếp tục vòng lặp
        current_round = len(history["rounds"]) + 1

    # Truyền blockchain vào engine (nếu hệ thống Proposed cần dùng)
    if hasattr(engine, 'set_blockchain_ref'):
        engine.set_blockchain_ref(blockchain)

    result = engine.run_round(current_round, req_data)
    # Cập nhật History (UI Dashboard)
    update_history_dynamic(history, current_round, result, requested_mode)
    # history["rounds"].append(current_round)
    # history["accuracy"].append(round(result['avg_accuracy'],2))
    history['blockchain_height'].append(len(blockchain.chain))
    # history["system_mode"].append(requested_mode)

    return jsonify({
        "status": "success",
        "round": current_round,
        **result,
        "history": history
    })
# @app.route('/run_round', methods=['POST'])
# def run_simulation_round():
#     """
#     LOGIC CHÍNH: Chạy một vòng huấn luyện (Round)
#     Bao gồm 5 Pha: Clustering -> CoCo -> Training/LDP -> Aggregation/BALANCE -> Consensus
#     """
#     global current_system_dataset

#     start_time = time.time()
#     current_round = len(blockchain.chain)

#     # Đọc cấu hình từ request
#     req_data = request.get_json(silent=True) or {}

#     attack_type = req_data.get('data_type', 'NONE')
#     malicious_percent = req_data.get('malicious_percent', 0.0)

#     num_malicious = int(len(workers) * malicious_percent)
#     malicious_indices = random.sample(range(len(workers)), num_malicious)

#     print(f"SCENARIO: {attack_type} on {num_malicious} workers ({malicious_indices})")

#     requested_dataset = req_data.get('dataset', current_system_dataset).lower()
#     # Kiểm tra chuyển đổi datasset
#     if requested_dataset != current_system_dataset:
#         print(f"Switching {current_system_dataset} to {requested_dataset}")
#         for w in workers:
#             w.reload_dataset(requested_dataset)
#         for ch in cluster_heads:
#             ch.reload_model(requested_dataset)

#         current_system_dataset = requested_dataset

#     for w in workers:
#         if w.id in malicious_indices:
#             w.set_attack_profile(attack_type)
#         else:
#             w.set_attack_profile("NONE")
#     print(f"\n---STARTING ROUND {current_round} ---")

#     # PHA 1: CLUSTERING (DFCA - Dynamic Federated Clustering)
#     print("[Phase 1] Clustering...")
    
#     # Reset thành viên của các CH vòng trước
#     for ch in cluster_heads:
#         ch.members = []
#         ch.member_metrics = {} 

#     # Worker chọn Cụm
#     for w in workers:
#         if w.cluster_id is None or random.random() < 0.1:
#             w.cluster_id = random.randint(0, Config.NUM_CLUSTERS - 1)
#         target_ch = cluster_heads[w.cluster_id]
#         target_ch.register_member(w.id)

#     # TRAINING & LDP (Local Differential Privacy)
#     print("[Phase 2] Local Training & LDP...")
#     worker_updates_cache = {}
    
#     for w in workers:
#         # if w.id in instruction_maps:
#         #     instr = instruction_maps[w.id]
#         #     trained_params = w.train() 
#         #     noisy_params = w.apply_ldp(trained_params)
#         #     target_ch = cluster_heads[w.cluster_id]
#         #     target_ch.receive_update(w.id, noisy_params)
#         trained_params = w.train()

#         # Áp dụng LDP
#         noisy_params = w.apply_ldp(trained_params)
#         worker_updates_cache[w.id] = noisy_params

#     # PHA 2: CoCo OPTIMIZATION (Topology & Compression)
#     print("[Phase 3] CoCo Optimization...")
    
#     all_topologies = {} 
#     instruction_maps = {} 

#     # B1: Worker báo cáo trạng thái
#     for w in workers:
#         metrics = {
#             'bandwidth': random.uniform(20, 100), 
#             'cpu_load': random.uniform(10, 80)
#         }
#         target_ch = cluster_heads[w.cluster_id]

#         cur_model = worker_updates_cache[w.id]
#         target_ch.receive_metrics(w.id, metrics, cur_model)

#     # B2: CH chạy thuật toán tối ưu
#     for ch in cluster_heads:
#         instructions, topology_viz = ch.run_coco_optimization()
#         instruction_maps.update(instructions)
#         all_topologies[ch.cluster_id] = topology_viz
    

#     # PHA 4: AGGREGATION & BALANCE FILTERING
#     print("[Phase 4] Aggregation...")
#     final_updates_for_ch = {}
#     # Thực thi giải thuật tổng hợp mô hình cục bộ của DFCA
#     worker_lookup = {w.id: w for w in workers}
#     for w_id, instr in instruction_maps.items():
#         receiver = worker_lookup.get(w_id)
#         if not receiver: continue

#         cr = instr.get('compression_ratio', 1.0) # Mô phỏng nén
#         assigned_neighbors = instr.get('neighbors', [])

#         for neighbor_id in assigned_neighbors:
#             sender = worker_lookup.get(neighbor_id)
#             if sender:
#                 sender_cluster = sender.cluster_id
#                 sender_params = worker_updates_cache[neighbor_id]

#                 receiver.apply_dfca_gossip_update(sender_cluster, sender_params)
#                 print(f"{w_id} pulled model from neighbor {neighbor_id}")

#         final_model_state = {k: v.cpu().clone() for k, v in receiver.model.state_dict().items()}
#         final_updates_for_ch[w_id] = final_model_state
    
#     # Gửi toàn bộ model đã xử lý về cho Cluster Head
#     for w_id, model_state in final_updates_for_ch.items():
#         worker = worker_lookup[w_id]
#         target_ch = cluster_heads[worker.cluster_id]
        
#         # CH nhận model cuối cùng để chạy FedAvg
#         target_ch.receive_update(w_id, model_state)
        
#     round_results = []
    
#     # Danh sách tạm để tính trung bình toàn mạng cho Dashboard
#     cluster_accuracies = []
#     cluster_losses = []
    
#     for ch in cluster_heads:
#         agg_model_state, model_hash = ch.aggregate(round_k=current_round)
        
#         # --- Giả lập Metrics (Để vẽ biểu đồ) ---
#         # Accuracy tăng dần theo thời gian (giới hạn 95%)
#         simulated_acc = min(95.0, 15.0 + current_round * 2.5 + random.uniform(-2, 3))
        
#         # Loss giảm dần theo thời gian
#         simulated_loss = max(0.1, 1.5 - current_round * 0.05 + random.uniform(-0.05, 0.05))
        
#         cluster_accuracies.append(simulated_acc)
#         cluster_losses.append(simulated_loss)
        
#         round_results.append({
#             "cluster_id": ch.cluster_id,
#             "hash": model_hash,
#             "accuracy": simulated_acc
#         })

#     # PHA 5: BLOCKCHAIN CONSENSUS (Smart Contract)
#     print("[Phase 5] Blockchain Consensus...")
    
#     consensus_log = []
    
#     for res in round_results:
#         success = blockchain.propose_update(
#             cluster_id=res['cluster_id'],
#             aggregated_model_hash=res['hash'],
#             accuracy=res['accuracy']
#         )
#         status = "Accepted" if success else "Rejected"
#         consensus_log.append(f"Cluster {res['cluster_id']}: {status} (Acc: {res['accuracy']:.2f}%)")

#     # TỔNG HỢP METRICS & LƯU HISTORY (PHỤC VỤ DASHBOARD)
    
#     # 1. Tính trung bình toàn mạng
#     avg_accuracy = sum(cluster_accuracies) / len(cluster_accuracies) if cluster_accuracies else 0
#     avg_loss = sum(cluster_losses) / len(cluster_losses) if cluster_losses else 0
    
#     # 2. Cập nhật biến toàn cục history
#     history["rounds"].append(current_round)
#     history["accuracy"].append(round(avg_accuracy, 2))
#     history["loss"].append(round(avg_loss, 4))
#     history["blockchain_height"].append(len(blockchain.chain))
    
#     # 3. Lấy Reputation của CH0 để hiển thị demo
#     ch0_reputation = blockchain.reputation_scores.get("ClusterHead_0", 0)

#     # --- KẾT THÚC VÒNG ---
#     execution_time = time.time() - start_time
#     print(f"Round {current_round} finished in {execution_time:.2f}s. Avg Acc: {avg_accuracy:.2f}%")

#     return jsonify({
#         # --- Data cho Logic Mới ---
#         "status": "success",
#         "round": current_round,
#         "execution_time": execution_time,
#         "topology": all_topologies,         # Dữ liệu vẽ CoCo
#         "logs": consensus_log,              # Log Blockchain
        
#         # --- Data cho Dashboard Cũ ---
#         "current_accuracy": round(avg_accuracy, 2),
#         "current_loss": round(avg_loss, 4),
#         "ch0_reputation": ch0_reputation,
#         "reputation": blockchain.reputation_scores, # Gửi cả bảng điểm về
#         "history": history # Gửi full history để frontend vẽ lại chart nếu cần
#     })

# ENDPOINT để lấy lịch sử metrics
@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    """Trả về toàn bộ dữ liệu huấn luyện từ trước đến giờ"""
    return jsonify(history)

# ENDPOINT để hiển thị Dashboard
@app.route('/dashboard')
def dashboard_view():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)