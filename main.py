# File chạy chính
from flask import Flask, jsonify, request, render_template
from app.core.worker import WorkerNode
from app.core.cluster_head import ClusterHead
from app.blockchain.consensus import Blockchain
import random # Thêm tạm thời để tính loss
from config import Config
from app.utils.helpers import model_to_json
import time

app = Flask(__name__)

# Khởi tạo hệ thống giả lập
workers = [WorkerNode(i, None) for i in range(Config.NUM_WORKERS)]
cluster_heads = [ClusterHead(i) for i in range(Config.NUM_CLUSTERS)]
blockchain = Blockchain()

# BIẾN TOÀN CỤC LƯU LỊCH SỬ
history = {
    "rounds": [],
    "accuracy": [],
    "loss": [],
    "blockchain_height": []
}

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
    """
    LOGIC CHÍNH: Chạy một vòng huấn luyện (Round)
    Bao gồm 5 Pha: Clustering -> CoCo -> Training/LDP -> Aggregation/BALANCE -> Consensus
    """
    start_time = time.time()
    current_round = len(blockchain.chain)
    print(f"\n---STARTING ROUND {current_round} ---")

    # PHA 1: CLUSTERING (DFCA - Dynamic Federated Clustering)
    print("[Phase 1] Clustering...")
    
    # Reset thành viên của các CH vòng trước
    for ch in cluster_heads:
        ch.members = []
        ch.member_metrics = {} 

    # Worker chọn Cụm
    for w in workers:
        if w.cluster_id is None or random.random() < 0.1:
            w.cluster_id = random.randint(0, Config.NUM_CLUSTERS - 1)
        target_ch = cluster_heads[w.cluster_id]
        target_ch.register_member(w.id)

    # PHA 2: CoCo OPTIMIZATION (Topology & Compression)
    print("[Phase 2] CoCo Optimization...")
    
    all_topologies = {} 
    instruction_maps = {} 

    # B1: Worker báo cáo trạng thái
    for w in workers:
        metrics = {
            'bandwidth': random.uniform(20, 100), 
            'cpu_load': random.uniform(10, 80)
        }
        target_ch = cluster_heads[w.cluster_id]
        target_ch.receive_metrics(w.id, metrics, w.model.state_dict())

    # B2: CH chạy thuật toán tối ưu
    for ch in cluster_heads:
        instructions, topology_viz = ch.run_coco_optimization()
        instruction_maps.update(instructions)
        all_topologies[ch.cluster_id] = topology_viz

    # PHA 3: TRAINING & LDP (Local Differential Privacy)
    print("[Phase 3] Local Training & LDP...")
    
    for w in workers:
        if w.id in instruction_maps:
            instr = instruction_maps[w.id]
            trained_params = w.train() 
            noisy_params = w.apply_ldp(trained_params)
            target_ch = cluster_heads[w.cluster_id]
            target_ch.receive_update(w.id, noisy_params)

    # PHA 4: AGGREGATION & BALANCE FILTERING
    print("[Phase 4] Aggregation & BALANCE Filtering...")
    
    round_results = []
    
    # Danh sách tạm để tính trung bình toàn mạng cho Dashboard
    cluster_accuracies = []
    cluster_losses = []
    
    for ch in cluster_heads:
        agg_model_state, model_hash = ch.aggregate(round_k=current_round)
        
        # --- Giả lập Metrics (Để vẽ biểu đồ) ---
        # Accuracy tăng dần theo thời gian (giới hạn 95%)
        simulated_acc = min(95.0, 15.0 + current_round * 2.5 + random.uniform(-2, 3))
        
        # Loss giảm dần theo thời gian
        simulated_loss = max(0.1, 1.5 - current_round * 0.05 + random.uniform(-0.05, 0.05))
        
        cluster_accuracies.append(simulated_acc)
        cluster_losses.append(simulated_loss)
        
        round_results.append({
            "cluster_id": ch.cluster_id,
            "hash": model_hash,
            "accuracy": simulated_acc
        })

    # PHA 5: BLOCKCHAIN CONSENSUS (Smart Contract)
    print("[Phase 5] Blockchain Consensus...")
    
    consensus_log = []
    
    for res in round_results:
        success = blockchain.propose_update(
            cluster_id=res['cluster_id'],
            aggregated_model_hash=res['hash'],
            accuracy=res['accuracy']
        )
        status = "Accepted" if success else "Rejected"
        consensus_log.append(f"Cluster {res['cluster_id']}: {status} (Acc: {res['accuracy']:.2f}%)")

    # ==============================================================================
    # TỔNG HỢP METRICS & LƯU HISTORY (PHỤC VỤ DASHBOARD)
    # ==============================================================================
    
    # 1. Tính trung bình toàn mạng
    avg_accuracy = sum(cluster_accuracies) / len(cluster_accuracies) if cluster_accuracies else 0
    avg_loss = sum(cluster_losses) / len(cluster_losses) if cluster_losses else 0
    
    # 2. Cập nhật biến toàn cục history
    history["rounds"].append(current_round)
    history["accuracy"].append(round(avg_accuracy, 2))
    history["loss"].append(round(avg_loss, 4))
    history["blockchain_height"].append(len(blockchain.chain))
    
    # 3. Lấy Reputation của CH0 để hiển thị demo
    ch0_reputation = blockchain.reputation_scores.get("ClusterHead_0", 0)

    # --- KẾT THÚC VÒNG ---
    execution_time = time.time() - start_time
    print(f"Round {current_round} finished in {execution_time:.2f}s. Avg Acc: {avg_accuracy:.2f}%")

    return jsonify({
        # --- Data cho Logic Mới ---
        "status": "success",
        "round": current_round,
        "execution_time": execution_time,
        "topology": all_topologies,         # Dữ liệu vẽ CoCo
        "logs": consensus_log,              # Log Blockchain
        
        # --- Data cho Dashboard Cũ ---
        "current_accuracy": round(avg_accuracy, 2),
        "current_loss": round(avg_loss, 4),
        "ch0_reputation": ch0_reputation,
        "reputation": blockchain.reputation_scores, # Gửi cả bảng điểm về
        "history": history # Gửi full history để frontend vẽ lại chart nếu cần
    })

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