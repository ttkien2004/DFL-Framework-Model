# File chạy chính
from flask import Flask, jsonify, request, render_template
from app.core.worker import WorkerNode
from app.core.cluster_head import ClusterHead
from app.blockchain.consensus import Blockchain
import random # Thêm tạm thời để tính loss
from config import Config
from app.utils.helpers import model_to_json

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
    return "BCFL CoCo Cluster Simulation API is Running!"

@app.route('/run_round', methods=['POST'])
def run_simulation_round():
    """Chạy 1 vòng lặp và LƯU kết quả vào history"""
    
    # --- [Phần logic chạy hệ thống giữ nguyên] ---
    # 1. Clustering
    cluster_models = {ch.cluster_id: ch.global_model.state_dict() for ch in cluster_heads}
    for w in workers:
        w.join_cluster(cluster_models)

    # 2. Training & LDP
    for w in workers:
        params = w.train()
        noisy_params = w.apply_ldp(params)
        target_ch = next(ch for ch in cluster_heads if ch.cluster_id == w.cluster_id)
        target_ch.receive_update(w.id, noisy_params)

    # 3. Aggregation & Metrics Calculation
    current_accuracies = []
    current_losses = []
    
    aggregated_updates = {}
    for ch in cluster_heads:
        agg_model = ch.aggregate()
        aggregated_updates[ch.cluster_id] = str(agg_model)[:50]
        model_hash = str(agg_model)[:50] # Hash giả lập
        
        # --- Giả lập lấy metrics từ Cluster Head ---
        # Trong thực tế, bạn sẽ lấy ch.current_accuracy, ch.current_loss
        # Ở đây tôi random để bạn thấy biểu đồ chạy
        acc = random.uniform(20, 35) + len(blockchain.chain) * 0.5 # Giả lập acc tăng dần
        loss = max(0.1, 1.0 - len(blockchain.chain) * 0.02)        # Giả lập loss giảm dần
        simulated_accuracy = min(90, 20 + len(blockchain.chain) * 2 + random.uniform(-2, 5))
        # 4. Gửi đề xuất lên Blockchain (MỚI: Truyền accuracy và ID cụ thể)
        success = blockchain.propose_update(
            cluster_id=ch.cluster_id, 
            aggregated_model_hash=model_hash,
            accuracy=simulated_accuracy
        )
        if success:
            round_status = "Success"
            avg_acc = simulated_accuracy
        
        current_accuracies.append(acc)
        current_losses.append(loss)

    # Tính trung bình toàn mạng
    avg_accuracy = sum(current_accuracies) / len(current_accuracies)
    avg_loss = sum(current_losses) / len(current_losses)

    # 4. Blockchain Consensus
    # success = blockchain.propose_update(aggregated_updates)
    
    # Lưu metrics vào lịch sử
    current_round = len(blockchain.chain)

    ch0_score = blockchain.reputation_scores.get("ClusterHead_0", 0)
    
    history["rounds"].append(current_round)
    history["accuracy"].append(round(avg_accuracy, 2))
    history["loss"].append(round(avg_loss, 4))
    history["blockchain_height"].append(current_round)
    
    return jsonify({
        "status": round_status, 
        "round": len(blockchain.chain),
        "current_accuracy": avg_accuracy,
        "ch0_reputation": ch0_score
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