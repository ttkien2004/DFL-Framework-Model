# File chạy chính
from platform import node
from platform import node
from flask import Flask, jsonify, request, render_template, redirect, url_for
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
    return redirect(url_for('config_page'))

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
        if requested_mode == "PROPOSED":
            blockchain = engine.blockchain
        else:
            blockchain = None
    else:
        # Nếu cùng mode, tiếp tục vòng lặp
        current_round = len(history["rounds"])

    result = engine.run_round(current_round, req_data)
    # Cập nhật History (UI Dashboard)
    update_history_dynamic(history, current_round, result, requested_mode)

    if blockchain is not None and blockchain.chain is not None:
        history['blockchain_height'].append(len(blockchain.chain))
    # history["system_mode"].append(requested_mode)

    return jsonify({
        "status": "success",
        "round": current_round,
        **result,
        "history": history
    })

@app.route('/homepage')
def config_page():
    return render_template('config.html')
# ENDPOINT để lấy lịch sử metrics
@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    """Trả về toàn bộ dữ liệu huấn luyện từ trước đến giờ"""
    return jsonify({
        "history": history
    })

# ENDPOINT để hiển thị Dashboard
@app.route('/dashboard')
def dashboard_view():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)