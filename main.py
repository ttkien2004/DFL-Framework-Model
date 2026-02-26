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
import argparse
import threading

app = Flask(__name__)

# Khởi tạo hệ thống giả lập
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

# Biến theo dõi trạng thái
training_status = {
    "is_running": False,
    "current_round": 0,
    "total_rounds": 0,
    "message": "Idle"
}

# --- HÀM LOGIC CHẠY VÒNG LẶP TRONG BACKGROUND ---
def training_loop(total_rounds, req_data):
    global training_status, history, blockchain
    
    with app.app_context(): # Đảm bảo context cho Flask nếu cần truy cập DB/Config
        try:
            training_status["is_running"] = True
            training_status["total_rounds"] = total_rounds
            
            # 1. Khởi tạo lại hệ thống (Reset)
            requested_mode = req_data.get('system_mode', 'PROPOSED').upper()
            print(f"[Training Loop] Starting {total_rounds} rounds for {requested_mode}...")
            
            engine.initialize_system(req_data)
            
            # Reset history
            history = defaultdict(list)
            
            if requested_mode == "PROPOSED":
                blockchain = engine.blockchain
            else:
                blockchain = None # Baseline không có blockchain
            
            # 2. Vòng lặp Training
            for r in range(total_rounds):
                training_status["current_round"] = r + 1
                training_status["message"] = f"Running round {r+1}/{total_rounds}..."
                
                # Gọi engine để chạy 1 vòng
                result = engine.run_round(r, req_data)
                
                # Cập nhật History
                update_history_dynamic(history, r, result, requested_mode)
                
                if blockchain is not None and blockchain.chain is not None:
                    history['blockchain_height'].append(len(blockchain.chain))
                
                print(f" -> Round {r+1} finished. Acc: {result.get('accuracy', 0):.4f}")

            training_status["message"] = "Completed"
            
        except Exception as e:
            print(f"[Training Error] {str(e)}")
            training_status["message"] = f"Error: {str(e)}"
        finally:
            training_status["is_running"] = False

# --- API MỚI: BẮT ĐẦU TRAINING ---
@app.route('/start_training', methods=['POST'])
def start_training():
    if training_status["is_running"]:
        return jsonify({"status": "error", "message": "Training is already running!"}), 400

    req_data = request.json
    # Lấy số vòng từ request (do JS gửi lên)
    # Lưu ý: JS cần gửi key "target_rounds" hoặc "rounds"
    total_rounds = int(req_data.get("target_rounds", 10))
    print(req_data, Config.NUM_CLUSTERS, "WTF", flush=True)
    # print(total_rounds, "CLGV", flush=True)
    # print(f"DEBUG: Total Rounds nhận được = {total_rounds} | CLGV", file=sys.stderr)
    
    # Chạy thread ngầm
    thread = threading.Thread(
        target=training_loop,
        args=(total_rounds, req_data),
        daemon=True
    )
    thread.start()

    return jsonify({"status": "started", "message": "Training started in background"})

# --- API MỚI: KIỂM TRA TRẠNG THÁI ---
@app.route('/training_status', methods=['GET'])
def get_training_status():
    return jsonify(training_status)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--nc",
        type=int,
        default=Config.NUM_CLUSTERS,
        help="Số lượng cluster (K)"
    )
    args = parser.parse_args()

    PORT = args.port
    Config.NUM_CLUSTERS = args.nc
    app.run(host=Config.HOST, port=PORT, debug=True, use_reloader=True)