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

# Biscotti imports
from app.biscotti.blockchain import Blockchain as BiscottiBlockchain
from app.biscotti.vrf import VRF
from app.biscotti.krum import KRUM
from app.biscotti.federated import FederatedNode, aggregate_updates
from app.biscotti.rpc import NodeCommunicator
from app.biscotti.evaluator import GlobalEvaluator
from app.models.cnn import get_model

import json
import os
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)

# Khởi tạo hệ thống giả lập
engine = SimulationEngine()


# BIẾN TOÀN CỤC LƯU LỊCH SỬ
history = {
    "rounds": [],
    "system_mode": [],
    "execution_time": [],
    "avg_acc": [],
    "avg_loss": [],
    "max_ter": [],
    "asr": [],
    "f1": [],
    "auc": [],
    "src_recall": [],
    "tgt_precision": [],
    "blockchain_height": []
}
# Biến lưu tên file hiện tại
history_filename = None

def save_history(system_mode, agg_algo=None, attack_type=None):
    global history_filename
    
    # Nếu chưa có file thì tạo tên mới với timestamp
    if history_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("histories", exist_ok=True)
        if not attack_type:
            history_filename = f"histories/history_{timestamp}_{system_mode}.json"
        else:
            if agg_algo:
                history_filename = f"histories/history_{timestamp}_{attack_type}_{agg_algo}.json"
            else:
                history_filename = f"histories/history_{timestamp}_{attack_type}_proposed.json"
    
    with open(history_filename, "w") as f:
        json.dump(history, f, indent=4)


def load_history(filename=None):
    global history, history_filename
    
    if filename is not None:
        history_filename = filename
    
    if history_filename and os.path.exists(history_filename):
        with open(history_filename, "r") as f:
            history = json.load(f)
    else:
        history = defaultdict(list)

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
            attack_type = req_data.get('attack_type', None)
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
                history_filename = None

                if r != 0:
                    req_data['reset'] = False
                
                # Gọi engine để chạy 1 vòng
                result = engine.run_round(r, req_data)
                
                # Cập nhật History
                update_history_dynamic(history, r, result, requested_mode)
                if attack_type:
                    agg_algo = req_data.get('aggregation_algorithm')
                    save_history(system_mode=requested_mode, agg_algo=agg_algo, attack_type=attack_type)
                else:
                    save_history(system_mode=requested_mode)
                
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
    # print(req_data, Config.NUM_CLUSTERS, "WTF", flush=True)
    # print(total_rounds, "CLGV", flush=True)
    # print(f"DEBUG: Total Rounds nhận được = {total_rounds} | CLGV", file=sys.stderr)
    
    # Chạy thread ngầm
    thread = threading.Thread(
        target=training_loop,
        args=(total_rounds, req_data)
    )
    thread.start()

    return jsonify({"status": "started", "message": "Training started in background"})

# --- API MỚI: KIỂM TRA TRẠNG THÁI ---
@app.route('/training_status', methods=['GET'])
def get_training_status():
    return jsonify(training_status)


@app.route('/run_biscotti', methods=['POST'])
def run_biscotti():
    import time
    import random
    
    # Simple Biscotti simulation with metrics
    bc = BiscottiBlockchain()
    vrf = VRF()
    krum = KRUM()
    nodes = [FederatedNode(model_name='simple_cnn') for _ in range(3)]  # 3 nodes
    stake_map = {i: 1 for i in range(3)}

    # Khởi tạo Evaluator với cấu trúc mô hình tương ứng
    base_model = get_model('simple_cnn', 10) 
    evaluator = GlobalEvaluator(base_model)
    
    ATTACK_TYPE = "LABEL_FLIPPING" # Hoặc "GAUSS", "BACKDOOR"
    MALICIOUS_RATIO = 0.3
    TOTAL_NODES = 30
    malicious_count = int(TOTAL_NODES * MALICIOUS_RATIO)
    
    # Giả định các node đầu tiên (0 đến malicious_count-1) là node độc hại
    malicious_node_ids = set(range(malicious_count))

    for iteration in range(1, MAX_ITERATIONS + 1):
        start_time = time.time()
        updates = []
        
        for node_id, node in enumerate(nodes):
            is_malicious = node_id in malicious_node_ids
            # Cần chỉnh sửa hàm train_local trong federated.py để nhận cờ is_malicious và attack_type
            delta = node.train_local(bc.get_latest_global_weights(), noise_scale=1.0, is_malicious=is_malicious, attack_type=ATTACK_TYPE)
            updates.append(delta)
            
        # Krum kiểm tra và trả về cả indices
        accepted, accepted_indices = krum.validate(updates)
        
        # Tính Max.TER (Toxic Error Rate) = Số lượng nút độc hại lọt qua / Tổng số nút lọt qua
        malicious_accepted = sum(1 for idx in accepted_indices if idx in malicious_node_ids)
        max_ter = malicious_accepted / len(accepted_indices) if len(accepted_indices) > 0 else 0.0

        # Cập nhật và tính toán Global Model
        new_weights = aggregate_updates(accepted, bc.get_latest_global_weights())
        bc.add_block({'iteration': iteration, 'global_weights': new_weights}, stake_map)
        
        # ---------------------------------------------------------
        # CHẠY ĐÁNH GIÁ THỰC TẾ TRÊN TẬP TEST
        # ---------------------------------------------------------
        metrics = evaluator.evaluate(new_weights, attack_type=ATTACK_TYPE, src_class=3, tgt_class=5)
        
        training_time = time.time() - start_time
        
        # Lưu vào lịch sử
        history["rounds"].append(iteration)
        history["system_mode"].append("BISCOTTI")
        history["execution_time"].append(training_time)
        history["avg_acc"].append(metrics["avg_acc"])
        history["avg_loss"].append(metrics["avg_loss"])
        history["max_ter"].append(max_ter)
        history["asr"].append(metrics["asr"])
        history["f1"].append(metrics["f1"])
        history["auc"].append(metrics["auc"])
        history["src_recall"].append(metrics["src_recall"])
        history["tgt_precision"].append(metrics["tgt_precision"])
        history["blockchain_height"].append(len(bc.chain))

    # Save history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("histories", exist_ok=True)
    history_filename = f"histories/history_{timestamp}_BISCOTTI.json"
    with open(history_filename, "w") as f:
        json.dump(history, f, indent=4)
    
    return jsonify({
        "message": "Biscotti simulation completed", 
        "blocks": len(bc.chain),
        "history_file": history_filename,
        "metrics": history
    })


# ================ ABLATION STUDY ENDPOINTS ================
@app.route('/run_ablation_study', methods=['POST'])
def run_ablation_study():
    """
    Endpoint để chạy các scenario ablation (loại bỏ từng tính năng)
    
    Expected JSON:
    {
        "bypass_mode": <int>,  # 0=Full, 1=NoClustering, 2=NoPrivacy, 4=NoByzantine, 8=NoBlockchain, 15=TraditionalDFL
        "total_rounds": <int>,
        "num_workers": <int>,
        "dataset": <str>,
        "model": <str>,
        ...other engine config...
    }
    """
    from app.core.bypass_ablation import BypassConfig
    
    req_data = request.json
    bypass_mode = int(req_data.get("bypass_mode", 0))
    total_rounds = int(req_data.get("total_rounds", 10))
    scenario_name = BypassConfig.get_name(bypass_mode)
    
    def ablation_loop():
        global training_status, history
        
        with app.app_context():
            try:
                training_status["is_running"] = True
                training_status["total_rounds"] = total_rounds
                training_status["message"] = f"Running {scenario_name} ablation study..."
                
                print(f"\n{'='*60}")
                print(f"[Ablation Study] {scenario_name} (bypass_mode={bypass_mode})")
                print(f"{'='*60}")
                
                # Khởi tạo engine với bypass mode
                engine.initialize_system(req_data)
                
                # Reset history
                history = defaultdict(list)
                
                # Vòng lặp training
                for r in range(total_rounds):
                    training_status["current_round"] = r + 1
                    training_status["message"] = f"{scenario_name}: Round {r+1}/{total_rounds}..."
                    
                    if r != 0:
                        req_data['reset'] = False
                    
                    # Chạy round
                    result = engine.run_round(r, req_data)
                    
                    # Cập nhật history
                    update_history_dynamic(history, r, result, req_data.get('system_mode', 'PROPOSED'))
                    
                    print(f" -> Round {r+1} ({scenario_name}) finished. Acc: {result.get('avg_acc', 0):.4f}")
                
                # Lưu history
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("histories", exist_ok=True)
                history_filename = f"histories/ablation_{timestamp}_{scenario_name}.json"
                
                with open(history_filename, "w") as f:
                    json.dump({
                        "ablation_scenario": scenario_name,
                        "bypass_mode": bypass_mode,
                        "total_rounds": total_rounds,
                        "metrics": dict(history),
                        "bypass_report": engine.bypass_executor.get_report() if engine.bypass_executor else {}
                    }, f, indent=4)
                
                training_status["message"] = f"{scenario_name} completed"
                training_status["output_file"] = history_filename
                
                print(f"[Ablation Study] Results saved to {history_filename}")
                
            except Exception as e:
                print(f"[Ablation Error] {str(e)}")
                import traceback
                traceback.print_exc()
                training_status["message"] = f"Error: {str(e)}"
            finally:
                training_status["is_running"] = False
    
    # Chạy trong thread ngầm
    thread = threading.Thread(target=ablation_loop)
    thread.start()
    
    return jsonify({
        "status": "started",
        "message": f"Ablation study '{scenario_name}' started in background",
        "bypass_mode": bypass_mode,
        "scenario": scenario_name
    })


@app.route('/list_bypass_modes', methods=['GET'])
def list_bypass_modes():
    """
    Liệt kê tất cả các bypass modes có sẵn
    """
    from app.core.bypass_ablation import BypassConfig
    
    modes = {
        0: "Full_Features - Chạy đầy đủ tính năng (CoCo + Privacy + Byzantine + Blockchain)",
        1: "No_Clustering - Tất cả nodes vào 1 cụm (No Dynamic Clustering)",
        2: "No_Privacy - Không LDP/SSS (gửi gradient sạch)",
        4: "No_Byzantine - Dùng FedAvg thay vì BALANCE",
        8: "No_Blockchain - Lưu model vào RAM thay vì blockchain",
        15: "Traditional_DFL - Tất cả bypass (trở thành DFL truyền thống)"
    }
    
    return jsonify({
        "available_modes": modes,
        "description": "Sử dụng bypass_mode trong request /run_ablation_study để chọn scenario"
    })


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
    app.run(host=Config.HOST, port=PORT, debug=False, use_reloader=False)