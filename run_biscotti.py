import argparse
import time
import json
import os
from datetime import datetime
from app.biscotti.blockchain import Blockchain
from app.biscotti.krum import KRUM
from app.biscotti.federated import FederatedNode, aggregate_updates
from app.biscotti.evaluator import GlobalEvaluator
from app.models.cnn import get_model

from app.utils.data_spliter import dirichlet_split_noniid 
from app.utils.data_loader import PersonalHealthDataset
from torchvision import datasets, transforms
from torch.utils.data import random_split
import torch
import numpy as np

def run_simulation(args):
    print("="*60)
    print(f"=== BẮT ĐẦU THỰC NGHIỆM BISCOTTI ===")
    print(f"Dataset: {args.dataset.upper()} | Attack: {args.attack_type}")
    print(f"Nodes: {args.num_nodes} | Malicious Ratio: {args.malicious_ratio}")
    print(f"Iterations: {args.max_iterations}")
    print("="*60)

    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        num_classes = 10
        model_name = 'simple_cnn'
        
    elif args.dataset == 'health':
        csv_path = './data/personal_health_data.csv'
        full_dataset = PersonalHealthDataset(csv_path)
        
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        generator = torch.Generator().manual_seed(42)
        full_train_dataset, _ = random_split(full_dataset, [train_size, test_size], generator=generator)
        full_train_dataset.targets = np.array([full_dataset.y[i].item() for i in full_train_dataset.indices])
        
        num_classes = 2
        model_name = 'health_mlp'
    else:
        raise ValueError("Dataset không hợp lệ")
    
    # Gọi hàm chia Dirichlet cho `num_nodes` phần với alpha=0.5
    print(f"[*] Đang chia dữ liệu bằng Dirichlet (alpha=0.5) cho {args.num_nodes} nodes...")
    client_datasets = dirichlet_split_noniid(full_train_dataset, num_clients=args.num_nodes, alpha=0.5, num_classes=num_classes)
    
    # 1. Khởi tạo mạng lưới
    bc = Blockchain()
    krum = KRUM(num_adversaries=args.malicious_ratio)
    
    # Khởi tạo các Node, truyền dataset_name vào
    nodes = []
    for i in range(args.num_nodes):
        node = FederatedNode(model_name=model_name, num_classes=num_classes, dataset_name=args.dataset)
        node.set_local_dataset(client_datasets[i]) # Gán dữ liệu cục bộ
        nodes.append(node)
    
    # 2. Phân loại nút độc hại
    malicious_count = int(args.num_nodes * args.malicious_ratio)
    malicious_node_ids = set(range(malicious_count))
    stake_map = {i: 10 for i in range(args.num_nodes)} 
    
    # 3. Khởi tạo Evaluator với đúng dataset
    base_model = get_model(model_name=model_name,num_classes=num_classes) 
    evaluator = GlobalEvaluator(base_model, dataset_name=args.dataset)
    
    # 4. Lưu trữ lịch sử (Ghi thêm thông tin cấu hình vào json)
    history = {
        "config": vars(args), # Lưu lại toàn bộ cấu hình chạy
        "rounds": [], "execution_time": [], "avg_acc": [], "avg_loss": [], 
        "max_ter": [], "asr": [], "f1": [], "auc": [], "src_recall": [], "tgt_precision": [],
        "gia_recon_mse": [], "gia_recon_psnr": []
    }

    # --- VÒNG LẶP HUẤN LUYỆN ---
    for iteration in range(1, args.max_iterations + 1):
        start_time = time.time()
        print(f"\n--- Đang chạy Vòng {iteration}/{args.max_iterations} ---")
        
        updates = []
        global_w = bc.get_latest_global_weights()
        
        for node_id, node in enumerate(nodes):
            print(f"  -> Đang train Node {node_id + 1}/{args.num_nodes}...")
            is_malicious = node_id in malicious_node_ids
            # Truyền các tham số tấn công động
            delta = node.train_local(
                global_w, 
                noise_scale=1.0, 
                is_malicious=is_malicious, 
                attack_type=args.attack_type,
                src_class=args.src_class, 
                tgt_class=args.tgt_class,
                gia_iterations=args.gia_iterations,
                gia_lr=args.gia_lr
            )
            updates.append(delta)
            
        # KRUM kiểm tra
        accepted, accepted_indices = krum.validate(updates)
        malicious_accepted = sum(1 for idx in accepted_indices if idx in malicious_node_ids)
        max_ter = malicious_accepted / len(accepted_indices) if len(accepted_indices) > 0 else 0.0
        
        print(f"KRUM chấp nhận: {len(accepted_indices)}/{args.num_nodes} nút | Lọt lưới (Max.TER): {max_ter:.4f}")

        # Tổng hợp & Thêm block
        new_weights = aggregate_updates(accepted, global_w)
        bc.add_block({'iteration': iteration, 'global_weights': new_weights}, stake_map)
        
        # Đánh giá Test set
        metrics = evaluator.evaluate(new_weights, attack_type=args.attack_type, src_class=args.src_class, tgt_class=args.tgt_class)
        exec_time = time.time() - start_time
        print(f"Accuracy: {metrics['avg_acc']:.4f} | F1: {metrics['f1']:.4f} | ASR: {metrics['asr']:.4f} | Time: {exec_time:.2f}s")
        
        # Ghi logs
        history["rounds"].append(iteration)
        history["execution_time"].append(exec_time)
        history["avg_acc"].append(metrics["avg_acc"])
        history["avg_loss"].append(metrics["avg_loss"])
        history["max_ter"].append(max_ter)
        history["asr"].append(metrics["asr"])
        history["f1"].append(metrics["f1"])
        history["auc"].append(metrics["auc"])
        history["src_recall"].append(metrics["src_recall"])
        history["tgt_precision"].append(metrics["tgt_precision"])

        if args.attack_type in ["GIA", "GRADIENT_INVERSION"]:
            gi_metrics = [node.gia_metrics for node in nodes if getattr(node, 'gia_metrics', None) is not None]
            if gi_metrics:
                avg_mse = sum(m['recon_mse'] for m in gi_metrics) / len(gi_metrics)
                avg_psnr = sum(m['recon_psnr'] for m in gi_metrics) / len(gi_metrics)
            else:
                avg_mse, avg_psnr = 0.0, 0.0
            history["gia_recon_mse"].append(avg_mse)
            history["gia_recon_psnr"].append(avg_psnr)
        else:
            history["gia_recon_mse"].append(0.0)
            history["gia_recon_psnr"].append(0.0)

    # --- LƯU KẾT QUẢ KHI CHẠY XONG ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("histories", exist_ok=True)
    filename = f"histories/history_{timestamp}_{args.attack_type}_{args.dataset.upper()}_BISCOTTI.json"
    with open(filename, "w") as f:
        json.dump(history, f, indent=4)
    print(f"\n[HOÀN THÀNH] Đã lưu kết quả tại: {filename}")

if __name__ == "__main__":
    # KHAI BÁO CÁC THAM SỐ DÒNG LỆNH
    parser = argparse.ArgumentParser(description="Biscotti Federated Learning Simulation")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "health"], help="Tập dữ liệu: mnist hoặc health")
    parser.add_argument("--attack-type", type=str, default="NONE", choices=["NONE", "GAUSS", "BACKDOOR", "LABEL_FLIPPING", "GIA", "GRADIENT_INVERSION"], help="Loại tấn công")
    parser.add_argument("--num-nodes", type=int, default=30, help="Tổng số nút tham gia")
    parser.add_argument("--malicious-ratio", type=float, default=0.3, help="Tỷ lệ nút độc hại (Ví dụ 0.3 = 30%)")
    parser.add_argument("--max-iterations", type=int, default=100, help="Số vòng giao tiếp tối đa")
    parser.add_argument("--src-class", type=int, default=3, help="Nhãn gốc (dùng cho Label Flipping)")
    parser.add_argument("--tgt-class", type=int, default=5, help="Nhãn mục tiêu (dùng cho Backdoor/Label Flipping)")
    parser.add_argument("--gia-iterations", type=int, default=2000, help="Số vòng lặp Gradient Inversion")
    parser.add_argument("--gia-lr", type=float, default=0.1, help="Learning rate cho Gradient Inversion")
    
    args = parser.parse_args()
    run_simulation(args)