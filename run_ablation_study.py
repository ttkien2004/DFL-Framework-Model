#!/usr/bin/env python3
"""
Ablation Study Runner - Chạy các scenario bypass riêng biệt

Sử dụng:
    python run_ablation_study.py --bypass_mode 1 --rounds 10 --workers 20
    python run_ablation_study.py --bypass_mode 15 --rounds 5 --workers 10
"""

import argparse
import json
import os
from datetime import datetime
from collections import defaultdict
from app.core.engine import SimulationEngine
from app.core.bypass_ablation import BypassConfig, BypassExecutor
from app.utils.history import update_history_dynamic


def run_ablation(bypass_mode, total_rounds, num_workers, dataset, model, batch_size, learning_rate, non_iid_alpha, num_classes):
    """
    Chạy ablation study với bypass mode chỉ định
    """
    scenario_name = BypassConfig.get_name(bypass_mode)
    
    print("\n" + "="*80)
    print(f"[Ablation Study] {scenario_name}")
    print(f"  Bypass Mode: {bypass_mode}")
    print(f"  Total Rounds: {total_rounds}")
    print(f"  Workers: {num_workers}")
    print(f"  Dataset: {dataset}")
    print(f"  Model: {model}")
    print("="*80 + "\n")
    
    # Cấu hình
    config = {
        "bypass_mode": bypass_mode,
        "num_workers": num_workers,
        "dataset": dataset,
        "model": model,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "non_iid_alpha": non_iid_alpha,
        "system_mode": "PROPOSED",
        "num_classes": num_classes,
        "reset": True
    }
    
    # Khởi tạo engine
    engine = SimulationEngine()
    engine.initialize_system(config)
    
    # Chạy vòng lặp
    history = defaultdict(list)
    
    for round_id in range(total_rounds):
        print(f"\n[Round {round_id+1}/{total_rounds}] {scenario_name}")
        
        if round_id > 0:
            config['reset'] = False
        
        try:
            # Chạy round
            result = engine.run_round(round_id, config)
            
            # Cập nhật history
            update_history_dynamic(history, round_id, result, "PROPOSED")
            
            # In metrics
            acc = result.get('avg_acc', 0)
            loss = result.get('avg_loss', 0)
            ter = result.get('max_ter', 0)
            comm = result.get('comm_traffic_mb', 0)
            time = result.get('execution_time', 0)
            
            print(f"  Accuracy: {acc:.4f} | Loss: {loss:.4f} | TER: {ter:.4f}")
            print(f"  Traffic: {comm:.2f} MB | Time: {time:.2f}s")
            
        except Exception as e:
            print(f"  [ERROR] {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    # Lưu kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("histories", exist_ok=True)
    
    output_file = f"histories/ablation_{timestamp}_{scenario_name}.json"
    
    result_data = {
        "ablation_scenario": scenario_name,
        "bypass_mode": bypass_mode,
        "total_rounds": total_rounds,
        "config": config,
        "metrics": dict(history),
        "bypass_report": engine.bypass_executor.get_report() if engine.bypass_executor else {}
    }
    
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=4)
    
    print(f"\n[✓ Completed] Results saved to: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Ablation Study Runner")
    
    parser.add_argument("--bypass_mode", type=int, default=0,
                       help="Bypass mode (0=Full, 1=NoClustering, 2=NoPrivacy, 4=NoByzantine, 8=NoBlockchain, 15=TraditionalDFL)")
    parser.add_argument("--rounds", type=int, default=10,
                       help="Number of training rounds")
    parser.add_argument("--workers", type=int, default=20,
                       help="Number of workers")
    parser.add_argument("--dataset", type=str, default="mnist",
                       help="Dataset (mnist, cifar10, health)")
    parser.add_argument("--model", type=str, default="simple_cnn",
                       help="Model architecture")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                       help="Learning rate")
    parser.add_argument("--non_iid_alpha", type=float, default=0.5,
                       help="Non-IID alpha (Dirichlet parameter)")
    parser.add_argument("--num_classes", type=int, default=10,
                       help="Number of classes")
    parser.add_argument("--list_modes", action="store_true",
                       help="List all available bypass modes")
    
    args = parser.parse_args()
    
    # Liệt kê modes
    if args.list_modes:
        print("\nAvailable Bypass Modes:")
        print("-" * 80)
        modes = {
            0: "Full_Features - All features enabled (CoCo + LDP + BALANCE + Blockchain)",
            1: "No_Clustering - Single cluster (traditional DFL)",
            2: "No_Privacy - No LDP/SSS (clean gradients)",
            4: "No_Byzantine - FedAvg instead of BALANCE",
            8: "No_Blockchain - RAM storage instead of blockchain",
            15: "Traditional_DFL - All features disabled"
        }
        for mode_id, desc in modes.items():
            print(f"  {mode_id:2d}: {desc}")
        print("-" * 80)
        return
    
    # Validate bypass mode
    if args.bypass_mode not in [0, 1, 2, 4, 8, 15]:
        print(f"[ERROR] Invalid bypass_mode: {args.bypass_mode}")
        print("Available modes: 0, 1, 2, 4, 8, 15")
        return
    
    # Chạy ablation
    output_file = run_ablation(
        bypass_mode=args.bypass_mode,
        total_rounds=args.rounds,
        num_workers=args.workers,
        dataset=args.dataset,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        non_iid_alpha=args.non_iid_alpha,
        num_classes=args.num_classes
    )
    
    if output_file:
        print(f"\n[Summary]")
        print(f"  Scenario: {BypassConfig.get_name(args.bypass_mode)}")
        print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()
