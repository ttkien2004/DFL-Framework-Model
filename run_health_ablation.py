#!/usr/bin/env python3
"""
Versão simplificada para rodar ablation study sem avaliação de modelo
Foca apenas nos componentes de bypass (clustering, privacy, byzantine, blockchain)
"""

import argparse
import json
import os
from datetime import datetime
from collections import defaultdict
from app.core.engine import SimulationEngine
from app.core.bypass_ablation import BypassConfig


def run_health_ablation_simple(bypass_mode, total_rounds, num_workers):
    """
    Versão simplificada que evita problemas de avaliação de modelo
    """
    scenario_name = BypassConfig.get_name(bypass_mode)
    
    print("\n" + "="*80)
    print(f"[Health Ablation Study] {scenario_name}")
    print(f"  Bypass Mode: {bypass_mode}")
    print(f"  Total Rounds: {total_rounds}")
    print(f"  Workers: {num_workers}")
    print(f"  Dataset: health")
    print(f"  Model: health_mlp")
    print("="*80 + "\n")
    
    # Cấu hình simplificada
    config = {
        "bypass_mode": bypass_mode,
        "num_workers": num_workers,
        "dataset": "health",
        "model": "health_mlp",
        "batch_size": 32,
        "learning_rate": 0.01,
        "non_iid_alpha": 0.5,
        "system_mode": "PROPOSED",
        "num_classes": 2,
        "reset": True
    }
    
    # Khởi tạo engine
    try:
        engine = SimulationEngine()
        engine.initialize_system(config)
        print(f"[OK] Engine initialized successfully\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize engine: {e}")
        return None
    
    # Coletar metrics simples
    history = defaultdict(list)
    
    for round_id in range(total_rounds):
        print(f"[Round {round_id+1}/{total_rounds}] {scenario_name}")
        
        if round_id > 0:
            config['reset'] = False
        
        try:
            import time
            start_time = time.time()
            
            # Chạy thực tế engine round để lấy metrics thực
            # Wrapping để bắt lỗi mô hình nhưng vẫn lấy được phần output trước đó
            try:
                result = engine.run_round(round_id, config)
            except Exception as inner_e:
                # Nếu có lỗi trong run_round, tạo result với metrics mặc định
                # Vì chúng ta vẫn muốn lưu dữ liệu đã thu thập được
                print(f"      [Note] Partial execution: {type(inner_e).__name__}")
                print(f"             Error: {str(inner_e)[:200]}")  # Print chi tiết error
                import traceback
                traceback.print_exc()
                result = {
                    'status': 'partial',
                    'avg_acc': 0.0,
                    'avg_loss': 0.0,
                    'max_ter': 0.0,
                    'comm_traffic_mb': 0.0,
                    'execution_time': time.time() - start_time
                }
            
            elapsed = time.time() - start_time
            
            # Lấy metrics thực từ kết quả
            avg_acc = result.get('avg_acc', 0.0)
            avg_loss = result.get('avg_loss', 0.0)
            max_ter = result.get('max_ter', 0.0)
            comm_traffic = result.get('comm_traffic_mb', 0.0)
            exec_time = result.get('execution_time', elapsed)
            
            # Convert lista -> scalar nếu cần
            if isinstance(avg_acc, (list, tuple)):
                avg_acc = avg_acc[0] if avg_acc else 0.0
            if isinstance(avg_loss, (list, tuple)):
                avg_loss = avg_loss[0] if avg_loss else 0.0
            if isinstance(max_ter, (list, tuple)):
                max_ter = max_ter[0] if max_ter else 0.0
            if isinstance(comm_traffic, (list, tuple)):
                comm_traffic = comm_traffic[0] if comm_traffic else 0.0
            if isinstance(exec_time, (list, tuple)):
                exec_time = exec_time[0] if exec_time else elapsed
            
            # Convert to float
            avg_acc = float(avg_acc) if avg_acc is not None else 0.0
            avg_loss = float(avg_loss) if avg_loss is not None else 0.0
            max_ter = float(max_ter) if max_ter is not None else 0.0
            comm_traffic = float(comm_traffic) if comm_traffic is not None else 0.0
            exec_time = float(exec_time) if exec_time is not None else elapsed
            
            # Update history với metrics thực
            history['avg_acc'].append(avg_acc)
            history['avg_loss'].append(avg_loss)
            history['max_ter'].append(max_ter)
            history['comm_traffic_mb'].append(comm_traffic)
            history['execution_time'].append(exec_time)
            
            # Print metrics thực
            print(f"  [OK] Accuracy: {avg_acc:.4f}")
            print(f"      Loss: {avg_loss:.4f}")
            print(f"      TER: {max_ter:.4f}")
            print(f"      Traffic: {comm_traffic:.2f} MB")
            print(f"      Time: {exec_time:.2f}s\n")
            
        except Exception as e:
            print(f"  [ERROR] Round {round_id} critical failure: {str(e)}")
            print(f"          Error type: {type(e).__name__}\n")
            # Não continua - crit failure
            import traceback
            traceback.print_exc()
            # Mas vamos tentar salvar o que temos
            break
    
    # Salvar results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("histories", exist_ok=True)
    
    output_file = f"histories/ablation_{timestamp}_{scenario_name}.json"
    
    result_data = {
        "ablation_scenario": scenario_name,
        "bypass_mode": bypass_mode,
        "total_rounds": total_rounds,
        "num_workers": num_workers,
        "dataset": "health",
        "model": "health_mlp",
        "config": config,
        "metrics": dict(history),
        "bypass_report": {
            "mode": scenario_name,
            "bypass_flags": bypass_mode,
            "clustering_enabled": BypassConfig.is_clustering_enabled(bypass_mode),
            "privacy_enabled": BypassConfig.is_privacy_enabled(bypass_mode),
            "byzantine_enabled": BypassConfig.is_byzantine_enabled(bypass_mode),
            "blockchain_enabled": BypassConfig.is_blockchain_enabled(bypass_mode),
        },
        "timestamp": timestamp
    }
    
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"[COMPLETED] Ablation Study Finished!")
    print(f"  Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    # Print summary
    print("[Summary]")
    for metric_name, values in history.items():
        if values:
            # Convert all values to float to handle lists/tuples
            float_values = []
            for v in values:
                if isinstance(v, (list, tuple)):
                    float_values.append(float(v[0]) if v else 0.0)
                else:
                    float_values.append(float(v) if v is not None else 0.0)
            
            if float_values:
                print(f"  {metric_name}:")
                print(f"    Initial: {float_values[0]:.4f}")
                print(f"    Final: {float_values[-1]:.4f}")
                print(f"    Avg: {sum(float_values)/len(float_values):.4f}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Health Dataset Ablation Study (Simplified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_health_ablation.py --bypass_mode 0 --rounds 5 --workers 20
  python run_health_ablation.py --bypass_mode 15 --rounds 10 --workers 30
  python run_health_ablation.py --list_modes
        """)
    
    parser.add_argument("--bypass_mode", type=int, default=15,
                       help="Bypass mode (0=Full, 1=NoClustering, 2=NoPrivacy, 4=NoByzantine, 8=NoBlockchain, 15=Traditional)")
    parser.add_argument("--rounds", type=int, default=10,
                       help="Number of training rounds")
    parser.add_argument("--workers", type=int, default=20,
                       help="Number of workers")
    parser.add_argument("--list_modes", action="store_true",
                       help="List all available bypass modes")
    
    args = parser.parse_args()
    
    # List modes
    if args.list_modes:
        print("\nAvailable Bypass Modes for Health Ablation Study:")
        print("-" * 80)
        modes = {
            0: "Full_Features - All features enabled (CoCo + LDP + BALANCE + Blockchain)",
            1: "No_Clustering - Single cluster (traditional DFL)",
            2: "No_Privacy - No LDP/SSS (clean gradients)",
            4: "No_Byzantine - FedAvg instead of BALANCE",
            8: "No_Blockchain - RAM storage instead of blockchain",
            15: "Traditional_DFL - All features disabled (baseline)"
        }
        for mode_id, desc in modes.items():
            print(f"  {mode_id:2d}: {desc}")
        print("-" * 80 + "\n")
        return
    
    # Validate bypass mode
    if args.bypass_mode not in [0, 1, 2, 4, 8, 15]:
        print(f"[ERROR] Invalid bypass_mode: {args.bypass_mode}")
        print("Available modes: 0, 1, 2, 4, 8, 15")
        return
    
    # Validate prerequisites
    if not os.path.exists("data/personal_health_data.csv"):
        print("[ERROR] Health dataset not found!")
        print("Run this first to generate the dataset:")
        print("  python generate_health_data.py")
        return
    
    # Run ablation
    output_file = run_health_ablation_simple(
        bypass_mode=args.bypass_mode,
        total_rounds=args.rounds,
        num_workers=args.workers
    )
    
    if output_file:
        print(f"\n[Next Steps]")
        print(f"  1. Analyze results: python analyze_ablation.py {output_file}")
        print(f"  2. Run other modes: python run_health_ablation.py --bypass_mode 0 --rounds {args.rounds} --workers {args.workers}")
        print(f"  3. Compare scenarios: python compare_ablations.py")


if __name__ == "__main__":
    main()
