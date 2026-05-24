#!/usr/bin/env python3
"""
run_all_ablations.py - Execute TODOS os 6 modos de bypass para comparação completa
"""

import subprocess
import json
import glob
import os
from datetime import datetime
from pathlib import Path

def run_ablation(mode, rounds=10, workers=20):
    """Executa um modo de ablação"""
    print(f"\n{'='*70}")
    print(f"  [Mode {mode}] Starting Ablation Study...")
    print(f"  Rounds: {rounds}, Workers: {workers}")
    print(f"{'='*70}")
    
    cmd = [
        "python", "run_health_ablation.py",
        "--bypass_mode", str(mode),
        "--rounds", str(rounds),
        "--workers", str(workers)
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def compare_results():
    """Compara e exibe todos os resultados"""
    print(f"\n{'='*70}")
    print("  COMPARING ALL RESULTS")
    print(f"{'='*70}\n")
    
    results = {}
    json_files = sorted(glob.glob("histories/ablation_*.json"))
    
    if not json_files:
        print("[WARN] Nenhum arquivo JSON encontrado em histories/")
        return
    
    print(f"{'Mode':<6} {'Scenario':<20} {'Accuracy':<12} {'Loss':<12} {'Traffic':<12}")
    print("-" * 62)
    
    for filepath in json_files:
        try:
            with open(filepath) as f:
                data = json.load(f)
                mode = data.get('bypass_mode', '?')
                scenario = data.get('ablation_scenario', 'Unknown')
                
                acc_list = data['metrics']['avg_acc']
                loss_list = data['metrics']['avg_loss']
                traffic_list = data['metrics']['comm_traffic_mb']
                
                final_acc = acc_list[-1] if acc_list else 0.0
                final_loss = loss_list[-1] if loss_list else 0.0
                avg_traffic = sum(traffic_list) / len(traffic_list) if traffic_list else 0.0
                
                results[mode] = {
                    'scenario': scenario,
                    'accuracy': final_acc,
                    'loss': final_loss,
                    'traffic': avg_traffic
                }
                
                print(f"{mode:<6} {scenario:<20} {final_acc:>10.4f}% {final_loss:>10.4f}  {avg_traffic:>10.2f} MB")
        
        except Exception as e:
            print(f"[ERROR] {filepath}: {e}")
    
    # Summary
    if results:
        print("\n" + "="*62)
        print("RANKING BY ACCURACY:")
        for mode, data in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            print(f"  Mode {mode}: {data['accuracy']:.4f}% ({data['scenario']})")

def main():
    print("\n" + "="*70)
    print("  DFL-FRAMEWORK: COMPLETE ABLATION STUDY")
    print(f"  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Configurações
    MODES = [0, 1, 2, 4, 8, 15]
    ROUNDS = 10
    WORKERS = 20
    
    print(f"\nConfiguração:")
    print(f"  Modes: {MODES}")
    print(f"  Rounds: {ROUNDS}")
    print(f"  Workers: {WORKERS}")
    print(f"\nTempo estimado: ~{len(MODES) * 2} minutos")
    
    # Executar
    success_count = 0
    failed_modes = []
    
    for mode in MODES:
        success = run_ablation(mode, rounds=ROUNDS, workers=WORKERS)
        if success:
            success_count += 1
        else:
            failed_modes.append(mode)
    
    # Resultados
    print(f"\n{'='*70}")
    print("  EXECUTION SUMMARY")
    print(f"{'='*70}")
    print(f"  Completed: {success_count}/{len(MODES)}")
    
    if failed_modes:
        print(f"  Failed modes: {failed_modes}")
    else:
        print(f"  Status: ALL MODES SUCCESSFUL! ✅")
    
    # Comparar resultados
    compare_results()
    
    print(f"\n{'='*70}")
    print(f"  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print("\nResultados salvos em: histories/ablation_*.json")

if __name__ == "__main__":
    main()
