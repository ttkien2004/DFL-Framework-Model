#!/usr/bin/env python3
"""
Test script to verify GIA metrics collection in health ablation study.
Recommended settings based on user request:
  - 5 rounds
  - 2000 GIA iterations  
  - 0.1 attack rate (10% malicious workers)
  - 30 workers
"""

import subprocess
import sys
import json
from pathlib import Path

def test_gia_metrics():
    """Test GIA metrics collection with recommended settings."""
    
    print("\n" + "="*80)
    print("[TEST] GIA Metrics Collection")
    print("="*80 + "\n")
    
    # Recommended parameters from user
    cmd = [
        "python", "run_health_ablation.py",
        "--bypass_mode", "0",        # Full DFL
        "--rounds", "5",              # 5 rounds as user specified
        "--workers", "30",            # Typical setup
        "--attack_type", "GIA",       # Gradient Inversion Attack
        "--gia_iterations", "2000",   # User's typical value
        "--gia_lr", "1.0",           # Standard learning rate
        "--attack_rate", "0.1"        # 10% malicious workers
    ]
    
    print("[CONFIG]")
    print("  Mode: Full DFL (bypass_mode=0)")
    print("  Rounds: 5")
    print("  Workers: 30") 
    print("  Attack: GIA (Gradient Inversion Attack)")
    print("  GIA Iterations: 2000")
    print("  Attack Rate: 0.1 (10% malicious)")
    print()
    
    print("[RUNNING] command:")
    print(f"  {' '.join(cmd)}\n")
    
    # Run the test
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        
        if result.stderr:
            print("[STDERR]")
            print(result.stderr)
        
        # Check if there were errors
        if result.returncode != 0:
            print(f"\n[ERROR] Test failed with return code {result.returncode}")
            return False
            
        # Find and verify the output JSON file
        print("\n[VERIFICATION] Checking output JSON file...")
        json_files = sorted(Path(".").glob("ablation_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not json_files:
            print("[ERROR] No ablation output file found!")
            return False
            
        latest_json = json_files[0]
        print(f"  Latest output: {latest_json}")
        
        with open(latest_json, 'r') as f:
            data = json.load(f)
        
        # Check for GIA metrics
        if 'recon_mse' in data and 'recon_psnr' in data:
            print(f"\n[SUCCESS] GIA metrics found in output!")
            print(f"  MSE values: {data['recon_mse']}")
            print(f"  PSNR values: {data['recon_psnr']}")
            
            # Validate metrics are reasonable
            if data['recon_mse'] and all(isinstance(x, (int, float)) for x in data['recon_mse']):
                print(f"\n[VALIDATION] MSE metrics are valid numbers")
                avg_mse = sum(data['recon_mse']) / len(data['recon_mse'])
                print(f"  Average MSE: {avg_mse:.6f}")
            
            if data['recon_psnr'] and all(isinstance(x, (int, float)) for x in data['recon_psnr']):
                print(f"[VALIDATION] PSNR metrics are valid numbers")
                avg_psnr = sum(data['recon_psnr']) / len(data['recon_psnr'])
                print(f"  Average PSNR: {avg_psnr:.2f} dB")
                
            return True
        else:
            print(f"\n[WARNING] GIA metrics NOT found in output")
            print(f"  Available metrics: {list(data.keys())}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gia_metrics()
    sys.exit(0 if success else 1)
