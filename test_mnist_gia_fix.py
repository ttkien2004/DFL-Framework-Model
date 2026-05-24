#!/usr/bin/env python3
"""
Quick test to verify channel mismatch fix for GIA + MNIST
"""
from app.core.engine import SimulationEngine

def test_gia_mnist():
    """Test GIA attack on MNIST dataset"""
    config = {
        "bypass_mode": 0,
        "num_workers": 5,
        "dataset": "mnist",
        "model": "simple_cnn",
        "batch_size": 32,
        "learning_rate": 0.01,
        "non_iid_alpha": 0.5,
        "system_mode": "PROPOSED",
        "num_classes": 10,
        "reset": True,
        # Attack config
        "attack_type": "GIA",
        "attack_rate": 0.2,  # 1 out of 5 workers
        "gia_iterations": 100,  # Quick test with 100 iterations
        "gia_lr": 1.0
    }
    
    print("[TEST] Initializing engine with MNIST + GIA...")
    try:
        engine = SimulationEngine()
        engine.initialize_system(config)
        print("[OK] Engine initialized successfully!")
        
        # Check worker configuration
        print(f"\n[INFO] Created {len(engine.workers)} workers")
        w = engine.workers[0]
        print(f"  Worker 0 info:")
        print(f"    Dataset: {w.dataset_name}")
        print(f"    Input channels: {w.input_channels}")
        print(f"    Model conv1 input: {w.model.conv1.in_channels}")
        print(f"    Data loader: {type(w.data_loader)}")
        
        # Run 1 round
        print(f"\n[TEST] Running round 0...")
        result = engine.run_round(0, config)
        print("[OK] Round completed successfully!")
        print(f"  Accuracy: {result.get('avg_acc', 0):.4f}")
        print(f"  Loss: {result.get('avg_loss', 0):.4f}")
        if 'recon_mse' in result:
            print(f"  GIA MSE: {result.get('recon_mse', 0):.6f}")
            print(f"  GIA PSNR: {result.get('recon_psnr', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = test_gia_mnist()
    sys.exit(0 if success else 1)
