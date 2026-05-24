#!/usr/bin/env python3
"""
Test channel and shape fixes for MNIST
"""
import torch
from app.utils.data_loader import get_dataloader
from app.models.cnn import get_model

def test_mnist_shape():
    """Test MNIST data loading and model inference"""
    print("[TEST] Testing MNIST data loading and model inference...")
    
    try:
        # Load MNIST data
        print("\n1. Loading MNIST dataset...")
        loader, num_classes, input_channels = get_dataloader('mnist', 0, 5, 32)
        print(f"   ✓ Dataset loaded: classes={num_classes}, channels={input_channels}")
        
        # Create model
        print("\n2. Creating SimpleCNN model...")
        model = get_model('simple_cnn', num_classes=num_classes, input_channels=input_channels)
        print(f"   ✓ Model created: input_channels={model.conv1.in_channels}")
        
        # Check channel match
        assert model.conv1.in_channels == input_channels, f"Channel mismatch! Model expects {model.conv1.in_channels}, data has {input_channels}"
        print(f"   ✓ Channel match verified: {input_channels}")
        
        # Get a batch and test forward pass
        print("\n3. Testing forward pass with actual data...")
        model.eval()
        for X, y in loader:
            print(f"   Input shape: {X.shape} (expected: [batch, {input_channels}, 32, 32])")
            assert X.shape[1] == input_channels, f"Input shape mismatch! Got {X.shape[1]} channels"
            
            with torch.no_grad():
                output = model(X)
            print(f"   ✓ Output shape: {output.shape} (expected: [batch, {num_classes}])")
            assert output.shape[1] == num_classes, f"Output size mismatch! Got {output.shape[1]}, expected {num_classes}"
            
            print(f"   ✓ Forward pass successful!")
            break
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = test_mnist_shape()
    sys.exit(0 if success else 1)
