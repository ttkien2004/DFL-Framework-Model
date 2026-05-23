## MNIST + GIA Channel/Shape Mismatch Fixes

### Problems Identified

**Problem 1: Channel Mismatch**
```
Error: Given groups=1, weight of size [32, 1, 3, 3], expected input[32, 3, 32, 32] 
to have 1 channels, but got 3 channels instead
```

**Root Cause:** 
- MNIST data is transformed to 3 channels (via `transforms.Grayscale(num_output_channels=3)`)
- But `SimpleCNN` was hardcoded to expect 1 channel input
- Other models (VGG9, ResNet20) were also hardcoded to 3 channels

**Problem 2: Shape Mismatch**
```
Error: shape '[-1, 1600]' is invalid for input of size 73728
```

**Root Cause:**
- Conv layers without padding: kernel=3 reduces spatial dimensions
- Image size: 32x32 → Conv1 (no pad) → 30x30 → Pool → 15x15 → Conv2 (no pad) → 13x13 → Pool → 6x6
- But actual computation gave different size (larger spatial dims)
- FC layer expected 64*5*5=1600 but got 73728 (likely 64*24*24)

### Solutions Implemented

#### 1. **Add `input_channels` parameter to all models** (`app/models/cnn.py`)

Updated function signature:
```python
def get_model(model_name, num_classes=10, input_channels=3, input_dim=None):
```

Now all model classes accept `input_channels` parameter:
- `SimpleCNN(num_classes=10, input_channels=3)`
- `VGG9(num_classes=10, input_channels=3)`
- `ResNet20(num_classes=10, input_channels=3)`

#### 2. **Fix SimpleCNN architecture** 

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(SimpleCNN, self).__init__()
        # Add padding=1 to preserve spatial dimensions
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Use AdaptiveAvgPool2d to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 128) 
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)  # [batch, 64, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 64]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

**Key changes:**
- `padding=1` on Conv layers → preserves spatial dimensions
- `AdaptiveAvgPool2d((1, 1))` → always produces 1x1 spatial size regardless of input
- FC layers adjusted to match: 64 → 128 → num_classes

#### 3. **Fix VGG9 architecture**

```python
self.features = nn.Sequential(
    # ... Conv layers with padding=1 ...
    nn.AdaptiveAvgPool2d((1, 1))  # Always produces [batch, 512, 1, 1]
)

self.classifier = nn.Sequential(
    nn.Linear(512, 512),  # No longer hardcoded 512*2*2
    # ... rest of classifier ...
)
```

#### 4. **Update all model instantiation calls**

Files updated:
- `app/core/worker.py` (line 52): Pass `input_channels` to `get_model()`
- `app/core/engine.py` (lines 438, 1656, 1736): Pass `input_channels` from workers
- `app/core/cluster_head.py` (lines 24, 60): Pass `input_channels` to `get_model()`

Example:
```python
# Before
self.model = get_model(model_name, num_classes=self.num_classes)

# After  
self.model = get_model(model_name, num_classes=self.num_classes, input_channels=self.input_channels)
```

#### 5. **Data loader returns correct channel info**

`app/utils/data_loader.py` already returns correct channels:
- MNIST: 3 channels (transformed via `Grayscale(num_output_channels=3)`)
- CIFAR10: 3 channels
- Health: 36 features
- etc.

### Testing

Use test script to verify:
```bash
python test_mnist_shape_fix.py
```

Expected output:
```
✓ Channel match verified: 3
✓ Input shape: [batch, 3, 32, 32]
✓ Forward pass successful!
✅ All tests passed!
```

### Compatibility

- ✅ MNIST (3 channels, 32x32)
- ✅ CIFAR10 (3 channels, 32x32)
- ✅ Health (36 features, treated as 2D)
- ✅ Variable input sizes (due to AdaptiveAvgPool2d)
- ✅ GIA attack on any dataset
- ✅ Backwards compatible (input_channels defaults to 3)

### Files Modified

1. `app/models/cnn.py` - Updated all models with input_channels support and AdaptiveAvgPool2d
2. `app/core/worker.py` - Pass input_channels to get_model()
3. `app/core/engine.py` - Pass input_channels from workers to get_model()
4. `app/core/cluster_head.py` - Pass input_channels to get_model()
5. `app/utils/data_loader.py` - No changes needed (already correct)

### Running GIA on MNIST

Now you can run:
```python
python test_mnist_gia_fix.py
```

Or through health ablation:
```bash
python run_health_ablation.py --bypass_mode 0 --rounds 2 --workers 5 --attack_type GIA
```
