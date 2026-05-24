## GIA Metrics (MSE/PSNR) Integration - Implementation Summary

### Overview
Successfully added support for collecting **MSE (Mean Squared Error)** and **PSNR (Peak Signal-to-Noise Ratio)** metrics from GIA (Gradient Inversion Attack) scenarios in the health ablation study framework.

### What Was Changed

#### 1. **run_health_ablation.py** - Main Changes

**Function Signature Updated:**
```python
def run_health_ablation_simple(bypass_mode, total_rounds, num_workers, 
                               attack_type="NONE", gia_iterations=2000, 
                               gia_lr=1.0, attack_rate=0.1):
```

**New Capabilities:**
- ✅ Support for attack_type parameter (NONE, LABEL_FLIPPING, BACKDOOR, MODEL_POISONING, GIA, GRADIENT_INVERSION, MIA)
- ✅ GIA-specific parameters (gia_iterations, gia_lr)
- ✅ Attack rate configuration (proportion of malicious workers)

**Config Added:**
```python
config = {
    ...existing fields...
    "attack_type": attack_type,
    "attack_rate": attack_rate,
    "gia_iterations": gia_iterations,
    "gia_lr": gia_lr
}
```

**Metrics Collection Added (Lines 96-158):**
- Extract `recon_mse` from engine results: `result.get('recon_mse', None)`
- Extract `recon_psnr` from engine results: `result.get('recon_psnr', None)`
- Conditional appending to history (only added if values exist):
  ```python
  if recon_mse is not None:
      history['recon_mse'].append(recon_mse)
  if recon_psnr is not None:
      history['recon_psnr'].append(recon_psnr)
  ```

**Console Output Enhanced:**
```
[OK] Accuracy: 0.8915
     Loss: 0.3421
     TER: 0.3333
     Traffic: 12.45 MB
     Time: 42.15s
     [GIA] MSE: 0.000234, PSNR: 36.31 dB  ← NEW
```

**Command-Line Arguments Added:**
```bash
--attack_type {NONE,LABEL_FLIPPING,BACKDOOR,MODEL_POISONING,GIA,GRADIENT_INVERSION,MIA}
--gia_iterations <int>      # Default: 2000
--gia_lr <float>            # Default: 1.0
--attack_rate <float>       # Default: 0.1
```

**Examples Added to Help:**
```bash
# GIA attack with recommended settings (5 rounds, 2000 iterations, 0.1 attack rate)
python run_health_ablation.py --bypass_mode 0 --rounds 5 --workers 30 \
    --attack_type GIA --gia_iterations 2000 --attack_rate 0.1
```

#### 2. **Metrics Flow Architecture**

The complete flow for GIA metrics:
```
SimulationEngine.run_round()
  ↓
_calculate_advanced_metrics()
  ↓ (attack_type == "GIA")
  ↓
worker.evaluate_privacy()
  ↓
attack_strategy.evaluate()
  ↓ (GradientInversionStrategy)
  ↓
Returns: {"recon_mse": float, "recon_psnr": float}
  ↓
engine.run_round() returns combined metrics
  ↓
run_health_ablation_simple() extracts recon_mse, recon_psnr
  ↓
Appended to history['recon_mse'] and history['recon_psnr']
  ↓
Saved to ablation_*.json output
```

### Usage

#### Standard Usage (Recommended by User)
```bash
python run_health_ablation.py \
    --bypass_mode 0 \
    --rounds 5 \
    --workers 30 \
    --attack_type GIA \
    --gia_iterations 2000 \
    --attack_rate 0.1
```

**Parameters:**
- `--bypass_mode 0`: Full DFL with all features enabled
- `--rounds 5`: 5 training rounds
- `--workers 30`: 30 total workers
- `--attack_type GIA`: Enable Gradient Inversion Attack
- `--gia_iterations 2000`: Number of iterations for gradient inversion
- `--attack_rate 0.1`: 10% of workers are malicious (3 out of 30)

#### JSON Output Format
The output `ablation_<timestamp>.json` now includes:
```json
{
    "config": {...},
    "metrics": {
        "avg_acc": [0.892, 0.895, ...],
        "avg_loss": [0.342, 0.334, ...],
        "max_ter": [0.333, 0.333, ...],
        "comm_traffic_mb": [12.45, 12.43, ...],
        "execution_time": [42.15, 43.21, ...],
        "recon_mse": [0.000234, 0.000241, ...],      ← NEW
        "recon_psnr": [36.31, 36.27, ...]            ← NEW
    },
    "metadata": {...}
}
```

### Expected Metrics Ranges

**MSE (Mean Squared Error):**
- Lower is better (less reconstruction error)
- Typical range for normalized images: 0.0001 - 0.01
- Represents how closely the attack reconstructed the original data

**PSNR (Peak Signal-to-Noise Ratio):**
- Higher is better (better reconstruction quality)
- Typical range: 20-50 dB
- Formula: 10 * log10(1.0 / MSE) for [0,1] normalized data
- Values > 35 dB indicate good reconstruction

### Testing

A test script is provided: `test_gia_metrics.py`
```bash
python test_gia_metrics.py
```

This will:
1. Run ablation study with GIA attack (5 rounds, recommended settings)
2. Verify metrics appear in JSON output
3. Validate metric values are reasonable
4. Print results summary

### Backwards Compatibility

✅ **Fully backwards compatible:**
- `attack_type="NONE"` by default (no attacks)
- Non-GIA attacks: recon_mse and recon_psnr not added to history (no errors)
- Existing scripts continue to work unchanged

### Notes

1. **Performance Impact:** GIA attacks with 2000 iterations may take additional time per round (typically 30-60s additional per round with 30 workers)

2. **Attack Rate:** If `attack_rate=0.1` with 30 workers, exactly 3 workers will be malicious and perform GIA attacks

3. **Metric Collection:** MSE/PSNR are only collected when:
   - `attack_type` is "GIA" or "GRADIENT_INVERSION"
   - Workers execute the gradient inversion attack successfully
   - Reconstruction comparison succeeds

4. **Integration with Other Attacks:** The system also supports other attack types (LABEL_FLIPPING, BACKDOOR, etc.) with future metric collection if implemented

### Future Enhancements

- Add MSE/PSNR visualization in analysis scripts
- Support for multiple privacy metrics comparison
- Optimization tracking across GIA iterations
- Convergence analysis for different attack rates
