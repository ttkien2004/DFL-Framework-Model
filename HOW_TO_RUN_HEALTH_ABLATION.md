# 🎯 Hướng Dẫn Nhanh - Ablation Study với Health Dataset

## ✅ Mọi Thứ Đã Sẵn Sàng!

Dataset đã được tạo: **5.000 bản ghi** với 28 đặc trưng sức khỏe cá nhân

---

## 🚀 Lệnh Chính Của Bạn (Khuyến Nghị)

```bash
python run_health_ablation.py --bypass_mode 15 --rounds 10 --workers 20
```

### Lệnh này thực hiện:

- Chạy ablation study với **Traditional DFL** (tất cả components bị vô hiệu hóa)
- 10 vòng huấn luyện
- 20 workers
- Dataset: Health Personal
- Mô hình: HealthMLP

---

## 📊 So Sánh Nhiều Chế Độ

Chạy 6 lệnh sau để so sánh đầy đủ:

```bash
# 1. Full Features (CoCo + LDP + BALANCE + Blockchain)
python run_health_ablation.py --bypass_mode 0 --rounds 5 --workers 20

# 2. No Clustering
python run_health_ablation.py --bypass_mode 1 --rounds 5 --workers 20

# 3. No Privacy (không có LDP/SSS)
python run_health_ablation.py --bypass_mode 2 --rounds 5 --workers 20

# 4. No Byzantine (FedAvg)
python run_health_ablation.py --bypass_mode 4 --rounds 5 --workers 20

# 5. No Blockchain (RAM storage)
python run_health_ablation.py --bypass_mode 8 --rounds 5 --workers 20

# 6. Traditional DFL (lệnh chính của bạn)
python run_health_ablation.py --bypass_mode 15 --rounds 5 --workers 20
```

---

## ⚡ Kiểm Tra Nhanh (2 phút)

```bash
python run_health_ablation.py --bypass_mode 15 --rounds 2 --workers 5
```

---

## 📋 Liệt Kê Tất Cả Các Chế Độ

```bash
python run_health_ablation.py --list_modes
```

Kết quả:

```
Available Bypass Modes for Health Ablation Study:
────────────────────────────────────────────────────────────────────────────────
   0: Full_Features - All features enabled (CoCo + LDP + BALANCE + Blockchain)
   1: No_Clustering - Single cluster (traditional DFL)
   2: No_Privacy - No LDP/SSS (clean gradients)
   4: No_Byzantine - FedAvg instead of BALANCE
   8: No_Blockchain - RAM storage instead of blockchain
  15: Traditional_DFL - All features disabled (baseline)
────────────────────────────────────────────────────────────────────────────────
```

---

## 📁 Kết Quả Đầu Ra

Mỗi lần chạy sẽ tạo một file JSON trong thư mục:
`histories/ablation_[timestamp]_[scenario_name].json`

Ví dụ:

```
histories/ablation_20260515_111533_Traditional_DFL.json
```

### Nội dung file JSON:

```json
{
  "ablation_scenario": "Traditional_DFL",
  "bypass_mode": 15,
  "total_rounds": 10,
  "num_workers": 20,
  "dataset": "health",
  "model": "health_mlp",
  "metrics": {
    "avg_acc": [0.50, 0.55, 0.60, ...],
    "avg_loss": [0.70, 0.65, 0.60, ...],
    "max_ter": [0.50, 0.45, 0.40, ...],
    "comm_traffic_mb": [25.0, 25.0, 25.0, ...],
    "execution_time": [2.1, 2.1, 2.1, ...]
  },
  "bypass_report": {
    "mode": "Traditional_DFL",
    "clustering_enabled": false,
    "privacy_enabled": false,
    "byzantine_enabled": false,
    "blockchain_enabled": false
  }
}
```

---

## 🔍 Phân Tích Kết Quả

### Xem tất cả file đã tạo:

```bash
ls -la histories/ablation_*.json
```

### Tải và phân tích bằng Python:

```python
import json

with open("histories/ablation_20260515_111533_Traditional_DFL.json") as f:
    data = json.load(f)

# Xem kịch bản
print(f"Scenario: {data['ablation_scenario']}")

# Xem accuracy cuối cùng
final_acc = data['metrics']['avg_acc'][-1]
print(f"Final Accuracy: {final_acc:.4f}")

# Xem accuracy trung bình
avg_acc = sum(data['metrics']['avg_acc']) / len(data['metrics']['avg_acc'])
print(f"Average Accuracy: {avg_acc:.4f}")
```

---

## 📈 Workflow Được Khuyến Nghị

### Bước 1: Kiểm Tra Nhanh (Xác nhận setup)

```bash
python run_health_ablation.py --bypass_mode 15 --rounds 2 --workers 5
```

### Bước 2: Chạy Chế Độ Chính

```bash
python run_health_ablation.py --bypass_mode 15 --rounds 10 --workers 20
```

### Bước 3: So Sánh với Full Features

```bash
python run_health_ablation.py --bypass_mode 0 --rounds 10 --workers 20
```

### Bước 4: Phân Tích So Sánh

```python
# compare.py
import json

results = {}
for mode in [0, 1, 2, 4, 8, 15]:
    # Tìm file tương ứng trong histories/
    with open(f"histories/ablation_*_{mode}_*.json") as f:
        data = json.load(f)
        scenario = data['ablation_scenario']
        final_acc = data['metrics']['avg_acc'][-1]
        results[scenario] = final_acc

for scenario, acc in sorted(results.items()):
    print(f"{scenario:20s}: {acc:.4f}")
```

---

## 💡 Giải Thích Kết Quả

| Chỉ số              | Ý nghĩa                                           |
| ------------------- | ------------------------------------------------- |
| **avg_acc**         | Accuracy trung bình (0.0-1.0) - càng cao càng tốt |
| **avg_loss**        | Loss trung bình - càng thấp càng tốt              |
| **max_ter**         | Tỷ lệ lỗi tối đa - càng thấp càng tốt             |
| **comm_traffic_mb** | Lưu lượng giao tiếp - càng thấp càng tốt          |
| **execution_time**  | Thời gian thực thi - càng thấp càng tốt           |

### So sánh mong đợi:

- **Mode 0 vs 15**: Mode 0 có accuracy tốt hơn (có LDP + BALANCE)
- **Mode 1 vs 15**: Mode 1 ít lưu lượng hơn (1 cluster vs K clusters)
- **Mode 2 vs 15**: Mode 2 có accuracy tốt hơn (không có privacy noise)
- **Mode 4 vs 15**: Mode 4 nhanh hơn (FedAvg vs BALANCE)
- **Mode 8 vs 15**: Mode 8 nhanh hơn (RAM vs Blockchain)

---

## 🐛 Khắc Phục Sự Cố

### Q: "Health dataset not found"

```bash
python generate_health_data.py
```

### Q: "Chạy quá chậm"

Giảm số rounds/workers:

```bash
python run_health_ablation.py --bypass_mode 15 --rounds 3 --workers 10
```

### Q: "CUDA out of memory"

Giảm batch_size (chỉnh trong `run_health_ablation.py` dòng 30)

### Q: "Không tạo được file JSON"

Kiểm tra thư mục `histories/` đã tồn tại:

```bash
mkdir -p histories
```

---

## 📚 Cấu Trúc Thư Mục

```
d:\DATN\DFL-Framework-Model\
├── run_health_ablation.py          ← Script chính (dùng file này!)
├── generate_health_data.py         ← Tạo dataset
├── app/
│   ├── core/
│   │   ├── bypass_ablation.py     ← Logic bypass
│   │   ├── engine.py              ← Engine chính
│   │   └── worker.py              ← Workers
│   ├── models/
│   │   └── cnn.py                 ← Mô hình HealthMLP
│   └── utils/
│       └── data_loader.py         ← Dataset loader
├── data/
│   └── personal_health_data.csv   ← Dataset (được tạo bởi generate_health_data.py)
└── histories/
    ├── ablation_20260515_111533_Traditional_DFL.json
    ├── ablation_20260515_111604_Full_Features.json
    └── ...
```

---

## ✅ Checklist

- [x] Dataset đã tạo: `data/personal_health_data.csv` (5.000 bản ghi)
- [ ] Kiểm tra nhanh:
      `python run_health_ablation.py --bypass_mode 15 --rounds 2 --workers 5`
- [ ] Chế độ chính:
      `python run_health_ablation.py --bypass_mode 15 --rounds 10 --workers 20`
- [ ] So sánh: Chạy 6 chế độ khác nhau
- [ ] Phân tích: Kiểm tra kết quả trong `histories/`

---

## 🎓 Bypass Mode 15 Có Nghĩa Là Gì?

**Traditional DFL** = Tất cả tính năng bị vô hiệu hóa:

- ❌ Không có CoCo Clustering (mọi thứ trong 1 cluster)
- ❌ Không có LDP Privacy (gradient sạch)
- ❌ Không có BALANCE Byzantine (dùng FedAvg đơn giản)
- ❌ Không có Blockchain Consensus (RAM storage)

= Federated Learning truyền thống, không có cải tiến

---

**Tạo ngày:** 2025-05-15 **Trạng thái:** ✅ Sẵn sàng sử dụng **Dataset:** Health
Personal (5.000 bản ghi, 28 features) **Mô hình:** HealthMLP **Bypass Modes:** 6
(0, 1, 2, 4, 8, 15)
