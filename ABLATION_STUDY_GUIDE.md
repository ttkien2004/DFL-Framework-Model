# Ablation Study Guide - Chạy từng Module Riêng

Hướng dẫn này giải thích cách chạy các scenario **ablation** (loại bỏ từng tính năng) để so sánh hiệu suất hệ thống.

## Tổng Quan

Hệ thống CoCo-Clustering có 4 tính năng chính có thể được loại bỏ:

| Bypass Mode | Tên                 | Mô tả                                           |
| ----------- | ------------------- | ----------------------------------------------- |
| 0           | **Full Features**   | Chạy đầy đủ (CoCo + LDP + BALANCE + Blockchain) |
| 1           | **No_Clustering**   | Tất cả nodes vào 1 cụm (không phân cụm)         |
| 2           | **No_Privacy**      | Không thêm nhiễu LDP/SSS, gửi gradient sạch     |
| 4           | **No_Byzantine**    | Dùng FedAvg thay vì BALANCE (không robust)      |
| 8           | **No_Blockchain**   | Lưu model vào RAM, không blockchain consensus   |
| 15          | **Traditional_DFL** | Tất cả bypass = DFL truyền thống cơ bản         |

## API Endpoints

### 1. Liệt kê các Bypass Modes
```bash
curl -X GET http://localhost:5000/list_bypass_modes
```

**Response:**
```json
{
  "available_modes": {
    "0": "Full_Features - ...",
    "1": "No_Clustering - ...",
    "2": "No_Privacy - ...",
    "4": "No_Byzantine - ...",
    "8": "No_Blockchain - ...",
    "15": "Traditional_DFL - ..."
  }
}
```

### 2. Chạy Ablation Study

**Endpoint:** `POST /run_ablation_study`

**Body:**
```json
{
  "bypass_mode": 1,
  "total_rounds": 10,
  "num_workers": 10,
  "dataset": "mnist",
  "model": "simple_cnn",
  "batch_size": 32,
  "learning_rate": 0.01,
  "non_iid_alpha": 0.5,
  "system_mode": "PROPOSED",
  "num_classes": 10,
  "reset": true
}
```

**Response:**
```json
{
  "status": "started",
  "message": "Ablation study 'No_Clustering' started in background",
  "bypass_mode": 1,
  "scenario": "No_Clustering"
}
```

### 3. Kiểm tra Trạng Thái Training
```bash
curl -X GET http://localhost:5000/training_status
```

**Response:**
```json
{
  "is_running": true,
  "current_round": 5,
  "total_rounds": 10,
  "message": "No_Clustering: Round 5/10..."
}
```

## Ví dụ Sử Dụng

### Scenario 1: So sánh với Baseline (No Clustering)
```bash
curl -X POST http://localhost:5000/run_ablation_study \
  -H "Content-Type: application/json" \
  -d '{
    "bypass_mode": 1,
    "total_rounds": 10,
    "num_workers": 20,
    "dataset": "mnist",
    "model": "simple_cnn",
    "batch_size": 32,
    "learning_rate": 0.01,
    "non_iid_alpha": 0.5,
    "system_mode": "PROPOSED",
    "num_classes": 10,
    "reset": true
  }'
```

### Scenario 2: So sánh với FedAvg (No Byzantine)
```bash
curl -X POST http://localhost:5000/run_ablation_study \
  -H "Content-Type: application/json" \
  -d '{
    "bypass_mode": 4,
    "total_rounds": 10,
    "num_workers": 20,
    "dataset": "mnist",
    "model": "simple_cnn",
    "batch_size": 32,
    "learning_rate": 0.01,
    "non_iid_alpha": 0.5,
    "system_mode": "PROPOSED",
    "num_classes": 10,
    "reset": true
  }'
```

### Scenario 3: Traditional DFL (tất cả bypass)
```bash
curl -X POST http://localhost:5000/run_ablation_study \
  -H "Content-Type: application/json" \
  -d '{
    "bypass_mode": 15,
    "total_rounds": 10,
    "num_workers": 20,
    "dataset": "mnist",
    "model": "simple_cnn",
    "batch_size": 32,
    "learning_rate": 0.01,
    "non_iid_alpha": 0.5,
    "system_mode": "PROPOSED",
    "num_classes": 10,
    "reset": true
  }'
```

## Hiểu Chi Tiết Về Các Bypass Modules

### 1. Bypass Clustering (`bypass_mode = 1`)

**Hành động:**
- Gán tất cả workers vào 1 cụm duy nhất (cluster_id = 0)
- Bỏ qua dynamic clustering dựa trên loss

**Khi nào sử dụng:**
- So sánh với DFL truyền thống không phân mảnh
- Đánh giá tác động của clustering đến hiệu suất

**Kết quả dự kiến:**
- ↓ Communication cost (1 cluster head thay vì K)
- ↑ Aggregation time (tất cả models gửi đến 1 head)
- Có thể ↓ Accuracy nếu clustering giúp personalization

### 2. Bypass Privacy (`bypass_mode = 2`)

**Hành động:**
- Không thêm nhiễu Laplace/Gaussian vào gradient
- Không chia nhỏ secret shares
- Gửi gradient "sạch" nguyên gốc

**Khi nào sử dụng:**
- Đánh giá tác động của LDP/SSS đến accuracy
- So sánh privacy-accuracy tradeoff
- Baseline không có privacy

**Kết quả dự kiến:**
- ↑ Accuracy (không mất thông tin do nhiễu)
- ↓ Privacy (gradient rò rỉ thông tin)
- ↓ Latency (không cần mã hóa/chia nhỏ)

### 3. Bypass Byzantine Robustness (`bypass_mode = 4`)

**Hành động:**
- Thay thế BALANCE (Robust Aggregation) bằng FedAvg (trung bình cộng đơn giản)
- Bỏ qua lọc outliers/malicious updates

**Khi nào sử dụng:**
- So sánh robust aggregation vs. simple averaging
- Đánh giá hiệu quả của BALANCE
- Test trên dữ liệu sạch (không tấn công)

**Kết quả dự kiến:**
- ↑ Accuracy (trên dữ liệu sạch)
- ↓ Robustness (không chịu đựng tấn công)
- ↓ Aggregation time (FedAvg nhanh hơn BALANCE)

### 4. Bypass Blockchain (`bypass_mode = 8`)

**Hành động:**
- Lưu model vào memory storage (RAM) thay vì blockchain
- Bỏ qua consensus, voting, block mining
- Trực tiếp accept tất cả aggregate models

**Khi nào sử dụng:**
- So sánh với hệ thống không blockchain
- Đánh giá overhead của blockchain consensus
- Baseline không có security blockchain

**Kết quả dự kiến:**
- ↑↑ Latency (bỏ qua costly consensus)
- ↑ Throughput (không đợi voting)
- ↓ Security (không có immutable record)

## Metrics Collected

Mỗi ablation study sẽ lưu metrics vào file JSON trong thư mục `histories/`:

```json
{
  "ablation_scenario": "No_Clustering",
  "bypass_mode": 1,
  "total_rounds": 10,
  "metrics": {
    "avg_acc": [0.50, 0.65, 0.72, ...],
    "avg_loss": [2.30, 1.80, 1.50, ...],
    "max_ter": [0.50, 0.35, 0.28, ...],
    "execution_time": [5.2, 5.1, 5.3, ...],
    "comm_traffic_mb": [12.5, 12.5, 12.5, ...],
    "latency_breakdown": [...]
  },
  "bypass_report": {
    "mode": "No_Clustering",
    "clustering_enabled": false,
    "privacy_enabled": true,
    "byzantine_enabled": true,
    "blockchain_enabled": true,
    "memory_storage_size": 0
  }
}
```

## Comparison Matrix

Để so sánh các ablation scenarios, hãy chạy series:

```bash
# Scenario 0: Full Features
curl -X POST http://localhost:5000/run_ablation_study \
  -H "Content-Type: application/json" \
  -d '{"bypass_mode": 0, "total_rounds": 10, "num_workers": 20, "dataset": "mnist", "model": "simple_cnn", "batch_size": 32, "learning_rate": 0.01, "non_iid_alpha": 0.5, "system_mode": "PROPOSED", "num_classes": 10, "reset": true}'

# Scenario 1: No Clustering
curl -X POST http://localhost:5000/run_ablation_study \
  -H "Content-Type: application/json" \
  -d '{"bypass_mode": 1, "total_rounds": 10, "num_workers": 20, "dataset": "mnist", "model": "simple_cnn", "batch_size": 32, "learning_rate": 0.01, "non_iid_alpha": 0.5, "system_mode": "PROPOSED", "num_classes": 10, "reset": true}'

# ... và cứ tiếp tục với các bypass_mode khác
```

Sau đó so sánh các file metrics trong `histories/` để thấy tác động của từng tính năng.

## Python API (Nếu chạy từ code)

```python
from app.core.bypass_ablation import BypassConfig, BypassExecutor
from app.core.engine import SimulationEngine

# Khởi tạo engine
engine = SimulationEngine()

# Cấu hình dữ liệu
config = {
    "bypass_mode": BypassConfig.BYPASS_CLUSTERING,  # Hoặc: 1
    "num_workers": 20,
    "dataset": "mnist",
    "model": "simple_cnn",
    "batch_size": 32,
    "learning_rate": 0.01,
    "non_iid_alpha": 0.5,
    "system_mode": "PROPOSED",
    "num_classes": 10,
    "reset": True
}

# Khởi tạo
engine.initialize_system(config)

# Chạy vòng
for round_id in range(10):
    result = engine.run_round(round_id, config)
    print(f"Round {round_id+1}: Acc={result['avg_acc']:.4f}")

# Kiểm tra bypass report
print(engine.bypass_executor.get_report())
```

## Kết Luận

Ablation Study giúp bạn:
- ✓ Hiểu rõ tác động của từng tính năng
- ✓ Xác định bottleneck hiệu suất
- ✓ Justify sự cần thiết của từng module
- ✓ So sánh với baseline/state-of-the-art

Sử dụng bypass modes để chạy các experiment có kiểm soát và có kết luận khoa học!
