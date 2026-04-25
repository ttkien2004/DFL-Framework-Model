# Các kịch bản thực nghiệm

## 1. Danh sách các lệnh chạy thực nghiệm (Scripts)

A. Thực nghiệm trên tập dữ liệu MNIST MNIST - Gaussian Noise (50 nodes):

```Bash
python run_biscotti.py --dataset mnist --attack-type GAUSS --num-nodes 50 --malicious-ratio 0.3 --max-iterations 100
```

MNIST - Backdoor Attack (50 nodes, Target: 5):

```Bash
python run_biscotti.py --dataset mnist --attack-type BACKDOOR --num-nodes 50 --malicious-ratio 0.3 --max-iterations 100 --tgt-class 5
```

MNIST - Label Flipping (30 nodes, 3 -> 5):

```Bash
python run_biscotti.py --dataset mnist --attack-type LABEL_FLIPPING --num-nodes 30 --malicious-ratio 0.3 --max-iterations 100 --src-class 3 --tgt-class 5

python run_biscotti.py --dataset mnist --attack-type GRADIENT_INVERSION --num-nodes 10 --malicious-ratio 0.3 --max-iterations 20 --gia-iterations 2000 --gia-lr 0.1
```

Hoặc nếu chạy bằng bash:

```bash
# 1. MNIST - Gaussian Noise
./run_biscotti.sh --dataset mnist --attack-type GAUSS --num-nodes 50 --malicious-ratio 0.3 --max-iterations 100

# 2. MNIST - Backdoor Attack
./run_biscotti.sh --dataset mnist --attack-type BACKDOOR --num-nodes 50 --malicious-ratio 0.3 --max-iterations 100 --tgt-class 5

# 3. MNIST - Label Flipping
./run_biscotti.sh --dataset mnist --attack-type LABEL_FLIPPING --num-nodes 30 --malicious-ratio 0.3 --max-iterations 100 --src-class 3 --tgt-class 5
```

B. Thực nghiệm trên tập dữ liệu Personal Health Health - Gaussian Noise (50
nodes):

```Bash
python run_experiment.py --dataset health --attack-type GAUSS --num-nodes 50 --malicious-ratio 0.3 --max-iterations 100
```

Health - Backdoor Attack (50 nodes, Target: 1):

```Bash
python run_experiment.py --dataset health --attack-type BACKDOOR --num-nodes 50 --malicious-ratio 0.3 --max-iterations 100 --tgt-class 1
```

Health - Label Flipping (30 nodes, 0 -> 1):

```Bash
python run_experiment.py --dataset health --attack-type LABEL_FLIPPING --num-nodes 30 --malicious-ratio 0.3 --max-iterations 100 --src-class 0 --tgt-class 1
```

Hoặc nếu chạy bằng bash

```bash
# 4. Health - Gaussian Noise
./run_biscotti.sh --dataset health --attack-type GAUSS --num-nodes 50 --malicious-ratio 0.3 --max-iterations 100

# 5. Health - Backdoor Attack
./run_biscotti.sh --dataset health --attack-type BACKDOOR --num-nodes 50 --malicious-ratio 0.3 --max-iterations 100 --tgt-class 1

# 6. Health - Label Flipping
./run_biscotti.sh --dataset health --attack-type LABEL_FLIPPING --num-nodes 30 --malicious-ratio 0.3 --max-iterations 100 --src-class 0 --tgt-class 1
```

## 2. Cách quản lý thực nghiệm hiệu quả với tmux

Để chạy cả 6 kịch bản này mà không phải chờ đợi lệnh trước xong mới chạy lệnh
sau (chạy song song), bạn nên tận dụng các Window trong tmux:

Khởi tạo phiên: tmux new -s biscotti_full_test

Cửa sổ 1 (MNIST - GAUSS): Chạy lệnh số 1.

Tạo Cửa sổ mới: Nhấn Ctrl + B, sau đó nhấn C. Chạy lệnh số 2.

Lặp lại: Nhấn Ctrl + B rồi C để tạo thêm các cửa sổ cho các lệnh còn lại.

Chuyển đổi giữa các cửa sổ: Nhấn Ctrl + B, sau đó nhấn số (0, 1, 2...) tương ứng
với cửa sổ muốn xem.

Thoát tạm thời (Detach): Nhấn Ctrl + B, sau đó nhấn D.

## 3. Một số lưu ý về cấu hình

Tham số --malicious-ratio 0.3: Đã được thiết lập cố định để đảm bảo 30% node tấn
công như yêu cầu của bạn.

Đường dẫn Dataset: Hãy đảm bảo file personal_health_data.csv nằm đúng trong thư
mục ./data/ như đã cấu hình trong mã nguồn.

Kết quả lưu trữ: Sau khi chạy xong, mỗi script sẽ tự động tạo một file JSON
trong thư mục histories/ với tên file chứa đầy đủ thông tin về TIMESTAMP,
ATTACK_TYPE và DATASET để bạn không bị nhầm lẫn khi thu thập số liệu viết báo
cáo.
