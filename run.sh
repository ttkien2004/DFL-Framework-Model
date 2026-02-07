#!/bin/bash

echo "=== BẮT ĐẦU CÀI ĐẶT MÔI TRƯỜNG CHO NVIDIA H100 ==="

# 1. Nâng cấp pip để tránh lỗi
echo "-> Đang nâng cấp pip..."
python -m pip install --upgrade pip

# 2. Gỡ bỏ bản PyTorch cũ (nếu có) để tránh xung đột
echo "-> Đang gỡ bỏ PyTorch cũ/lỗi..."
pip uninstall torch torchvision torchaudio -y

# 3. Cài đặt PyTorch bản mới nhất tương thích CUDA 12.1 (Dành cho H100)
echo "-> Đang cài đặt PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Cài đặt các thư viện còn lại từ requirements.txt
echo "-> Đang cài đặt các thư viện khác..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Cảnh báo: Không tìm thấy file requirements.txt!"
fi

echo "=== CÀI ĐẶT HOÀN TẤT! HÃY CHẠY python main.py ==="