#!/bin/bash

echo "=== BẮT ĐẦU CÀI ĐẶT MÔI TRƯỜNG CHO NVIDIA H100 ==="

echo "-> Đang nâng cấp pip..."
python -m pip install --upgrade pip

echo "-> Đang gỡ bỏ PyTorch cũ/lỗi..."
pip uninstall torch torchvision torchaudio -y

echo "-> Đang cài đặt PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "-> Đang cài đặt các thư viện khác..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Cảnh báo: Không tìm thấy file requirements.txt!"
fi

echo "=== CÀI ĐẶT HOÀN TẤT! HÃY CHẠY python main.py ==="

