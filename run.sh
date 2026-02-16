#!/bin/bash
set -e

echo "--- 1. XỬ LÝ LỖI PROTOBUF & LÀM SẠCH HỆ THỐNG ---"
# Gỡ cài đặt để xóa sạch các bản cũ gây xung đột
pip uninstall -y protobuf numpy pandas torch torchvision torchaudio
# Xóa vật lý thư mục để tránh lỗi "binary incompatibility" (96 vs 88)
rm -rf /usr/local/lib/python3.10/dist-packages/google/protobuf
rm -rf /usr/local/lib/python3.10/dist-packages/numpy*
rm -rf /usr/local/lib/python3.10/dist-packages/pandas*

echo "--- 2. CÀI ĐẶT PHIÊN BẢN CHO MÔI TRƯỜNG ---"
# Cài protobuf bản 3.20.x là bản ổn định nhất để né lỗi "Descriptors cannot be created"
pip install "protobuf==3.20.3"

# Cài đặt Numpy và Pandas theo đúng requirements của bạn
pip install "numpy==1.24.3" "pandas==2.0.3" "requests==2.31.0" "flask==2.3.2" \
            "tabulate==0.9.0" "matplotlib==3.7.1" "scikit-learn" "cryptography"

echo "--- 3. CÀI TORCH 2.1.0 CHO H100 (CUDA 12.1) ---"
# Như đã biết, 2.0.1 không có trên cu121, nên ta dùng 2.1.0 là bản tối thiểu cho H100
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

echo "--- 4. KIỂM TRA LỖI TRƯỚC KHI CHẠY MAIN.PY ---"
python -c "
import google.protobuf
import torch
import numpy
import pandas
print(f'Protobuf Version: {google.protobuf.__version__}')
print(f'Torch CUDA: {torch.cuda.is_available()}')
print(f'Numpy: {numpy.__version__}')
print('=> MÔI TRƯỜNG ĐÃ SẴN SÀNG, THỬ CHẠY LẠI MAIN.PY!')
"