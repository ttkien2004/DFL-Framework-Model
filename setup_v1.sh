#!/bin/bash

echo "=== BẮT ĐẦU CÀI ĐẶT MÔI TRƯỜNG CHO NVIDIA H100 ==="

echo "-> Tạo virtual environment..."
python -m venv venv

echo "-> Kích hoạt venv..."
source venv/bin/activate

echo "-> Nâng cấp pip..."
pip install --upgrade pip

echo "-> Cài PyTorch CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "-> Cài requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Cảnh báo: Không tìm thấy file requirements.txt!"
fi

echo "=== HOÀN TẤT! ==="
echo "Chạy: source venv/bin/activate && python main.py"
