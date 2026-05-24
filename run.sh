#!/bin/bash

# Dừng script ngay nếu có lệnh bị lỗi
set -e

echo "========================================================"
echo "   CÀI ĐẶT MÔI TRƯỜNG CHO NVIDIA H100 (FINAL FIX)"
echo "   (Hỗ trợ: PyTorch 2.x, CUDA 12.1, Numpy < 2.0)"
echo "========================================================"

# 1. Nâng cấp pip
echo "[1/5] Nâng cấp pip..."
python -m pip install --upgrade pip

# 2. DỌN DẸP SẠCH SẼ (Deep Clean)
# Bước này cực quan trọng để sửa lỗi "undefined symbol: ncclCommRegister"
echo "[2/5] Gỡ bỏ các thư viện xung đột cũ..."

# Gỡ các gói chính
pip uninstall torch torchvision torchaudio -y || true
pip uninstall numpy pandas scikit-learn pyarrow -y || true

# Gỡ sạch các gói NVIDIA (Thủ phạm gây lỗi NCCL cũ)
pip freeze | grep "nvidia-" | xargs pip uninstall -y || true

# Xóa cache và file rác
pip cache purge
rm -rf /usr/local/lib/python3.10/dist-packages/torch
rm -rf /usr/local/lib/python3.10/dist-packages/nvidia

# 3. CÀI ĐẶT CÁC THƯ VIỆN NỀN TẢNG (TƯƠNG THÍCH RAPIDS/CUDF)
# Phải cài Numpy < 2.0 và Pandas < 2.2 để tránh lỗi "binary incompatibility"
echo "[3/5] Cài đặt Numpy/Pandas phiên bản tương thích..."
pip install "numpy<2.0" "pandas<2.2" "pyarrow<16.0" "scikit-learn" "tabulate" "matplotlib" "requests" "flask==2.3.2" "cryptography"

# 4. CÀI ĐẶT PYTORCH CHO H100 (CUDA 12.1)
echo "[4/5] Cài đặt PyTorch (CUDA 12.1) cho H100..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

# 5. CÀI ĐẶT CÁC THƯ VIỆN CÒN LẠI TỪ REQUIREMENTS (NẾU CÓ)
# Script sẽ tự tạo file tạm, loại bỏ torch/numpy/pandas để tránh cài đè bản sai
echo "[5/5] Cài đặt bổ sung từ requirements.txt (Safe Mode)..."
if [ -f "requirements.txt" ]; then
    # Tạo file tạm, loại bỏ các dòng chứa torch, numpy, pandas, pyarrow, flask
    grep -vE "torch|numpy|pandas|pyarrow|flask|scikit-learn|torchvision" requirements.txt > temp_requirements.txt
    
    # Cài đặt từ file tạm
    if [ -s temp_requirements.txt ]; then
        pip install -r temp_requirements.txt
    else
        echo " -> Không có thư viện bổ sung nào cần cài thêm."
    fi
    
    # Xóa file tạm
    rm temp_requirements.txt
else
    echo " -> Không tìm thấy requirements.txt, bỏ qua."
fi

echo "========================================================"
echo "   CÀI ĐẶT HOÀN TẤT - KIỂM TRA HỆ THỐNG"
echo "========================================================"

python -c "
import torch
import numpy
import pandas
print(f'PyTorch Version : {torch.__version__}')
print(f'CUDA Available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device Name     : {torch.cuda.get_device_name(0)}')
print(f'Numpy Version   : {numpy.__version__}')
print(f'Pandas Version  : {pandas.__version__}')

if torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name(0) and torch.version.cuda.startswith('12'):
    print('\n>>> TRẠNG THÁI: OK TUYỆT ĐỐI (Sẵn sàng chạy main.py) <<<')
else:
    print('\n>>> CẢNH BÁO: Kiểm tra lại thông số ở trên <<<')
"