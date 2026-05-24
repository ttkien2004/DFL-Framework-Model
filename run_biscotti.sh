#!/bin/bash

# Thiết lập giới hạn tài nguyên CPU (tránh nghẽn cổ chai khi chạy nhiều node)
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

# Thiết lập cấu hình GPU (Tối ưu hóa quản lý bộ nhớ VRAM cho PyTorch)
export CUDA_VISIBLE_DEVICES="0" # Chỉ định dùng GPU 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================================="
echo "🚀 Đang khởi động hệ thống thực nghiệm với 6 Threads CPU..."
echo "⚙️ Cấu hình lệnh: python run_experiment.py $@"
echo "=========================================================="

# Chạy hệ thống và truyền toàn bộ tham số lệnh (arguments) vào file Python
python run_experiment.py "$@"