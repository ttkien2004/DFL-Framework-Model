#!/bin/bash

# Kiểm tra xem người dùng có truyền Port vào không
if [ -z "$1" ]; then
    echo "Lỗi: Bạn chưa nhập Port!"
    echo "Cách dùng: ./run_node.sh <PORT>"
    exit 1
fi

PORT=$1

# Thiết lập giới hạn tài nguyên an toàn
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export CUDA_VISIBLE_DEVICES="0:0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "🚀 Đang khởi động COBRA-FL trên Port $PORT với 6 Threads CPU..."

# Chạy hệ thống
python main.py --port $PORT