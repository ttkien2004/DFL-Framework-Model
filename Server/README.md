# Cách thiết lập và chạy Server

PORT=15578 SERVER_IP=103.73.232.191

Lệnh chạy kết nối Server

```bash
ssh -p SERVER_PORT root@SERVER_IP -i path/to/VNGKeys
```

Xem Dashboard từ máy tính cá nhân

```bash
ssh -p SERVER_PORT -i path/to/VNGKeys -L 5000:localhost:YOUR_PORT root@SERVER_IP
```

Tạo tmux session

```bash
tmux new-session -s <ten_session>
```

Vào lại tmux session

```bash
tmux attach -t <ten_session>
```

Hủy tmux session

```bash
tmux detach -as
```

Cách chạy remote thư mục trên server

- Đầu tiên vào `extension` trên vscode và tìm và cài đặt 'REMOTE - SSH' của
  Microsoft.

- Tiếp theo, nhấn tổ hợp `CTRL + Shift + P`, tìm Remote SSH: Open Configuration
  file

- Nó sẽ hiển thị file config với nội dung có cấu trúc dưới đây, đổi các trường
  dưới đây:

```bash
Host <YOUR_HOST_NAME>
    HostName <SERVER_IP>
    User root
    Port <SERVER_PORT>
    # 1. Dùng đường dẫn tuyệt đối (Thay Admin bằng tên user máy bạn)
    IdentityFile /path/to/VNGKeys

    # 2. QUAN TRỌNG: Lệnh này ép VS Code chỉ được dùng key trên, cấm thử key lung tung
    IdentitiesOnly yes

    # 3. Tăng thời gian chờ (để tránh lỗi timeout nếu mạng lag)
    ConnectTimeout 60
```

- Sau khi lưu file `config`, ở góc trái phía dưới VSCode có biểu tượng `><`,
  nhấn vào, chọn Connect to Host, chọn Tên Host bạn vừa mới thay đổi.

- Khi đó, VSCode sẽ tự động mở trình duyệt mới, lúc này VS sẽ yêu cầu chọn Ngôn
  ngữ, chọn `Linux`, sau đó nhập passphrase của server. Chờ khoảng 5 phút để kết
  nối thành công.

Cách chạy dự án trên server

- `cd` vào thư mục có tên của bạn

- Kiểm tra thông qua lệnh `ls -a` đã có file setup.sh, nếu chưa thì tạo và copy
  đoạn code sh vào file như sau:

```bash
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
```

- Ngược lại thì chạy file: `./setup.sh`

- Sau khi chạy xong và có hiển thị dòng Cài đặt hoàn tất, thử chạy:

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0)}')"
```

- Nếu có dòng hiển thị: `GPU Available`, thì tiếp tục chạy `python main.py` để
  khởi động hệ thống.
