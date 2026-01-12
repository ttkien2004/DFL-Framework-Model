# Sử dụng Python Slim để nhẹ máy
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn
COPY . .

# Mở port 5000 cho Flask
EXPOSE 5000

# Lệnh chạy mặc định
CMD ["python", "main.py"]