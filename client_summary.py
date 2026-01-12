import requests
import pandas as pd
import sys

# Cấu hình
SERVER_URL = "http://localhost:5000"

def get_simulation_summary():
    print(f"🔄 Đang tải dữ liệu từ {SERVER_URL}...")
    
    try:
        # 1. Gọi API lấy toàn bộ lịch sử
        response = requests.get(f"{SERVER_URL}/get_metrics")
        
        # Kiểm tra nếu server không chạy hoặc lỗi
        if response.status_code != 200:
            print(f"❌ Lỗi Server: {response.status_code}")
            return

        data = response.json()
        
        # Kiểm tra nếu chưa có dữ liệu nào
        if not data or not data.get("rounds"):
            print("⚠️ Server chưa có dữ liệu nào. Hãy chạy thử nghiệm (run_round) trước.")
            return

        # 2. Tạo DataFrame (Bảng dữ liệu)
        df = pd.DataFrame(data)

        # 3. Định dạng lại bảng cho đẹp
        # Đổi tên cột sang tiếng Việt hoặc tên chuẩn
        df.rename(columns={
            'rounds': 'Vòng (Round)',
            'accuracy': 'Độ chính xác (%)',
            'loss': 'Độ mất mát (Loss)',
            'blockchain_height': 'Độ cao Block'
        }, inplace=True)

        # Làm tròn số liệu (Accuracy 2 số lẻ, Loss 4 số lẻ)
        # (Chỉ áp dụng nếu cột đó tồn tại và là số)
        if 'Độ chính xác (%)' in df.columns:
            df['Độ chính xác (%)'] = df['Độ chính xác (%)'].round(2)
        if 'Độ mất mát (Loss)' in df.columns:
            df['Độ mất mát (Loss)'] = df['Độ mất mát (Loss)'].round(4)

        # 4. Hiển thị bảng kết quả
        print("\n" + "="*50)
        print("📊 BẢNG TỔNG KẾT KẾT QUẢ THỬ NGHIỆM")
        print("="*50)
        
        # In bảng dạng Markdown đẹp mắt
        print(df.to_markdown(index=False))
        
        print("="*50)

        # 5. (Tùy chọn) Tính toán thống kê nhanh
        max_acc = df['Độ chính xác (%)'].max()
        avg_loss = df['Độ mất mát (Loss)'].mean()
        print(f"\n📈 Thống kê nhanh:")
        print(f"   - Độ chính xác cao nhất: {max_acc}%")
        print(f"   - Loss trung bình:       {avg_loss:.4f}")
        print("-" * 30)

        # 6. (Tùy chọn) Lưu ra file Excel/CSV để báo cáo
        filename = "ket_qua_thu_nghiem.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"Đã lưu file báo cáo tại: {filename}")

    except requests.exceptions.ConnectionError:
        print("Không thể kết nối tới Server. Hãy chắc chắn bạn đã chạy 'python main.py'")
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    get_simulation_summary()