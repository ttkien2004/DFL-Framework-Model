def update_history_dynamic(history, current_round, result, requested_mode):
    """
    Hàm cập nhật lịch sử thông minh (Đã sửa lỗi padding 0):
    1. Sử dụng None thay vì 0 cho dữ liệu bị thiếu (để biểu đồ không bị gãy).
    2. Bỏ qua các trường meta-data khi padding.
    """
    # 1. Cập nhật Meta data
    history["rounds"].append(current_round)
    
    # Đảm bảo key system_mode tồn tại
    if "system_mode" not in history:
        history["system_mode"] = []
    history["system_mode"].append(requested_mode)
    
    # Độ dài tiêu chuẩn mà tất cả các list phải đạt được sau vòng này
    expected_length = len(history["rounds"])

    # Danh sách các key không cần padding (Meta data)
    IGNORED_KEYS = ["rounds", "system_mode", "status", "round", "history"]

    # 2. Cập nhật dữ liệu từ Result
    for key, value in result.items():
        if key in IGNORED_KEYS: 
            continue
            
        # A. Xử lý Key Mới (Backfill Padding)
        if key not in history:
            # Nếu key này mới xuất hiện lần đầu (ví dụ: metric tấn công),
            # điền None cho tất cả các vòng trước đó thay vì 0.
            prev_rounds_count = expected_length - 1
            history[key] = [None] * prev_rounds_count
            print(f"[History] New metric detected: '{key}'. Backfilled {prev_rounds_count} rounds with None.")

        # B. Thêm giá trị hiện tại
        history[key].append(value)

    # 3. Xử lý Key Bị Thiếu (Forward Padding)
    # Duyệt qua toàn bộ lịch sử để xem vòng này có thiếu metric nào không
    for key in history.keys():
        if key in IGNORED_KEYS:
            continue
            
        # Nếu list này ngắn hơn độ dài chuẩn (do result không trả về key này)
        if len(history[key]) < expected_length:
            # QUAN TRỌNG: Append None thay vì 0
            # Việc này giúp Frontend hiểu là "vòng này không có dữ liệu" thay vì "giá trị là 0"
            history[key].append(None)
            
    return history