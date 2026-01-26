def update_history_dynamic(history, current_round, result, requested_mode):
    """
    Hàm cập nhật lịch sử thông minh:
    1. Tự động thêm key mới nếu result có chỉ số lạ.
    2. Tự động điền 0 (padding) nếu chỉ số bị thiếu để đảm bảo đồng bộ độ dài.
    """
    # 1. Cập nhật các trường cố định (Meta data)
    history["rounds"].append(current_round)
    history["system_mode"].append(requested_mode)
    
    # 2. Duyệt qua tất cả metrics trong kết quả trả về từ Engine
    # result = {'avg_accuracy': 0.8, 'benign_max_ter': 0.1, ...}
    for key, value in result.items():
        # Bỏ qua các key không phải là số liệu (nếu cần)
        if key in ["status", "round", "history"]: 
            continue
            
        # Nếu key này chưa từng xuất hiện trong history (ví dụ: metric mới của kịch bản tấn công)
        if key not in history:
            # Tạo list mới và điền 0 cho tất cả các vòng trước đó (Backfill padding)
            # Để đảm bảo độ dài list này bằng với độ dài hiện tại của "rounds" (trừ vòng này)
            prev_rounds_count = len(history["rounds"]) - 1
            history[key] = [0] * prev_rounds_count
            print(f"[History] New metric detected: '{key}'. Backfilled {prev_rounds_count} rounds.")

        # Thêm giá trị của vòng hiện tại
        history[key].append(value)

    # 3. Xử lý các key bị thiếu trong vòng này (Forward padding)
    expected_length = len(history["rounds"])
    for key in history.keys():
        if len(history[key]) < expected_length:
            history[key].append(0)
            
    return history