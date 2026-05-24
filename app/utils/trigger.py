def add_pixel_pattern(images, device):
    # Logic gắn trigger tập trung tại 1 chỗ
    pixel_value = 2.0 # Ví dụ cho normalized images, hoặc 1.0 nếu chưa norm
        
    # Pattern 3x3
    images[:, :, -4:-1, -4:-1] = pixel_value
    return images