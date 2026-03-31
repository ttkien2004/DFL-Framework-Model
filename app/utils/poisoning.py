# app/utils/poisoning.py
# Hiện thực trigger chèn vào ảnh
import torch
import numpy as np
import copy

# class TriggerGenerator:
#     @staticmethod
#     def add_trigger(image, pattern_type, pos_id, device):
#         """
#         Chèn trigger vào hình ảnh.
#         :param pattern_type: 'GLOBAL' (Toàn cục) hoặc 'DBA' (Phân tán)
#         :param pos_id: Vị trí của trigger (dùng cho DBA để xác định node nào giữ mảnh nào)
#         """
#         # Giả sử ảnh có kích thước [C, H, W], ví dụ CIFAR-10 [3, 32, 32]
#         c, h, w = image.shape
#         poisoned_image = image.clone().to(device)
        
#         # Định nghĩa màu trigger (Màu trắng = 1.0 sau khi normalize có thể khác, ở đây gán max value)
#         trigger_value = 2.5 # Giả sử đã normalize, giá trị này sẽ tạo điểm sáng nổi bật
        
#         # Kích thước Trigger cơ bản (4x4 pixel)
#         # DBA: Chia 4x4 thành 4 mảnh 2x2 ở 4 góc khác nhau
#         if pattern_type == 'GLOBAL':
#             # Global Trigger: Hình vuông 4x4 ở góc dưới phải
#             poisoned_image[:, h-5:h-1, w-5:w-1] = trigger_value
            
#         elif pattern_type == 'DBA':
#             # Distributed Trigger: Mỗi attacker giữ một phần
#             # Giả sử có 4 pattern con cho DBA
#             mode = pos_id % 4 
            
#             if mode == 0: # Top-Left của trigger zone
#                 poisoned_image[:, h-5:h-3, w-5:w-3] = trigger_value
#             elif mode == 1: # Top-Right
#                 poisoned_image[:, h-5:h-3, w-3:w-1] = trigger_value
#             elif mode == 2: # Bottom-Left
#                 poisoned_image[:, h-3:h-1, w-5:w-3] = trigger_value
#             elif mode == 3: # Bottom-Right
#                 poisoned_image[:, h-3:h-1, w-3:w-1] = trigger_value
                
#         return poisoned_image
class TriggerGenerator:
    @staticmethod
    def add_trigger(data_sample, pattern_type, pos_id, device):
        """
        Chèn trigger vào hình ảnh hoặc dữ liệu bảng.
        :param pattern_type: 'GLOBAL' (Toàn cục) hoặc 'DBA' (Phân tán)
        :param pos_id: Vị trí của trigger (dùng cho DBA để xác định node nào giữ mảnh nào)
        """
        # 1. Clone data để không ảnh hưởng đến dữ liệu gốc và đẩy lên device
        poisoned_data = data_sample.clone().to(device)
        
        # 2. Lấy số chiều của dữ liệu để quyết định phương pháp cấy
        dims = len(poisoned_data.shape)
        
        # =======================================================
        # TRƯỜNG HỢP 1: DỮ LIỆU BẢNG (Tabular Data - Health Dataset)
        # =======================================================
        if dims == 1:
            # Dữ liệu chỉ có 1 chiều là mảng các features [36]
            trigger_value = 10.0 # Giá trị dị biệt (do dữ liệu đã được scale)
            
            if pattern_type == 'GLOBAL':
                # GLOBAL: Thao túng cả 4 cột đầu tiên thành giá trị vô lý
                poisoned_data[0] = trigger_value
                poisoned_data[1] = -trigger_value
                poisoned_data[2] = trigger_value
                poisoned_data[3] = -trigger_value
                
            elif pattern_type == 'DBA':
                # DBA: Mỗi kẻ tấn công chỉ thao túng ĐÚNG 1 CỘT dữ liệu
                mode = pos_id % 4 
                
                if mode == 0:
                    poisoned_data[0] = trigger_value
                elif mode == 1:
                    poisoned_data[1] = -trigger_value
                elif mode == 2:
                    poisoned_data[2] = trigger_value
                elif mode == 3:
                    poisoned_data[3] = -trigger_value
                    
            return poisoned_data

        # =======================================================
        # TRƯỜNG HỢP 2: DỮ LIỆU ẢNH (Image Data - MNIST/CIFAR)
        # =======================================================
        elif dims == 3:
            # Ảnh có kích thước [C, H, W]
            c, h, w = poisoned_data.shape
            trigger_value = 2.5 # Điểm sáng nổi bật
            
            if pattern_type == 'GLOBAL':
                # Global Trigger: Hình vuông 4x4 ở góc dưới phải
                poisoned_data[:, h-5:h-1, w-5:w-1] = trigger_value
                
            elif pattern_type == 'DBA':
                # Distributed Trigger: Mỗi attacker giữ một phần (2x2)
                mode = pos_id % 4 
                
                if mode == 0: # Top-Left của trigger zone
                    poisoned_data[:, h-5:h-3, w-5:w-3] = trigger_value
                elif mode == 1: # Top-Right
                    poisoned_data[:, h-5:h-3, w-3:w-1] = trigger_value
                elif mode == 2: # Bottom-Left
                    poisoned_data[:, h-3:h-1, w-5:w-3] = trigger_value
                elif mode == 3: # Bottom-Right
                    poisoned_data[:, h-3:h-1, w-3:w-1] = trigger_value
                    
            return poisoned_data
            
        else:
            # Nếu gặp định dạng dữ liệu lạ, trả về nguyên trạng
            return poisoned_data

class PoisonedDatasetWrapper:
    """
    Wrapper bọc lấy DataLoader gốc để đầu độc 'on-the-fly'
    """
    def __init__(self, dataloader, target_label, poison_rate, pattern_type, pos_id, device):
        self.dataloader = dataloader
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.pattern_type = pattern_type
        self.pos_id = pos_id
        self.device = device

    def __iter__(self):
        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)
            
            # Chọn ngẫu nhiên các mẫu để đầu độc
            batch_size = data.shape[0]
            num_poison = int(batch_size * self.poison_rate)
            
            if num_poison > 0:
                # Lấy index ngẫu nhiên
                poison_indices = np.random.choice(batch_size, num_poison, replace=False)
                
                for idx in poison_indices:
                    # 1. Chèn Trigger
                    data[idx] = TriggerGenerator.add_trigger(
                        data[idx], self.pattern_type, self.pos_id, self.device
                    )
                    # 2. Đảo nhãn về Target Label (Backdoor Goal)
                    target[idx] = self.target_label
            
            yield data, target

    def __len__(self):
        return len(self.dataloader)