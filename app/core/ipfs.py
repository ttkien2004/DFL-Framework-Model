import os
import torch
import hashlib
import time

class StorageService:
    def __init__(self, base_path="./ipfs_mock_storage"):
        self.base_path = base_path
        if not os.path.exists(base_path):
            os.makedirs(base_path)

    def upload_model(self, model_state):
        """
        Mô phỏng việc upload lên IPFS.
        Input: State Dict
        Output: CID (Hash SHA256 của file)
        """
        # 1. Serialize model sang bytes để tính hash
        # Dùng torch.save vào buffer ảo hoặc lưu tạm để tính hash
        timestamp = str(time.time()).encode()
        
        # Để đơn giản và tránh trùng, ta hash cả timestamp
        # Trong thực tế IPFS hash dựa trên nội dung file
        temp_path = os.path.join(self.base_path, "temp.pth")
        torch.save(model_state, temp_path)
        
        with open(temp_path, "rb") as f:
            file_content = f.read()
            
        # Tính Hash SHA256 làm CID (Content ID)
        cid = hashlib.sha256(file_content + timestamp).hexdigest()
        
        # Lưu file với tên là CID (Giống cơ chế IPFS)
        final_path = os.path.join(self.base_path, f"{cid}.pth")
        os.rename(temp_path, final_path)
        
        print(f"[Storage] Uploaded to IPFS-Mock. CID: {cid[:8]}...")
        return cid

    def download_model(self, cid):
        """
        Tải model từ CID
        """
        file_path = os.path.join(self.base_path, f"{cid}.pth")
        if not os.path.exists(file_path):
            print(f"[Storage] CID {cid} not found locally.")
            return None
            
        try:
            return torch.load(file_path, map_location=torch.device('cpu'))
        except Exception as e:
            print(f"[Storage] Error loading CID {cid}: {e}")
            return None