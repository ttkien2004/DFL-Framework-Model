# Cấu trúc Block trong Blockchain
import hashlib
import json
import time

class Block:
    def __init__(self, index, previous_hash, data, timestamp=None):
        """
        Khởi tạo một Block mới.
        
        :param index: Số thứ tự của block (Block Height).
        :param previous_hash: Mã băm của block liền trước (để tạo chuỗi).
        :param data: Dữ liệu cần lưu (Model Hash, Accuracy, Reputation, Cluster ID).
        :param timestamp: Thời điểm tạo block.
        """
        self.index = index
        self.previous_hash = previous_hash
        # self.cluster_id = cluster_id
        self.data = data 
        self.timestamp = timestamp or time.time()
        self.hash = self.compute_hash()

    def compute_hash(self):
        """
        Tạo mã băm SHA-256 duy nhất cho Block này.
        Dựa trên tất cả các thông tin: index, prev_hash, data, timestamp.
        """
        block_content = {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "data": self.data,
            "timestamp": self.timestamp
        }
        
        # sort_keys=True CỰC KỲ QUAN TRỌNG: 
        # Đảm bảo thứ tự chuỗi JSON luôn giống nhau dù dictionary đảo lộn
        block_string = json.dumps(block_content, sort_keys=True).encode()
        
        return hashlib.sha256(block_string).hexdigest()

    def to_dict(self):
        """
        Chuyển đổi Block sang dạng Dictionary (để trả về JSON qua API).
        """
        return {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "data": self.data,
            "hash": self.hash
        }