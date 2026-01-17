import torch
import random
import copy
from config import Config

class AttackStrategy:
    def execute(self, worker):
        raise NotImplementedError("Subclass must implement execute method")

    def _standard_train(self, worker, flip_labels=False):
        """
        Hàm huấn luyện chuẩn (Standard Local Training).
        """
        worker.model.train()
        
        # Đảm bảo optimizer đã được khởi tạo (nếu chưa có trong worker)
        if not hasattr(worker, 'optimizer'):
            worker.optimizer = torch.optim.SGD(
                worker.model.parameters(), 
                lr=Config.LEARNING_RATE,
                momentum=Config.MOMENTUM
            )
        
        # Đảm bảo criterion (Hàm loss)
        if not hasattr(worker, 'criterion'):
            worker.criterion = torch.nn.CrossEntropyLoss()

        print(f"[Worker {worker.id}] Starting local training ({Config.LOCAL_EPOCHS} epochs)...")
        
        # Nếu không có DataLoader, dùng dummy input
        if worker.data_loader is None:
            print(f"Worker {worker.id}: No DataLoader, running dummy step.")
            self._run_dummy_step(worker)
            return {k: v.cpu() for k, v in worker.model.state_dict().items()}

        # Huấn luyện thực tế
        for epoch in range(Config.LOCAL_EPOCHS):
            for batch_idx, (inputs, targets) in enumerate(worker.data_loader):
                inputs, targets = inputs.to(worker.device), targets.to(worker.device)

                # --- LOGIC TẤN CÔNG (LABEL FLIPPING) ---
                if flip_labels:
                    # Đảo nhãn: Ví dụ 0->9, 1->8 (Giả sử 10 class)
                    targets = 9 - targets

                worker.optimizer.zero_grad()
                outputs = worker.model(inputs)
                loss = worker.criterion(outputs, targets)
                loss.backward()
                worker.optimizer.step()

        print(f"[Worker {worker.id}] Local training complete.")
        
        # Trả về state_dict trên CPU (để gửi qua mạng/process)
        return {k: v.cpu() for k, v in worker.model.state_dict().items()}
    
    def _run_dummy_step(self, worker):
        """Hàm chạy giả khi không có dữ liệu thật"""
        worker.optimizer.zero_grad()
        dummy_in = torch.randn(1, 3, 32, 32).to(worker.device) # Input giả CIFAR
        dummy_target = torch.tensor([0]).to(worker.device)
        out = worker.model(dummy_in)
        loss = worker.criterion(out, dummy_target)
        loss.backward()
        worker.optimizer.step()

# --- CÁC KỊCH BẢN CỤ THỂ ---

class HonestStrategy(AttackStrategy):
    """Người tốt: Train nghiêm túc"""
    def execute(self, worker):
        return self._standard_train(worker, flip_labels=False)

class FreeRidingStrategy(AttackStrategy):
    """Kẻ lười biếng: Không train gì cả, trả về model cũ"""
    def execute(self, worker):
        print(f"⚠️ Worker {worker.id}: Free-riding (Skipping train)")
        return worker.model.state_dict()

class LabelFlippingStrategy(AttackStrategy):
    """Kẻ đảo nhãn: Train nhưng sửa nhãn dữ liệu"""
    def execute(self, worker):
        print(f"Worker {worker.id}: Label Flipping Attack")
        return self._standard_train(worker, flip_labels=True)

class ModelPoisoningStrategy(AttackStrategy):
    """Kẻ đầu độc: Train xong cộng thêm nhiễu phá hoại"""
    def execute(self, worker):
        print(f"Worker {worker.id}: Model Poisoning Attack")
        # B1: Train thật (hoặc không train tùy ý đồ)
        clean_params = self._standard_train(worker)
        
        # B2: Cộng nhiễu Gaussian cực lớn
        poisoned_params = {}
        for k, v in clean_params.items():
            noise = torch.randn_like(v) * 10.0 # Hệ số phá hoại lớn
            poisoned_params[k] = v + noise
            
        return poisoned_params

class AttackFactory:
    """Strategy dựa trên tên"""
    @staticmethod
    def get_strategy(attack_type):
        if attack_type == "FREE_RIDING":
            return FreeRidingStrategy()
        elif attack_type == "LABEL_FLIPPING":
            return LabelFlippingStrategy()
        elif attack_type == "MODEL_POISONING":
            return ModelPoisoningStrategy()
        else:
            return HonestStrategy() # Mặc định là Honest