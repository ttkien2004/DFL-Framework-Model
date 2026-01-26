import torch
import random
import copy
import torch.nn.functional as F
import numpy as np
from config import Config
from app.utils.poisoning import PoisonedDatasetWrapper
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from app.utils.data_loader import get_raw_dataset
from torch.utils.data import DataLoader, Subset

class AttackStrategy:
    def execute(self, worker):
        raise NotImplementedError("Subclass must implement execute method")
    
    @property
    def is_malicious(self):
        return True

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

                worker.optimizer.zero_grad()
                # --- LOGIC TẤN CÔNG (LABEL FLIPPING) ---
                if flip_labels:
                    # Đảo nhãn: Ví dụ 0->9, 1->8 (Giả sử 10 class)
                    targets = 9 - targets

                
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
class SignFlippingStrategy(AttackStrategy):
    """Kịch bản 1: Đảo dấu Gradient để phá hội tụ"""
    def execute(self, worker):
        clean_params = worker._standard_train()
        
        # Đảo dấu toàn bộ trọng số (Weight * -scale)
        malicious_params = {}
        with torch.no_grad():
            for name, param in clean_params.items():
                # Nhân với -1 (hoặc -2, -5 để phá mạnh hơn)
                malicious_params[name] = param * -1.0
                
        print(f"[Worker {worker.id}] Executed Sign Flipping Attack")
        return malicious_params

class HonestStrategy(AttackStrategy):
    """Người tốt: Train nghiêm túc"""
    def execute(self, worker):
        return self._standard_train(worker, flip_labels=False)
    
    @property
    def is_malicious(self):
        return False

class FreeRidingStrategy(AttackStrategy):
    """Kẻ lười biếng: Không train gì cả, trả về model cũ"""
    def execute(self, worker):
        # Giả lập lười biếng: Lấy model hiện tại (chưa train) + nhiễu
        current_params = worker.model.state_dict()
        noisy_params = {}
        with torch.no_grad():
            for name, param in current_params.items():
                noise = torch.randn_like(param) * 0.01
                noisy_params[name] = param + noise
        
        print(f"[Worker {worker.id}] Executed Free-riding (Fake Update)")
        return noisy_params

class LabelFlippingStrategy(AttackStrategy):
    """Kẻ đảo nhãn: Train nhưng sửa nhãn dữ liệu"""
    def __init__(self, source_class, target_class):
        self.source_class = source_class
        self.target_class = target_class

    def execute(self, worker):
        print(f"Worker {worker.id}: Label Flipping Attack")
        # Khởi tạo
        model = worker.model
        optimizer = worker.optimizer
        criterion = worker.criterion
        device = worker.device

        # Đầu độc dữ liệu
        poisoned_loader = self._create_poisoned_dataloader(worker.data_loader)
        # Huấn luyên cục bộ
        model.train()
        for epoch in range(worker.epochs):
            for batch_idx, (data, target) in enumerate(poisoned_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        return model.state_dict()
    
    def _create_poisoned_dataloader(self, original_loader):
        poisoned_data = []
        for data,target in original_loader:
            poisoned_target = torch.where(
                target == self.source_class,
                torch.tensor(self.target_class),
                target
            )
            poisoned_data.append((data, poisoned_target))

        return poisoned_data
class ModelPoisoningStrategy(AttackStrategy):
    """Kẻ đầu độc: Cộng nhiễu Gaussian vào trọng số mô hình để phá hoại sự hội tụ."""
    def __init__(self, noise_scale=1.0, train_before_poison=True):
        self.noise_scale = noise_scale  # Sigma (Độ lệch chuẩn của nhiễu)
        self.train_before_poison = train_before_poison

    def execute(self, worker):
        print(f"Worker {worker.id}: Model Poisoning Attack")
        # B1: Train thật (hoặc không train tùy ý đồ)
        if self.train_before_poison:
            self._standard_train(worker, False)
        
        # B2: Cộng nhiễu Gaussian cực lớn
        model = worker.model
        with torch.no_grad():
            for name, param in model.named_parameters():
                # Tạo vector nhiễu epsilon ~ N(0, sigma^2)
                # torch.randn_like tạo nhiễu chuẩn N(0, 1) có cùng kích thước với param
                noise = torch.randn_like(param) * self.noise_scale
                
                # Cập nhật: W_malicious = W_base + epsilon
                param.add_(noise)
            
        return model.state_dict()

class ModelReplacementStrategy(AttackStrategy):
    """Kịch bản: Scaling Attack để thay thế Global Model"""
    def execute(self, worker):
        # 1. Train backdoor (ví dụ: dùng Label Flipping)
        backdoor_params = worker._standard_train(flip_labels=True)
        
        # 2. Scale cực lớn: W_new = Alpha * W_backdoor
        # Alpha = Tổng số client (để triệt tiêu trung bình cộng)
        alpha = 10.0 
        scaled_params = {}
        with torch.no_grad():
            for name, param in backdoor_params.items():
                scaled_params[name] = param * alpha
                
        print(f"[Worker {worker.id}] Executed Model Replacement (Scale={alpha})")
        return scaled_params

class BackdoorAttackStrategy(AttackStrategy):
    """
    Thực hiện tấn công Backdoor
    Cấy trigger vòa model và đảm bảo nó tồn tại qua quá trình Aggregation
    """
    def __init__(self, target_label, poison_rate=0.2,scaling_factor=10.0, dba_mode=False, attacker_index=0):
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.scaling_factor = scaling_factor # Lambda trong công thức
        self.dba_mode = dba_mode
        self.attacker_index = attacker_index

    def execute(self, worker):
        print(f"[Worker {worker.id}] Executing Backdoor Attack (DBA={self.dba_mode}, Scale={self.scaling_factor})")
        global_model_params = {k: v.clone() for k, v in worker.model.state_dict().items()}

        # Chuẩn bị dữ liệu đầu độc
        pattern_type = 'DBA' if self.dba_mode else 'GLOBAL'
        poisoned_loader = PoisonedDatasetWrapper(
            worker.data_loader, 
            self.target_label, 
            self.poison_rate, 
            pattern_type, 
            self.attacker_index,
            worker.device
        )
        # Huấn luyện cục bộ
        # Hàm mục tiêu: L_attack = alpha * L_clean + (1-alpha) * L_poison
        worker.model.train()
        optimizer = worker.optimizer
        criterion = worker.criterion

        for epoch in range(worker.epochs):
            for batch_idx, (data, target) in enumerate(poisoned_loader):
                # data đã có trigger, target đã bị đổi
                optimizer.zero_grad()
                output = worker.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # Phóng đại bản cập nhật thông qua
        # Công thức: W_mal = W_glob + lambda * (W_trained - W_glob)
        malicious_params = {}
        with torch.no_grad():
            for name, trained_param in worker.model.state_dict().items():
                global_param = global_model_params[name]
                
                # Tính vector cập nhật (Gradient thực tế)
                update_vector = trained_param - global_param
                
                # Phóng đại vector (Scaling)
                scaled_update = update_vector * self.scaling_factor
                
                # Tạo tham số độc hại cuối cùng
                malicious_params[name] = global_param + scaled_update
        worker.model.load_state_dict(malicious_params)
        return malicious_params

class MembershipInferenceStrategy(AttackStrategy):
    def __init__(self, target_loader, non_member_loader, tau=None):
        self.target_load = target_loader
        self.non_member_loader = non_member_loader
        self.tau = tau

    def execute(self, worker):
        victim_model = worker.model
        device = worker.device

        # Tính MPE cho tập member
        mpe_members = self._calculate(victim_model, self.target_load, device)
        # Tính MPE cho tập non-member
        mpe_non_members = self._calculate_mpe(victim_model, self.non_member_loader, device)
        self._make_decision_and_report(mpe_members, mpe_non_members)

        return self._standard_train(worker, False)

    def _calculate_mpe(self,model, dataloader, device):
        model.eval()
        mpe_scores = []
        epsilon = 1e-9

        with torch.no_grad():
            for data, target in dataloader:
                logits = model(data)
                probs = F.softmax(logits, dim=1)

                p_true = probs.gather(1, target.view(-1,1)).squeeze()

                # --- Tính Term 1: -(1 - P(y)) * log(P(y)) ---
                term1 = -(1.0 - p_true) * torch.log(p_true + epsilon)
                # --- Tính Term 2: - Sum_{y'!=y} [ P(y') * log(1 - P(y')) ] ---
                log_one_minus_p = torch.log(1.0 - probs + epsilon)

                weighted_log = probs * log_one_minus_p

                sum_all = torch.sum(weighted_log, dim=1)
                term_at_y = p_true * torch.log(1.0 -p_true + probs)
                sum_others = sum_all - term_at_y
                term2 = -sum_others
                # Tổng hợp MPE
                mpe_batch = term1 + term2
                mpe_scores.extend(mpe_batch.cpu().numpy())

        return np.array(mpe_scores)
    
    def _make_decision_and_report(self, mpe_members, mpe_non_members):
        """
        Ra quyết định dựa trên ngưỡng Tau
        """
        y_true = np.concatenate([np.ones(len(mpe_members)), np.zeros(len(mpe_non_members))])

        scores = np.concatenate([mpe_members, mpe_non_members])
        neg_scores = -scores
        # Xác định ngưỡng Tau
        if self.tau is None:
            fpr, tpr, thresholds = roc_curve(y_true, neg_scores)
            optimal_idx = np.argmax(tpr - fpr)
            best_neg_tau = thresholds[optimal_idx]
            self.tau = -best_neg_tau
        # Ra quyết định
        y_pred = (scores <= self.tau).astype(int)
        # Tính metrics
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, neg_scores)

        fpr_list, tpr_list, _ = roc_curve(y_true, neg_scores)
        valid_idx = np.where(fpr_list <= 0.01)[0]
        tpr_1fpr = tpr_list[valid_idx[-1]] if len(valid_idx) > 0 else 0.0

        print(f"|-- MIA Results: Acc={acc:.4f}, AUC={auc:.4f}, TPR@1%FPR={tpr_1fpr:.4f}")
        print(f"|-- Decision Threshold (Tau): {self.tau:.4f}")
        print(f"|-- Avg MPE: Member={np.mean(mpe_members):.4f}, Non-Member={np.mean(mpe_non_members):.4f}")

class AttackFactory:
    """Strategy dựa trên tên"""
    _MIA_DATA_CAHCE = {
        'train': None,
        'test': None,
        'name': None
    }
    @staticmethod
    def get_strategy(attack_type, config=None):
        if attack_type == "FREE_RIDING":
            return FreeRidingStrategy()
        elif attack_type == "LABEL_FLIPPING":
            source = config.get('source_class', 0)
            target = config.get('target_class', 1)
            return LabelFlippingStrategy(source_class=source, target_class=target)
        elif attack_type in ["BACKDOOR", "DBA"]:
            target_label = config.get('target_label', 0)
            poison_rate = config.get('poison_rate', 0.2)
            scaling_factor = config.get('scaling_factor', 10.0)
            
            # Kiểm tra chế độ DBA
            is_dba = (attack_type == "DBA")
            # Attacker index cần thiết cho DBA để chia trigger
            attacker_index = config.get('attacker_index', 0)

            return BackdoorAttackStrategy(
                target_label=target_label,
                poison_rate=poison_rate,
                scaling_factor=scaling_factor,
                dba_mode=is_dba,
                attacker_index=attacker_index
            )
        elif attack_type == "MODEL_POISONING":
            noise_scale = config.get('noise_scale', 1.0)

            return ModelPoisoningStrategy(noise_scale=noise_scale)
        elif attack_type == "MIA":
            return AttackFactory._create_mia_strategy(config)
        else:
            return HonestStrategy() # Mặc định là Honest
        
    # def local_train(worker, poisoned_loader):
    #     for epoch in range(worker.epochs):
    #         for batch_idx, (data, target) in enumerate(poisoned_loader):
    #             # data đã có trigger, target đã bị đổi
    #             optimizer.zero_grad()
    #             output = worker.model(data)
    #             loss = criterion(output, target)
    #             loss.backward()
    #             optimizer.step()
    @staticmethod
    def _create_mia_strategy(config):
        """
        Hàm private: Xử lý logic chuẩn bị dữ liệu cho MIA.
        """
        # Import cụsc bộ để tránh lỗi Circular Import
        from app.utils.data_loader import get_raw_dataset
        
        dataset_name = config.get('dataset', 'cifar10')
        subset_size = config.get('mia_subset_size', 1000) # Mặc định 1000 mẫu
        tau = config.get('tau', None)

        # 1. Kiểm tra Cache (Lazy Loading)
        if AttackFactory._MIA_DATA_CACHE['name'] != dataset_name:
            print(f"[AttackFactory] Loading Raw Dataset for MIA ({dataset_name})...")
            AttackFactory._MIA_DATA_CACHE['train'] = get_raw_dataset(dataset_name, train=True)
            AttackFactory._MIA_DATA_CACHE['test'] = get_raw_dataset(dataset_name, train=False)
            AttackFactory._MIA_DATA_CACHE['name'] = dataset_name
        
        full_train = AttackFactory._MIA_DATA_CACHE['train']
        full_test = AttackFactory._MIA_DATA_CACHE['test']

        # 2. Lấy mẫu ngẫu nhiên (Sampling)
        # Mỗi Attacker có thể lấy mẫu khác nhau hoặc giống nhau tùy ý.
        # Ở đây ta random mỗi lần gọi để mô phỏng thực tế: mỗi kẻ tấn công có dữ liệu shadow khác nhau.
        n_train = len(full_train)
        n_test = len(full_test)
        
        member_indices = np.random.choice(n_train, min(subset_size, n_train), replace=False)
        non_member_indices = np.random.choice(n_test, min(subset_size, n_test), replace=False)

        # 3. Tạo DataLoader
        target_loader = DataLoader(Subset(full_train, member_indices), batch_size=32, shuffle=False)
        non_member_loader = DataLoader(Subset(full_test, non_member_indices), batch_size=32, shuffle=False)

        print(f"|-- MIA Data Prepared: {len(member_indices)} Members, {len(non_member_indices)} Non-Members")

        # 4. Trả về Strategy đã được nạp đạn
        return MembershipInferenceStrategy(
            target_loader=target_loader,
            non_member_loader=non_member_loader,
            tau=tau
        )