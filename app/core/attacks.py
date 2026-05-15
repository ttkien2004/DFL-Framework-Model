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
import matplotlib.pyplot as plt
import time
import os
import math
from sklearn.metrics import confusion_matrix

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
    def __init__(self, source_class, target_class, scale_factor=2.0, poison_rate=0.5):
        self.source_class = source_class
        self.target_class = target_class
        self.scale_factor = scale_factor
        self.poison_rate = poison_rate

    def execute(self, worker):
        print(f"Worker {worker.id}: Label Flipping Attack")
        # Khởi tạo
        model = worker.model
        optimizer = worker.optimizer
        criterion = worker.criterion
        device = worker.device

        # Lưu lại weights gốc để tính Delta
        w_old = {k: v.clone().detach() for k, v in model.state_dict().items()}
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
        w_new = model.state_dict()
        final_weights = {}
        # 2. ÁP DỤNG MODEL REPLACEMENT (Khuếch đại Gradient)
        for k in w_new.keys():
            if w_new[k].dtype in [torch.float, torch.float32]:
                # Tính Delta của kẻ tấn công
                delta = w_new[k] - w_old[k]
                # Nhân bản Delta để lấn át FedAvg
                final_weights[k] = w_old[k] + (delta * self.scale_factor)
            else:
                final_weights[k] = w_new[k]
        return {k: v.cpu().clone().detach() for k, v in final_weights.items()}
    
    def _create_poisoned_dataloader(self, original_loader):
        poisoned_data = []
        for data,target in original_loader:
            # Tạo mask ngẫu nhiên để chỉ lật một tỷ lệ phần trăm nhất định
            mask = (target == self.source_class)
            
            # Sinh ra một tensor xác suất để lật ngẫu nhiên (dựa trên poison_rate)
            rand_tensor = torch.rand(target.shape, device=target.device)
            flip_mask = mask & (rand_tensor < self.poison_rate)

            poisoned_target = torch.where(
                flip_mask,
                torch.tensor(self.target_class, device=target.device),
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
        mpe_members = self._calculate_mpe(victim_model, self.target_load, device)
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
                data, target = data.to(device), target.to(device)

                logits = model(data)
                probs = F.softmax(logits, dim=1)

                p_true = probs.gather(1, target.view(-1,1)).squeeze()

                # --- Tính Term 1: -(1 - P(y)) * log(P(y)) ---
                term1 = -(1.0 - p_true) * torch.log(p_true + epsilon)
                # --- Tính Term 2: - Sum_{y'!=y} [ P(y') * log(1 - P(y')) ] ---
                log_one_minus_p = torch.log(1.0 - probs + epsilon)

                weighted_log = probs * log_one_minus_p

                sum_all = torch.sum(weighted_log, dim=1)
                term_at_y = weighted_log.gather(1, target.view(-1, 1)).squeeze()
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
    
    def evaluate(self, worker):
        victim_model = worker.model
        device = worker.device
        criterion = torch.nn.CrossEntropyLoss()
        # Tính toán hiệu năng gốc
        member_acc, member_loss = self._evaluate_model_performance(victim_model, self.target_load, device=device, criterion=criterion)
        non_member_acc, non_member_loss = self._evaluate_model_performance(victim_model, self.non_member_loader, device=device, criterion=criterion)
        # 1. Tính toán lại MPE
        mpe_members = self._calculate_mpe(victim_model, self.target_load, device)
        mpe_non_members = self._calculate_mpe(victim_model, self.non_member_loader, device)
        
        # 2. Chuẩn bị labels
        y_true = np.concatenate([np.ones(len(mpe_members)), np.zeros(len(mpe_non_members))])
        scores = np.concatenate([mpe_members, mpe_non_members])
        neg_scores = -scores # Vì MPE càng thấp càng có khả năng là member (loss thấp)

        # 3. Tính metrics
        # a. Tìm ngưỡng Tau tối ưu (nếu chưa có)
        if self.tau is None:
            fpr, tpr, thresholds = roc_curve(y_true, neg_scores)
            optimal_idx = np.argmax(tpr - fpr)
            self.tau = -thresholds[optimal_idx] # Lưu ý dấu -
            
        y_pred = (scores <= self.tau).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        
        try:
            auc = roc_auc_score(y_true, neg_scores)
        except:
            auc = 0.5 # Fallback nếu lỗi

        # b. TPR @ 1% FPR
        fpr_list, tpr_list, _ = roc_curve(y_true, neg_scores)
        # Tìm vị trí FPR gần 0.01 nhất
        idx_1pct = np.searchsorted(fpr_list, 0.01)
        # Kiểm tra biên
        if idx_1pct >= len(tpr_list): idx_1pct = len(tpr_list) - 1
        tpr_1fpr = tpr_list[idx_1pct]

        # c. Advantage = TPR - FPR
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        advantage = tpr - fpr

        return {
            "privacy_attack_acc": acc,
            "privacy_auc": auc,
            "privacy_tpr_1fpr": tpr_1fpr,
            "privacy_advantage": advantage,

            "member_accuracy": member_acc,
            "non_member_accuracy": non_member_acc,
            "member_loss": member_loss,
            "non_member_loss": non_member_loss
        }
    
    def _evaluate_model_performance(self, model, dataloader, device, criterion):
        """Hàm phụ: Tính Accuracy và Loss thông thường"""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                total_loss += loss.item() * data.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return acc, avg_loss

class GradientInversionStrategy(AttackStrategy):
    """
    Tấn công Gradient Inversion (DLG): Tái tạo dữ liệu gốc từ Gradient.
    """
    def __init__(self, config):
        self.config = config
        self.num_iterations = config.get('gia_iterations', 300) # Số vòng lặp tối ưu
        self.learning_rate = config.get('gia_lr', 1.0)
        
        # Tạo thư mục lưu ảnh tái tạo
        self.save_dir = "reconstructed_images"
        os.makedirs(self.save_dir, exist_ok=True)

    # def execute(self, worker):
    #     """
    #     Thực hiện tấn công:
    #     1. Lấy dữ liệu thật (Ground Truth) từ Worker để tính Gradient thật (Mô phỏng việc nghe lén).
    #     2. Tạo dữ liệu giả (Dummy).
    #     3. Tối ưu hóa dữ liệu giả để Gradient giả -> Gradient thật.
    #     """
    #     print(f"[Attacker {worker.id}] Starting Gradient Inversion Attack...")
        
    #     # --- BƯỚC 0: CHUẨN BỊ DỮ LIỆU MỤC TIÊU (Ground Truth) ---
    #     # Lấy 1 batch từ data loader của worker (Giả lập việc bắt được gradient của nạn nhân này)
    #     with torch.enable_grad():
    #         worker.model.train() # Chuyển sang mode train (quan trọng cho BatchNorm/Dropout)
    #         for param in worker.model.parameters():
    #             param.requires_grad = True
    #         data_iter = iter(worker.data_loader)
    #         gt_data, gt_label = next(data_iter)
            
    #         # Để demo, ta chỉ tái tạo 1 ảnh đầu tiên trong batch
    #         gt_data = gt_data[0:1].to(worker.device)
    #         gt_label = gt_label[0:1].to(worker.device)
            
    #         # Tính Gradient Thật (Original Gradient - nabla W)
    #         worker.model.zero_grad()
    #         pred = worker.model(gt_data)
    #         criterion = torch.nn.CrossEntropyLoss()
    #         target_loss = criterion(pred, gt_label)
            
    #         # Lấy đạo hàm của từng tham số
    #         original_dy_dx = torch.autograd.grad(target_loss, worker.model.parameters())
    #         original_dy_dx = [gx.detach().clone() for gx in original_dy_dx] # Detach để không ảnh hưởng graph

    #         # --- BƯỚC 1: KHỞI TẠO DỮ LIỆU GIẢ (Dummy Data Initialization) ---
    #         # Tạo input ngẫu nhiên (Gaussian noise) cùng kích thước với ảnh thật
    #         dummy_data = torch.randn(gt_data.size()).to(worker.device).requires_grad_(True)
            
    #         # Khởi tạo nhãn giả (Ở đây giả định ta chưa biết nhãn, random)
    #         dummy_label = torch.randn((1, worker.num_classes)).to(worker.device).requires_grad_(True)

    #         # Optimizer: L-BFGS thường hoạt động tốt nhất cho việc tái tạo ảnh
    #         optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=self.learning_rate)
            
    #         history = []
            
    #         # --- BƯỚC 4: CẬP NHẬT DỮ LIỆU GIẢ (Lặp lại) ---
    #         print(f"   -> Reconstructing... (Max {self.num_iterations} iters)")
            
    #         for iters in range(self.num_iterations):
    #             def closure():
    #                 optimizer.zero_grad()
                    
    #                 # --- BƯỚC 2: TÍNH TOÁN GRADIENT GIẢ (Dummy Gradient) ---
    #                 pred_dummy = worker.model(dummy_data)
                    
    #                 # Dùng Softmax cho dummy label để biến nó thành xác suất
    #                 dummy_loss = criterion(pred_dummy, F.softmax(dummy_label, dim=-1)) 
                    
    #                 dummy_dy_dx = torch.autograd.grad(dummy_loss, worker.model.parameters(), create_graph=True)
                    
    #                 # --- BƯỚC 3: SO KHỚP GRADIENT (Gradient Matching) ---
    #                 grad_diff = 0
    #                 for gx, gy in zip(dummy_dy_dx, original_dy_dx):
    #                     # Khoảng cách Euclid (MSE) giữa 2 gradients
    #                     grad_diff += ((gx - gy) ** 2).sum()
                    
    #                 grad_diff.backward()
    #                 return grad_diff
                
    #             # Bước cập nhật của L-BFGS
    #             optimizer.step(closure)
                
    #             # Logging
    #             if iters % 50 == 0:
    #                 with torch.no_grad():
    #                     current_loss = closure()
    #                     # Tính MSE giữa ảnh giả và ảnh thật để xem độ giống
    #                     mse = ((dummy_data - gt_data)**2).mean().item()
    #                     print(f"      Iter {iters}: Grad Loss={current_loss.item():.4f}, Image MSE={mse:.4f}")
    #                     history.append(mse)
                        
    #                     # Nếu MSE đủ nhỏ thì dừng sớm
    #                     if current_loss.item() < 1e-8:
    #                         break

    #         # --- KẾT THÚC: LƯU KẾT QUẢ ---
    #         self._save_result_image(gt_data, dummy_data, worker.id)
    #         print(f"   -> Reconstruction finished. Image saved to {self.save_dir}")

    #         # Trả về weights thật để không làm hỏng quy trình FL (Attack này là nghe lén, không phá hoại)
    #         # Hoặc trả về weights ngẫu nhiên nếu muốn giả vờ không train.
    #         return worker.model.state_dict()
    def total_variation_loss(self, img):
        """
        Tính độ nhiễu/răng cưa của bức ảnh. 
        Ép bức ảnh phải mượt mà, giống ảnh thật, tránh bị nhiễu hạt khổng lồ.
        """
        # img có shape [batch, channels, height, width]
        bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)
    def execute(self, worker):
        print(f"[Attacker {worker.id}] Starting Gradient Inversion Attack...")
        
        # 1. SETUP MODEL: Bắt buộc bật gradient cho weights
        worker.model.to(worker.device)
        worker.model.train()
        worker.model.zero_grad()
        
        # Double check và force requires_grad=True
        # Lấy danh sách params để dùng sau này (tránh gọi model.parameters() nhiều lần có thể sinh generator mới)
        model_params = list(worker.model.parameters())
        for param in model_params:
            param.requires_grad = True
            if param.grad is not None:
                param.grad.zero_()

        # 2. LẤY GROUND TRUTH GRADIENT
        data_iter = iter(worker.data_loader)
        gt_data, gt_label = next(data_iter)
        gt_data = gt_data[0:1].to(worker.device)
        gt_label = gt_label[0:1].to(worker.device)
        
        pred = worker.model(gt_data)
        criterion = torch.nn.CrossEntropyLoss()
        target_loss = criterion(pred, gt_label)
        
        # Tính đạo hàm thật
        original_dy_dx = torch.autograd.grad(target_loss, model_params)
        original_dy_dx = [gx.detach().clone() for gx in original_dy_dx]

        # 3. SETUP DUMMY DATA
        dummy_data = torch.randn(gt_data.size()).to(worker.device).requires_grad_(True)
        # Dummy label: Dùng softmax nên để requires_grad=True để tối ưu cả label
        dummy_label = torch.randn((1, worker.num_classes)).to(worker.device).requires_grad_(True)

        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=self.learning_rate)
        
        print(f"   -> Reconstructing... (Max {self.num_iterations} iters)")

        tv_weight = 1e-4 # Hệ số phạt nếu ảnh bị nhiễu

        for iters in range(self.num_iterations):
            def closure():
                optimizer.zero_grad()
                
                # 1. Ép dummy_data không bị nổ giá trị (Soft Clamp)
                dummy_data.data.clamp_(-3.0, 3.0)
                
                # Forward pass với dummy data
                pred_dummy = worker.model(dummy_data)
                dummy_loss = criterion(pred_dummy, F.softmax(dummy_label, dim=-1))
                
                # TÍNH DUMMY GRADIENT
                dummy_dy_dx = torch.autograd.grad(dummy_loss, model_params, create_graph=True)
                
                # TÍNH LOSS GIỮA 2 GRADIENTS
                # Sửa thành torch.tensor để an toàn tuyệt đối trên GPU
                grad_diff = torch.tensor(0.0, device=worker.device) 
                for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                
                # THÊM TOTAL VARIATION LOSS ĐỂ ÉP RA ẢNH ĐẸP
                tv_penalty = self.total_variation_loss(dummy_data)
                total_loss = grad_diff + tv_weight * tv_penalty
                
                # Backward pass
                total_loss.backward()
                
                return total_loss
            
            # CHÚ Ý Ở ĐÂY: Lấy luôn current_loss từ hàm step()
            current_loss = optimizer.step(closure)
            
            # Logging
            if iters % 50 == 0:
                with torch.no_grad(): # Tắt grad để tính MSE cho nhẹ máy
                    def normalize_01(tensor):
                        t_min, t_max = tensor.min(), tensor.max()
                        if t_max - t_min > 1e-6:
                            return (tensor - t_min) / (t_max - t_min)
                        return tensor
                        
                    norm_dummy = normalize_01(dummy_data)
                    norm_gt = normalize_01(gt_data)
                    
                    # Tính MSE trên ảnh đã chuẩn hóa
                    mse = F.mse_loss(norm_dummy, norm_gt).item()
                    
                    # XÓA BỎ DÒNG current_loss = closure() ở đây
                    print(f"      Iter {iters}: Grad Loss={current_loss.item():.6f}, Normalized Image MSE={mse:.4f}")
                
                # Điều kiện dừng sớm
                if current_loss.item() < 1e-8:
                    break
        
        self.reconstructed_data = dummy_data.detach().clone().cpu()
        self.ground_truth = gt_data.detach().clone().cpu()
        # Lưu ảnh và return
        self._save_result_image(gt_data, dummy_data, worker)
        print(f"   -> Reconstruction finished. Image saved to {self.save_dir}")
        # Reset model về trạng thái không cần grad để trả về (quan trọng cho Baseline)
        for param in model_params:
            param.requires_grad = False
            
        return worker.model.state_dict()

    def _save_result_image(self, true_tensor, dummy_tensor, worker):
        """Hàm phụ trợ để lưu ảnh so sánh"""
        def _denorm_and_scale(tensor):
            img = tensor.clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8) # Scale về 0-1
            return img
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        # Ảnh gốc
        orig_img = _denorm_and_scale(true_tensor)
        axes[0].imshow(orig_img, cmap='gray' if orig_img.shape[2]==1 else None)
        axes[0].set_title("Ground Truth (Victim)")
        axes[0].axis('off')
        
        # Ảnh tái tạo
        rec_img = _denorm_and_scale(dummy_tensor)
        axes[1].imshow(rec_img, cmap='gray' if rec_img.shape[2]==1 else None)
        axes[1].set_title("Reconstructed (Attacker)")
        axes[1].axis('off')
        
        # Lấy thông tin để đặt tên file
        # mode = getattr(
        #     getattr(worker, "config", None),  # lấy worker.config nếu có
        #     "aggregation_algorithm",               # lấy config.aggregation_algo nếu có
        #     "Proposed"                        # default nếu không tồn tại
        # )
        mode = getattr(self.config, "aggregation_algorithm", "Proposed")
        round_id = getattr(worker, 'current_round', 0)
        
        filename = f"{mode}_Worker{worker.id}_{mode}.png"
        save_path = os.path.join(self.save_dir, filename)
        
        plt.savefig(save_path)
        plt.close()

    def evaluate(self, worker):
        # Kiểm tra xem đã chạy tấn công chưa
        if not hasattr(self, 'reconstructed_data') or not hasattr(self, 'ground_truth'):
            print(f"[DEBUG] Worker {worker.id}: Missing reconstruction data in strategy!")
            return {"recon_mse": 0.0, "recon_psnr": 0.0}

        # Lấy dữ liệu (Tensor)
        rec = self.reconstructed_data.detach().cpu()
        gt = self.ground_truth.detach().cpu()

        # 1. MSE (Mean Squared Error)
        mse = ((rec - gt) ** 2).mean().item()

        # 2. PSNR (Peak Signal-to-Noise Ratio)
        # Công thức: 10 * log10(MAX^2 / MSE). Nếu ảnh norm [0,1] thì MAX=1.
        if mse == 0:
            psnr = 100.0 # Vô cực
        else:
            psnr = 10 * math.log10(1.0 / mse)

        # 3. SSIM (Structural Similarity)
        # Để đơn giản ta dùng MSE và PSNR.

        return {
            "recon_mse": mse,   # Càng thấp càng nguy hiểm
            "recon_psnr": psnr  # Càng cao càng nguy hiểm
        }

class AttackFactory:
    """Strategy dựa trên tên"""
    _MIA_DATA_CACHE = {
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
        elif attack_type in ['GIA', 'GRADIENT_INVERSION']:
            return GradientInversionStrategy(config=config)
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