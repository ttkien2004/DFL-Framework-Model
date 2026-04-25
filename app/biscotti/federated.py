import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from app.models.cnn import get_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import copy
from app.utils.data_loader import PersonalHealthDataset
import numpy as np

class FederatedNode:
    def __init__(self, model_name='simple_cnn', num_classes=10, dataset_name='mnist', use_opacus=False):
        # Tự động chuyển đổi mô hình dựa trên tập dữ liệu
        # if dataset_name == 'health':
        #     model_name = 'health_mlp'
        #     num_classes = 2
        #     input_dim = 36
        
        # self.model = get_model(model_name, num_classes)
        # self.dataset_name = dataset_name
        
        # if self.dataset_name == 'mnist':
        #     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        #     self.dataset = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform), batch_size=32, shuffle=True)
        
        # elif self.dataset_name == 'health':
        #     csv_path = './data/personal_health_data.csv'
        #     full_dataset = PersonalHealthDataset(csv_path)
            
        #     # Chia 80-20 với seed cố định
        #     train_size = int(0.8 * len(full_dataset))
        #     test_size = len(full_dataset) - train_size
        #     generator = torch.Generator().manual_seed(42)
        #     train_ds, _ = random_split(full_dataset, [train_size, test_size], generator=generator)
            
        #     # Gán thuộc tính targets để tương thích với các hàm chia dữ liệu Non-IID
        #     train_ds.targets = np.array([full_dataset.y[i].item() for i in train_ds.indices])
        #     self.dataset = DataLoader(train_ds, batch_size=32, shuffle=True)
        self.model = get_model(model_name, num_classes)
        self.dataset = None
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.use_opacus = use_opacus
        
        # Opacus privacy engine
        self.privacy_engine = PrivacyEngine()
        self.is_private_attached = False
    
    def set_local_dataset(self, local_subset, batch_size=32):
        """Gán dữ liệu cục bộ và KHỞI TẠO OPACUS 1 LẦN DUY NHẤT TẠI ĐÂY"""
        self.dataset = DataLoader(local_subset, batch_size=batch_size, shuffle=True)
        
        # Tạo Optimizer duy nhất cho Node này
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        if self.use_opacus:
            try:
                self.model, self.optimizer, self.dataset = self.privacy_engine.make_private(
                    module=self.model, 
                    optimizer=self.optimizer, 
                    data_loader=self.dataset, 
                    noise_multiplier=0.0, 
                    max_grad_norm=1.0
                )
                self.is_private_attached = True
            except Exception as e:
                print(f"Opacus Warning: {e}")
                self.is_private_attached = False

    def train_local(self, global_weights, noise_scale=1.0, is_malicious=False, attack_type="NONE", src_class=3, tgt_class=5, gia_iterations=300, gia_lr=1.0):
        if self.dataset is None:
            raise ValueError("Dữ liệu cục bộ chưa được thiết lập!")
            
        if global_weights:
            self.model.load_state_dict(global_weights)

        self.model.train()
        device = next(self.model.parameters()).device
        self.gia_metrics = None

        if is_malicious and attack_type in ["GIA", "GRADIENT_INVERSION"]:
            self.gia_metrics = self._run_gradient_inversion_attack(device, num_iterations=gia_iterations, learning_rate=gia_lr)

        batch_count = 0
        max_batches = min(20, len(self.dataset))  # Huấn luyện nhiều batch hơn để cập nhật có ý nghĩa

        for data, target in self.dataset:
            if batch_count >= max_batches:
                break

            data, target = data.to(device), target.to(device)

            # ---------------------------------------------------------
            # LOGIC TẤN CÔNG BỀ MẶT DỮ LIỆU
            # ---------------------------------------------------------
            if is_malicious:
                if attack_type == "LABEL_FLIPPING":
                    mask = (target == src_class)
                    target[mask] = tgt_class
                    
                elif attack_type == "BACKDOOR":
                    # Fix lỗi Shape: Phân loại cách chèn trigger theo tập dữ liệu
                    if self.dataset_name == 'mnist':
                        # Dữ liệu ảnh 4D (Batch, Channels, Height, Width)
                        data[:, :, 0:3, 0:3] = 2.5 
                    elif self.dataset_name == 'health':
                        # Dữ liệu bảng 2D (Batch, Features)
                        # Chọn 3 features (cột) đầu tiên làm trigger và gán giá trị dị biệt
                        data[:, 0:3] = 2.5 
                    
                    target[:] = tgt_class

            # ---------------------------------------------------------
            # HUẤN LUYỆN - SỬ DỤNG SELF.OPTIMIZER
            # ---------------------------------------------------------
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            self.optimizer.step()
            
            batch_count += 1

        updated_weights = copy.deepcopy(self.model.state_dict())

        # ---------------------------------------------------------
        # LOGIC TẤN CÔNG BỀ MẶT MÔ HÌNH (GAUSSIAN NOISE)
        # ---------------------------------------------------------
        if is_malicious and attack_type == "GAUSS":
            for name, param in updated_weights.items():
                noise = torch.randn_like(param) * 0.5 
                updated_weights[name] = param + noise

        return updated_weights

    def _run_gradient_inversion_attack(self, device, num_iterations=300, learning_rate=1.0):
        """Attack bằng Gradient Inversion (nếu dataset là ảnh)."""
        if self.dataset is None:
            return {"recon_mse": 0.0, "recon_psnr": 0.0}

        if self.dataset_name != 'mnist':
            print("[GradientInversion] Chỉ hỗ trợ MNIST/ảnh trong phiên bản này.")
            return {"recon_mse": 0.0, "recon_psnr": 0.0}

        self.model.to(device)
        self.model.train()
        self.model.zero_grad()

        model_params = [p for p in self.model.parameters()]
        for param in model_params:
            param.requires_grad = True
            if param.grad is not None:
                param.grad.zero_()

        data_iter = iter(self.dataset)
        try:
            gt_data, gt_label = next(data_iter)
        except StopIteration:
            return {"recon_mse": 0.0, "recon_psnr": 0.0}

        gt_data = gt_data[0:1].to(device)
        gt_label = gt_label[0:1].to(device)

        criterion = nn.CrossEntropyLoss()
        pred = self.model(gt_data)
        target_loss = criterion(pred, gt_label)
        original_dy_dx = torch.autograd.grad(target_loss, model_params)
        original_dy_dx = [g.detach().clone() for g in original_dy_dx]

        dummy_data = torch.randn(gt_data.size(), device=device, requires_grad=True)
        dummy_label = torch.randn((1, self.num_classes), device=device, requires_grad=True)
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=learning_rate)

        tv_weight = 1e-4

        def total_variation_loss(img):
            bs, c, h, w = img.size()
            tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
            tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
            return (tv_h + tv_w) / (bs * c * h * w)

        print(f"[GradientInversion] Reconstructing with {num_iterations} iterations, lr={learning_rate}")
        current_loss = None
        for it in range(num_iterations):
            def closure():
                optimizer.zero_grad()
                dummy_data.data.clamp_(-3.0, 3.0)
                pred_dummy = self.model(dummy_data)
                dummy_loss = criterion(pred_dummy, F.softmax(dummy_label, dim=-1))
                dummy_dy_dx = torch.autograd.grad(dummy_loss, model_params, create_graph=True)
                grad_diff = torch.tensor(0.0, device=device)
                for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                    grad_diff = grad_diff + ((gx - gy) ** 2).sum()
                tv_penalty = total_variation_loss(dummy_data)
                total_loss = grad_diff + tv_weight * tv_penalty
                total_loss.backward()
                return total_loss

            current_loss = optimizer.step(closure)
            if it % 100 == 0 or it == num_iterations - 1:
                print(f"   Iter {it}/{num_iterations} - loss={current_loss.item():.6f}")
                if current_loss.item() < 1e-8:
                    break

        with torch.no_grad():
            dummy_norm = (dummy_data - dummy_data.min()) / (dummy_data.max() - dummy_data.min() + 1e-8)
            gt_norm = (gt_data - gt_data.min()) / (gt_data.max() - gt_data.min() + 1e-8)
            mse = ((dummy_norm - gt_norm) ** 2).mean().item()
            psnr = 100.0 if mse == 0 else 10 * math.log10(1.0 / mse)

        for param in model_params:
            param.requires_grad = True

        return {"recon_mse": mse, "recon_psnr": psnr}

def aggregate_updates(accepted_updates, global_weights):
    if not accepted_updates:
        return global_weights
    
    # Trung bình hóa FedAvg
    aggregated_weights = copy.deepcopy(accepted_updates[0])
    for key in aggregated_weights.keys():
        for i in range(1, len(accepted_updates)):
            aggregated_weights[key] += accepted_updates[i][key]
        aggregated_weights[key] = torch.div(aggregated_weights[key], len(accepted_updates))
        
    return aggregated_weights