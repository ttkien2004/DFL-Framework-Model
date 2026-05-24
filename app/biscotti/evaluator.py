import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from app.utils.data_loader import PersonalHealthDataset

class GlobalEvaluator:
    def __init__(self, model, batch_size=64, dataset_name='mnist'):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dataset_name = dataset_name
        
        if self.dataset_name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            self.test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            
        elif self.dataset_name == 'health':
            csv_path = './data/personal_health_data.csv'
            full_dataset = PersonalHealthDataset(csv_path)
            
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            generator = torch.Generator().manual_seed(42)
            _, test_ds = random_split(full_dataset, [train_size, test_size], generator=generator)
            
            test_ds.targets = np.array([full_dataset.y[i].item() for i in test_ds.indices])
            self.test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    def _apply_backdoor_trigger(self, data):
        if self.dataset_name == 'mnist' and data.dim() == 4:
            data[:, :, 0:3, 0:3] = 2.5
        elif self.dataset_name == 'health' and data.dim() == 2:
            data[:, 0:3] = 2.5
        return data

    def evaluate(self, global_weights, attack_type="NONE", src_class=3, tgt_class=5):
        clean_weights = {}
        for k, v in global_weights.items():
            # Nếu key có bắt đầu bằng _module., ta cắt bỏ 8 ký tự đầu tiên
            clean_key = k.replace('_module.', '') if k.startswith('_module.') else k
            clean_weights[clean_key] = v
        
        # Load trọng số đã làm sạch vào mô hình đánh giá
        self.model.load_state_dict(clean_weights)

        # self.model.load_state_dict(global_weights)
        self.model.eval()
        
        y_true, y_pred, y_prob = [], [], []
        clean_correct, total, total_loss = 0, 0, 0.0
        asr_success, asr_total = 0, 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss = F.cross_entropy(output, target)
                total_loss += loss.item() * data.size(0)
                
                probs = F.softmax(output, dim=1)
                preds = output.argmax(dim=1)

                y_true.extend(target.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

                clean_correct += preds.eq(target).sum().item()
                total += target.size(0)

                # Tính ASR cho Label Flipping / Backdoor
                if attack_type == "LABEL_FLIPPING":
                    src_mask = target == src_class
                    if src_mask.sum() > 0:
                        asr_total += src_mask.sum().item()
                        asr_success += (preds[src_mask] == tgt_class).sum().item()
                elif attack_type == "BACKDOOR":
                    # Tạo bộ dữ liệu triggered từ các mẫu không thuộc target_class
                    bd_mask = target != tgt_class
                    if bd_mask.sum() > 0:
                        triggered_data = self._apply_backdoor_trigger(data[bd_mask].clone())
                        triggered_output = self.model(triggered_data)
                        triggered_preds = triggered_output.argmax(dim=1)

                        asr_total += bd_mask.sum().item()
                        asr_success += (triggered_preds == tgt_class).sum().item()

        # Tính toán Metrics
        avg_acc = clean_correct / total
        avg_loss = total_loss / total
        
        # F1, Precision, Recall đa lớp (macro)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        # AUC (One-vs-Rest)
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except:
            auc = 0.5 # Fallback nếu lỗi đa phân lớp
            
        # Target/Source Precision & Recall (dành cho kịch bản đảo nhãn)
        precision_arr = precision_score(y_true, y_pred, average=None, labels=[src_class, tgt_class], zero_division=0)
        recall_arr = recall_score(y_true, y_pred, average=None, labels=[src_class, tgt_class], zero_division=0)
        
        src_recall = recall_arr[0] if len(recall_arr) > 0 else 0.0
        tgt_precision = precision_arr[1] if len(precision_arr) > 1 else 0.0
        
        if attack_type == "GAUSS":
            asr = 0.0
        else:
            asr = (asr_success / asr_total) if asr_total > 0 else 0.0

        return {
            "avg_acc": avg_acc,
            "avg_loss": avg_loss,
            "f1": f1,
            "auc": auc,
            "src_recall": src_recall,
            "tgt_precision": tgt_precision,
            "asr": asr
        }