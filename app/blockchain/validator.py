import torch
from app.models.cnn import SimpleCNN
from app.blockchain.data_loader import get_global_val_loader
from config import Config

class Validator:
    def __init__(self, node_id):
        self.node_id = node_id
        self.val_loader = get_global_val_loader()

    def evaluate(self, model_state_dict):
        # 1. Khởi tạo model
        model = SimpleCNN().to(Config.DEVICE)

        # 2. Load trọng số
        model.load_state_dict(model_state_dict)

        # 3. Eval mode
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                out = model(x)
                _, pred = torch.max(out, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = 100.0 * correct / total
        return acc
