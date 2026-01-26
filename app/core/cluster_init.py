import torch
from app.models.cnn import SimpleCNN
from config import Config


def initialize_cluster_models_same_seed(k, seed=42):
    """khởi tạo k mô hình CNN cho DFCA"""
    torch.manual_seed(seed)
    cluster_models = {}

    for cluster_id in range(k):
        model = SimpleCNN().to(Config.DEVICE)
        cluster_models[cluster_id] = model.state_dict()

    return cluster_models

