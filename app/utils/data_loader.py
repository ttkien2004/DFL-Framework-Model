# app/utils/data_loader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataloader(dataset_name, node_id, num_workers, batch_size=32):
    """
    Factory method để lấy DataLoader dựa trên tên dataset
    """
    dataset_name = dataset_name.lower()
    root = './data'
    
    # 1. Cấu hình Transform & Download Dataset
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        num_classes = 10
        input_channels = 3

    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((32, 32)), # Resize lên 32x32 để khớp với mô hình CNN hiện tại
            transforms.Grayscale(num_output_channels=3), # Giả lập 3 kênh màu để khớp input layer
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        num_classes = 10
        input_channels = 3

    elif dataset_name == 'gtsrb':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
        ])
        # GTSRB cần download thủ công hoặc dùng link có sẵn, ở đây dùng hàm có sẵn nếu torch hỗ trợ version mới
        # Nếu không, bạn cần code tải riêng. Giả sử environment đã có sẵn hoặc torch hỗ trợ:
        train_dataset = datasets.GTSRB(root=root, split='train', download=True, transform=transform)
        num_classes = 43
        input_channels = 3
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 2. Chia dữ liệu cho Worker (IID)
    total_size = len(train_dataset)
    indices = list(range(total_size))
    split_size = total_size // num_workers
    
    # Lấy phần dữ liệu riêng cho worker này
    node_indices = indices[node_id * split_size : (node_id + 1) * split_size]
    local_dataset = Subset(train_dataset, node_indices)
    
    loader = DataLoader(local_dataset, batch_size=batch_size, shuffle=True)
    
    return loader, num_classes, input_channels