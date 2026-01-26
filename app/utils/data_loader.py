import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# =========================================================================
# 1. HÀM LÕI (CORE FUNCTION) - Private
# Nhiệm vụ duy nhất: Trả về Dataset Object với Transform chuẩn.
# =========================================================================
def _get_dataset_core(dataset_name, train=True):
    """
    Hàm nội bộ: Cấu hình Transform và tải Dataset.
    Dùng chung cho cả Worker training và Attacker MIA.
    """
    dataset_name = dataset_name.lower()
    root = './data'
    
    # --- Cấu hình Transform (Duy nhất 1 nơi để sửa) ---
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_cls = datasets.CIFAR10
        num_classes = 10
        input_channels = 3

    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset_cls = datasets.MNIST
        num_classes = 10
        input_channels = 3

    elif dataset_name == 'gtsrb':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
        ])
        # Xử lý đặc biệt cho GTSRB
        split = 'train' if train else 'test'
        try:
            ds = datasets.GTSRB(root=root, split=split, download=True, transform=transform)
            return ds, 43, 3
        except:
            return None, 43, 3

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # --- Khởi tạo Dataset ---
    try:
        ds = dataset_cls(root=root, train=train, download=True, transform=transform)
        return ds, num_classes, input_channels
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None, 0, 0


# =========================================================================
# 2. CÁC HÀM PUBLIC (Được gọi từ bên ngoài)
# =========================================================================

def get_raw_dataset(dataset_name, train=True):
    """
    Dùng cho: MIA Attack (lấy tập Member/Non-member) hoặc lấy targets để chia Dirichlet.
    Trả về: Dataset Object (chưa chia batch).
    """
    dataset, _, _ = _get_dataset_core(dataset_name, train=train)
    return dataset

def get_dataloader_from_indices(dataset_name, indices, batch_size=32, train=True):
    """
    Dùng cho: Worker Training (Non-IID).
    Trả về: DataLoader từ danh sách indices cụ thể.
    
    Args:
        train (bool): Mặc định True (Worker train). 
                      Nếu cần test loader cho worker thì truyền False.
    """
    # Gọi hàm lõi để lấy dataset
    dataset, num_classes, input_channels = _get_dataset_core(dataset_name, train=train)
    
    if dataset is None:
        raise ValueError("Dataset not found")

    # Tạo subset từ indices
    subset = Subset(dataset, indices)
    
    # Tạo Loader
    # drop_last=True để tránh lỗi BatchNorm với batch nhỏ lẻ
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return loader, num_classes, input_channels

def get_dataloader(dataset_name, node_id, num_workers, batch_size=32):
    """
    Dùng cho: Kịch bản IID đơn giản (Tự chia index).
    """
    # 1. Lấy toàn bộ dataset train
    dataset, num_classes, input_channels = _get_dataset_core(dataset_name, train=True)
    
    # 2. Tự tính toán indices (Chia đều IID)
    total_size = len(dataset)
    indices = list(range(total_size))
    split_size = total_size // num_workers
    node_indices = indices[node_id * split_size : (node_id + 1) * split_size]
    
    # 3. Tạo Loader (Tái sử dụng logic của hàm trên hoặc viết thẳng)
    subset = Subset(dataset, node_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return loader, num_classes, input_channels

def get_global_test_loader(dataset_name, batch_size=32):
    """
    Hàm mới: Lấy toàn bộ tập Test (Test Set) để đánh giá Global Model.
    Không chia nhỏ, không xáo trộn (shuffle=False).
    """
    # Gọi hàm core với train=False để lấy tập Test
    test_dataset,_,_ = _get_dataset_core(dataset_name, train=False)
    
    # Tạo DataLoader chuẩn
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)