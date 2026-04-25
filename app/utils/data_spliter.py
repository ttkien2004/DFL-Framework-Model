import numpy as np
from torch.utils.data import Subset

def dirichlet_split_noniid(dataset, num_clients, alpha=0.5, num_classes=10):
    """
    Chia dataset thành các phần Non-IID dựa trên phân phối Dirichlet.
    Trả về danh sách các Subset.
    """
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    elif hasattr(dataset, 'y'):
        labels = np.array(dataset.y)
    else:
        # Dự phòng nếu không truy cập trực tiếp được
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
    num_samples = len(labels)
    client_indices = [[] for _ in range(num_clients)]
    
    # Sinh phân phối cho từng nhãn (class)
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Điều chỉnh để cân bằng số lượng mẫu (tránh một nút bị rỗng)
        proportions = np.array([p * (len(idx_j) < num_samples / num_clients) for p, idx_j in zip(proportions, client_indices)])
        if proportions.sum() > 0:
            proportions = proportions / proportions.sum()
        else:
            proportions = np.ones(num_clients) / num_clients 
            
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        
        idx_c_split = np.split(idx_c, proportions)
        
        for i in range(num_clients):
            client_indices[i].extend(idx_c_split[i].tolist())
            
    # Trả về danh sách các Dataset con
    return [Subset(dataset, indices) for indices in client_indices]