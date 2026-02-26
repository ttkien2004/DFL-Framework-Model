# app/scenarios/scenario_1_baseline.py
import numpy as np
from app.scenarios.base_scenario import BaseScenario
from torch.utils.data import Subset, DataLoader
# Import dataset loaders gốc (raw) để chia
from torchvision import datasets, transforms
from app.scenarios.base_scenario import BaseScenario

class ScenarioExperiment1(BaseScenario):
    """
    Kịch bản 1: Baseline Comparison (IID vs Non-IID)
    Tập trung vào việc chia dữ liệu Dirichlet.
    """
    _DIRICHLET_INDICES_CACHE = {}

    def setup_data(self, workers, dataset_name):
        dataset_name = dataset_name.lower()
        alpha = self.config.get('non_iid_alpha', None) # Lấy alpha từ config
        num_workers = len(workers)
        
        print(f"[Scenario 1] Setting up Data for {num_workers} workers. Dataset: {dataset_name}, Alpha: {alpha}", flush=True)

        # BƯỚC A: Lấy Targets (Nhãn) của toàn bộ dataset
        # Ta cần load dataset gốc 1 lần để biết phân phối nhãn
        targets = self._get_raw_targets(dataset_name)
        
        # BƯỚC B: Tính toán phân chia Index
        if alpha is None:
            # --- Chia IID (Chia đều) ---
            print("   -> Partitioning Mode: IID (Uniform)")
            total_size = len(targets)
            indices = np.arange(total_size)
            split_size = total_size // num_workers
            
            # Tạo dictionary {worker_id: [indices]}
            indices_map = {
                i: indices[i * split_size : (i + 1) * split_size].tolist() 
                for i in range(num_workers)
            }
        else:
            # --- Chia Non-IID (Dirichlet) ---
            # Gọi hàm static method bên dưới
            indices_map = self.partition_dirichlet(targets, num_workers, float(alpha))

        # BƯỚC C: Áp dụng Index cho từng Worker
        # Worker sẽ tự tạo DataLoader dựa trên list index này
        for w in workers:
            if w.id in indices_map:
                w.apply_new_indices(indices_map[w.id], dataset_name)
                
        print("[Scenario 1] Data setup completed.")

    def setup_network(self, workers):
        # Kịch bản 1 không quan tâm băng thông, để mặc định
        print("[Scenario 1] Network: Default Homogeneous")
        for w in workers:
            w.bandwidth = 100 # Mbps

    def setup_security(self, workers):
        # Kịch bản 1 không có tấn công
        print("[Scenario 1] Security: None")
        for w in workers:
            w.set_attack_profile("NONE")

    def _get_raw_targets(self, dataset_name):
        """Helper để tải targets nhanh mà không cần transform ảnh"""
        root = './data'
        try:
            if dataset_name == 'cifar10':
                # Download=True để đảm bảo data đã có
                ds = datasets.CIFAR10(root=root, train=True, download=True)
                return np.array(ds.targets)
            elif dataset_name == 'mnist':
                ds = datasets.MNIST(root=root, train=True, download=True)
                return np.array(ds.targets)
            elif dataset_name == 'gtsrb':
                ds = datasets.GTSRB(root=root, split='train', download=True)
                # GTSRB hơi đặc biệt, target nằm trong list samples
                return np.array([y for _, y in ds._samples])
            # Thêm các dataset khác nếu cần
        except Exception as e:
            print(f"Error loading raw targets for {dataset_name}: {e}")
            return []
        return []
    
    @staticmethod
    def partition_dirichlet(targets, num_workers, alpha, seed=42):
        """
        Chia index dữ liệu theo phân phối Dirichlet.
        
        Args:
            targets (list/array): Danh sách nhãn (labels) của toàn bộ dataset.
            num_workers (int): Số lượng worker.
            alpha (float): Tham số tập trung (Concentration parameter). 
                        Alpha càng nhỏ -> Càng Non-IID (mất cân bằng).
                        Alpha càng lớn -> Càng giống IID (cân bằng).
            seed (int): Random seed để đảm bảo tính nhất quán.
            
        Returns:
            dict: {worker_id: [list_of_indices]}
        """
    # Kiểm tra Cache trước
        cache_key = f"{len(targets)}_{num_workers}_{alpha}_{seed}"
        if cache_key in ScenarioExperiment1._DIRICHLET_INDICES_CACHE:
            return ScenarioExperiment1._DIRICHLET_INDICES_CACHE[cache_key]

        print(f"⚡ [Data Partition] Executing Dirichlet Partition (Alpha={alpha})...", flush=True)
        
        min_size = 0
        targets = np.array(targets)
        num_classes = len(np.unique(targets))
        N = len(targets)
        
        # Đảm bảo tính nhất quán
        np.random.seed(seed)

        # Dictionary chứa index cho từng worker
        net_dataidx_map = {i: [] for i in range(num_workers)}

        # Lặp qua từng class để chia (để đảm bảo class nào cũng được phân phối)
        for k in range(num_classes):
            # Lấy tất cả index của class k
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            
            # Tạo phân phối Dirichlet cho class k
            # proportions: mảng [p1, p2, ..., pn] tổng bằng 1
            proportions = np.random.dirichlet(np.repeat(alpha, num_workers))
            
            # Cân bằng lại proportions để tránh trường hợp một worker nhận quá ít hoặc quá nhiều
            
            proportions = np.array([p * (len(idx_j) < N / num_workers) for p, idx_j in zip(proportions, net_dataidx_map.values())])
            if proportions.sum() == 0:
                proportions = np.ones(num_workers)
            proportions = proportions / proportions.sum()
            
            # Tính điểm cắt (Split points) dựa trên proportions
            # Cumsum giúp chia mảng index thành các đoạn
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            # Thực hiện chia và gán vào map
            idx_batch = np.split(idx_k, proportions)
            for i in range(num_workers):
                net_dataidx_map[i] += idx_batch[i].tolist()

        # Lưu vào Cache
        ScenarioExperiment1._DIRICHLET_INDICES_CACHE[cache_key] = net_dataidx_map
        
        # Log thống kê để kiểm tra (Optional)
        # for i in range(num_workers):
        #     print(f"   -> Worker {i}: {len(net_dataidx_map[i])} samples")
            
        return net_dataidx_map