# app/scenarios/scenario_4_security.py
from app.scenarios.base_scenario import BaseScenario
from app.core.attacks import (
    AttackFactory,
    HonestStrategy
)
import random
import numpy as np
from torchvision import datasets, transforms

class ScenarioExperiment4(BaseScenario):
    _DIRICHLET_INDICES_CACHE = {}
    def __init__(self, workers, config):
        super().__init__(workers, config)

    def setup_security(self, workers):
        attack_type = self.config.get('attack_type', 'NONE').upper()
        ratio = self.config.get('malicious_ratio', 0.3) # 30% node độc hại
        distribution = self.config.get('distribution', 'RANDOM').upper()

        num_workers = len(workers)
        num_malicious = int(num_workers * ratio)
        
        for w in workers:
            w.attack_strategy = HonestStrategy()

        if attack_type == 'NONE' or num_malicious == 0:
            return
        
        malicious_indices = self._select_malicious_indices(num_workers, num_malicious, distribution)
        if attack_type == 'LABEL_FLIPPING':
            self._setup_label_flipping(workers, malicious_indices)
            
        elif attack_type in ['BACKDOOR', 'DBA']:
            # DBA là một dạng đặc biệt của Backdoor
            is_dba = (attack_type == 'DBA')
            self._setup_backdoor(workers, malicious_indices, is_dba)
        elif attack_type == 'GAUSSIAN_MODEL_POISONING':
            self._setup_model_poisoning(workers, malicious_indices)
        elif attack_type == 'MIA':
            self._setup_mia(workers, malicious_indices)
        elif attack_type in ['GIA', 'GRADIENT_INVERSION']:
            self._setup_gradient_inversion(workers, malicious_indices)
        else:
            print(f"Warning: Unknown attack type '{attack_type}'. No attack applied.")
    
    def setup_data(self, workers, dataset_name):
        """
        Hiện thực hàm setup_data (Bắt buộc).
        Quyết định phân phối dữ liệu (IID hoặc Non-IID).
        """
        dist_type = self.config.get('distribution', 'IID').upper()
        alpha = self.config.get('non_iid_alpha', 0.5)
        num_workers = len(workers)
        
        print(f"Setting up Data: {dist_type} (alpha={alpha})")
        
        # Nếu là Non-IID thì gọi hàm chia lại dữ liệu (giả sử bạn có hàm này)
        # Nếu không có logic đặc biệt thì chỉ cần pass hoặc log ra
        targets = self._get_raw_targets(dataset_name)
        if alpha is not None:
            # self._apply_non_iid_distribution(self.workers, alpha)
            indices_map = self.partition_dirichlet(targets, num_workers, float(alpha))
            for w in workers:
                if w.id in indices_map:
                    w.apply_new_indices(indices_map[w.id], dataset_name)
                    
            print("[Scenario 4] Data setup completed.")
        else:
            # Mặc định là IID (đã được load từ lúc init worker)
            total_size = len(targets)
            indices = np.arange(total_size)
            np.random.shuffle(indices)
            split_size = total_size // num_workers
            
            indices_map = {i: indices[i * split_size : (i + 1) * split_size].tolist() for i in range(num_workers)}
            for w in workers:
                if w.id in indices_map:
                    w.apply_new_indices(indices_map[w.id], dataset_name)
            print(" -> IID Partitioning Applied.")
    
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
    
    def setup_network(self):
        """
        Hiện thực hàm setup_network (Bắt buộc).
        Quyết định cấu trúc mạng (Topology).
        """
        print("Setting up Network Topology: Random P2P")
        
        # Ví dụ: Setup topology ngẫu nhiên cho các worker
        num_workers = len(self.workers)
        connectivity = 0.5 # Mỗi node kết nối với 50% node khác
        
        for w in self.workers:
            # Chọn ngẫu nhiên hàng xóm (trừ chính mình)
            candidates = [i for i in range(num_workers) if i != w.id]
            k = max(2, int(num_workers * connectivity))
            neighbors = random.sample(candidates, k)
            w.set_neighbors(neighbors)

    def _setup_label_flipping(self, workers, mal_indices):
        dataset_name = self.config.get('dataset', 'cifar10').lower()
        
        # Default cho CIFAR-10
        req_src = self.config.get('source_class')
        req_tar = self.config.get('target_class')

        if req_src is not None and req_tar is not None:
            # Nếu request có gửi, dùng luôn (3, 5)
            src, tar = int(req_src), int(req_tar)
        else:
            # 2. Nếu không gửi, mới dùng Default (Fallback)
            src, tar = 0, 2 
            if dataset_name == 'mnist':
                src, tar = 7, 9
            elif dataset_name == 'fashion_mnist':
                src, tar = 1, 3
        print(f"   -> Configuring Attack: {src} -> {tar}", flush=True)
        attack_config = {
            'source_class': src,
            'target_class': tar
        }
        strategy = AttackFactory.get_strategy("LABEL_FLIPPING", attack_config)

        for i in mal_indices:
            workers[i].attack_strategy = strategy

    def _setup_backdoor(self,workers, mal_indices, is_dba):
        target_class = self.config.get('target_class', self.config.get('target_label', 0))
        poison_rate = self.config.get('poison_rate', 0.2)
        scaling_factor = self.config.get('scaling_factor', 5.0) # Model Replacement scale

        attack_config = {
            "target_class": target_class,
            "poison_rate": poison_rate,
            "scaling_factor": scaling_factor,
            "dba_mode": is_dba,
        }

        attacker_cnt = 0 # Biến đếm để chia phần trigger cho DBA (nếu có)
        for idx in mal_indices:
            # Tạo strategy riêng cho từng worker (vì DBA cần attacker_index khác nhau)
            attack_config['attacker_index'] = attacker_cnt
            strategy = AttackFactory.get_strategy(
                "BACKDOOR",
                attack_config
            )
            
            workers[idx].attack_strategy = strategy
            attacker_cnt += 1

    def _setup_mia(self, workers, mal_indices):
        attack_config = {
            'dataset': self.config.get('dataset', 'cifar10'),
            'mia_subset_size': self.config.get('mia_subset_size', 1000),
            'tau': self.config.get('tau', None)
        }
        for idx in mal_indices:
            strategy = AttackFactory.get_strategy(
                'MIA',
                attack_config
            )
            workers[idx].attack_strategy = strategy

    def _select_malicious_indices(self, total_workers, num_malicious, distribution):
        """
        Hàm phụ trợ: Chọn index dựa trên phân bố (Random/Clustered).
        Theo tài liệu, vị trí phân bố ảnh hưởng lớn đến DBA.
        """
        all_indices = np.arange(total_workers)
        
        if distribution == 'CLUSTERED':
            # Chọn một điểm bắt đầu ngẫu nhiên và lấy N node liên tiếp
            # Giả lập việc các node độc hại nằm gần nhau trong topology
            start_node = random.randint(0, total_workers - num_malicious)
            return all_indices[start_node : start_node + num_malicious]
            
        elif distribution == 'FIXED':
            # Luôn chọn các node đầu tiên (Dễ debug)
            return all_indices[:num_malicious]
            
        else: # RANDOM (Mặc định)
            # Chọn ngẫu nhiên rải rác
            return np.random.choice(all_indices, num_malicious, replace=False)
        
    def _setup_model_poisoning(self, workers, malicious_indices):
        """
        Cấu hình tấn công bơm nhiễu Gauss.
        Tham số cần: noise_scale (sigma).
        """
        noise_scale = self.config.get('std', 2.0) # Mặc định nhiễu lớn (sigma=2.0)
        
        print(f"Strategy: Model Poisoning (Gauss Noise sigma={noise_scale})")
        
        # Tạo config để gửi cho Factory
        attack_config = {'noise_scale': noise_scale}
        
        from app.core.attacks import AttackFactory
        strategy = AttackFactory.get_strategy("MODEL_POISONING", attack_config)
        
        for idx in malicious_indices:
            workers[idx].attack_strategy = strategy
    
    def _setup_gradient_inversion(self, workers, malicious_indices):
        """
        Cấu hình tấn công Gradient Inversion.
        """
        print(f"Strategy: Gradient Inversion Attack (Reconstructing inputs from gradients)")
        
        # Config cho GIA
        gia_config = {
            'gia_iterations': self.config.get('gia_iterations', 300),  # Số vòng lặp tối ưu
            'gia_lr': self.config.get('gia_lr', 0.1)           # Learning rate cho L-BFGS
        }
        
        for idx in malicious_indices:
            # Gán chiến lược tấn công cho worker
            # Lưu ý: Worker này sẽ đóng vai kẻ tấn công, nghe lén và tái tạo ảnh            
            workers[idx].attack_strategy = AttackFactory.get_strategy('GIA', gia_config)
            print(f"   -> Worker {workers[idx].id} is set as Eavesdropper/Attacker")
    
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
        if cache_key in ScenarioExperiment4._DIRICHLET_INDICES_CACHE:
            return ScenarioExperiment4._DIRICHLET_INDICES_CACHE[cache_key]

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
        ScenarioExperiment4._DIRICHLET_INDICES_CACHE[cache_key] = net_dataidx_map
        
        # Log thống kê để kiểm tra (Optional)
        # for i in range(num_workers):
        #     print(f"   -> Worker {i}: {len(net_dataidx_map[i])} samples")
            
        return net_dataidx_map