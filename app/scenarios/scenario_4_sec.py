# app/scenarios/scenario_4_security.py
from app.scenarios.base_scenario import BaseScenario
from app.core.attacks import (
    AttackFactory,
    HonestStrategy
)
import random
import numpy as np

class ScenarioExperiment4(BaseScenario):
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
        else:
            print(f"Warning: Unknown attack type '{attack_type}'. No attack applied.")
    
    def setup_data(self):
        """
        Hiện thực hàm setup_data (Bắt buộc).
        Quyết định phân phối dữ liệu (IID hoặc Non-IID).
        """
        dist_type = self.config.get('distribution', 'IID').upper()
        alpha = self.config.get('alpha', 0.5)
        
        print(f"Setting up Data: {dist_type} (alpha={alpha})")
        
        # Nếu là Non-IID thì gọi hàm chia lại dữ liệu (giả sử bạn có hàm này)
        # Nếu không có logic đặc biệt thì chỉ cần pass hoặc log ra
        if dist_type == 'NON_IID':
            # self._apply_non_iid_distribution(self.workers, alpha)
            pass 
        else:
            # Mặc định là IID (đã được load từ lúc init worker)
            pass
    

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
        scaling_factor = self.config.get('scaling_factor', 10.0) # Model Replacement scale

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
            'mia_subset_size': 1000,
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
        noise_scale = self.config.get('noise_scale', 2.0) # Mặc định nhiễu lớn (sigma=2.0)
        
        print(f"Strategy: Model Poisoning (Gauss Noise sigma={noise_scale})")
        
        # Tạo config để gửi cho Factory
        attack_config = {'noise_scale': noise_scale}
        
        from app.core.attacks import AttackFactory
        strategy = AttackFactory.get_strategy("MODEL_POISONING", attack_config)
        
        for idx in malicious_indices:
            workers[idx].attack_strategy = strategy