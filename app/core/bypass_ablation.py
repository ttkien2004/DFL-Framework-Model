"""
Bypass Ablation Module - Loại bỏ các tính năng để so sánh hiệu suất
Bao gồm 4 scenario bypass:
1. Bypass Clustering: Gán tất cả nodes vào 1 cụm duy nhất
2. Bypass Privacy (LDP/SSS): Không cộng nhiễu, không chia secret shares
3. Bypass Byzantine (BALANCE): Dùng FedAvg thay vì BALANCE
4. Bypass Blockchain: Lưu model vào RAM thay vì blockchain
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import copy


class BypassConfig:
    """Cấu hình các bypass modes"""
    
    BYPASS_NONE = 0            # Chạy bình thường (full features)
    BYPASS_CLUSTERING = 1       # Tất cả vào 1 cụm
    BYPASS_PRIVACY = 2          # Không LDP/SSS
    BYPASS_BYZANTINE = 4        # Không robust aggregation
    BYPASS_BLOCKCHAIN = 8       # Không blockchain consensus
    BYPASS_ALL = 15             # Tất cả bypass (trở thành DFL truyền thống)
    
    BYPASS_NAMES = {
        0: "Full_Features",
        1: "No_Clustering",
        2: "No_Privacy",
        4: "No_Byzantine",
        8: "No_Blockchain",
        15: "Traditional_DFL"
    }
    
    @staticmethod
    def get_name(bypass_mode: int) -> str:
        """Lấy tên scenario dựa trên bypass mode"""
        return BypassConfig.BYPASS_NAMES.get(bypass_mode, "Unknown")
    
    @staticmethod
    def is_clustering_enabled(bypass_mode: int) -> bool:
        return not (bypass_mode & BypassConfig.BYPASS_CLUSTERING)
    
    @staticmethod
    def is_privacy_enabled(bypass_mode: int) -> bool:
        return not (bypass_mode & BypassConfig.BYPASS_PRIVACY)
    
    @staticmethod
    def is_byzantine_enabled(bypass_mode: int) -> bool:
        return not (bypass_mode & BypassConfig.BYPASS_BYZANTINE)
    
    @staticmethod
    def is_blockchain_enabled(bypass_mode: int) -> bool:
        return not (bypass_mode & BypassConfig.BYPASS_BLOCKCHAIN)


class BypassClustering:
    """
    Bypass Clustering: Gán tất cả nodes vào 1 cụm duy nhất
    """
    
    @staticmethod
    def cluster_all_nodes(workers, cluster_id=0) -> Dict[int, List]:
        """
        Gom tất cả workers vào 1 cụm duy nhất
        
        Returns:
            Dict[int, List]: {cluster_id: [worker1, worker2, ...]}
        """
        clusters = {cluster_id: workers}
        
        # Gán cluster_id cho tất cả workers
        for w in workers:
            w.cluster_id = cluster_id
        
        # Chọn worker đầu tiên làm Head
        if workers:
            workers[0].is_head = True
            workers[0].cluster_head_id = workers[0].id
            
            # Gán cluster_head_id cho các worker khác
            for w in workers[1:]:
                w.cluster_head_id = workers[0].id
                w.members = workers
        
        print(f"[Bypass Clustering] All {len(workers)} workers assigned to Cluster {cluster_id}")
        return clusters


class BypassPrivacy:
    """
    Bypass Privacy: Không cộng nhiễu LDP/SSS, gửi gradient "sạch"
    """
    
    @staticmethod
    def skip_ldp_noise(gradient: Dict) -> Dict:
        """
        Bỏ qua thêm nhiễu Laplace/Gaussian
        Trả về gradient nguyên gốc
        """
        # Đơn giản - return gradient như cũ, không thêm nhiễu
        return gradient
    
    @staticmethod
    def skip_secret_sharing(model_state: Dict) -> Dict:
        """
        Bỏ qua chia nhỏ Secret Shares
        Trả về model state nguyên gốc, không mã hóa
        """
        # Không chia nhỏ, return nguyên model
        return model_state
    
    @staticmethod
    def flatten_model(model_state: Dict) -> torch.Tensor:
        """
        Flatten model state thành vector 1D (để tính toán dễ hơn)
        """
        flat_params = []
        for param in model_state.values():
            if isinstance(param, torch.Tensor):
                flat_params.append(param.view(-1).cpu())
            else:
                flat_params.append(torch.tensor(param).view(-1).cpu())
        
        return torch.cat(flat_params) if flat_params else torch.tensor([])
    
    @staticmethod
    def unflatten_model(flat_tensor: torch.Tensor, model_template: Dict) -> Dict:
        """
        Unflatten vector 1D trở lại model state
        """
        result = {}
        offset = 0
        
        for key, param in model_template.items():
            if isinstance(param, torch.Tensor):
                size = param.numel()
            else:
                size = len(param) if hasattr(param, '__len__') else 1
            
            result[key] = flat_tensor[offset:offset+size].reshape_as(param)
            offset += size
        
        return result


class BypassByzantine:
    """
    Bypass Byzantine Robustness: Dùng FedAvg thay vì BALANCE
    """
    
    @staticmethod
    def fedavg_aggregation(local_updates: List[Dict]) -> Dict:
        """
        Aggregation đơn giản - FedAvg (Trung bình cộng)
        
        Args:
            local_updates: List of model states từ các workers
            
        Returns:
            Aggregated model state (dictionary)
        """
        if not local_updates:
            return {}
        
        # Filter out non-dict items and ensure they're valid
        valid_updates = []
        for update in local_updates:
            if isinstance(update, dict):
                valid_updates.append(update)
        
        if not valid_updates:
            return {}
        
        # Khởi tạo accumulated weights
        accumulated = None
        
        for update in valid_updates:
            if accumulated is None:
                # Copy deep lần đầu
                accumulated = {}
                for key, val in update.items():
                    if isinstance(val, torch.Tensor):
                        accumulated[key] = val.clone().float()
                    else:
                        accumulated[key] = val
            else:
                # Add các updates sau
                for key in update:
                    if key in accumulated:
                        if isinstance(accumulated[key], torch.Tensor) and isinstance(update[key], torch.Tensor):
                            accumulated[key] = accumulated[key] + update[key].float()
        
        # Trung bình cộng
        if accumulated and valid_updates:
            n = len(valid_updates)
            for key in accumulated:
                if isinstance(accumulated[key], torch.Tensor):
                    accumulated[key] = accumulated[key] / n
        
        print(f"[Bypass Byzantine] FedAvg aggregated {len(valid_updates)} updates")
        return accumulated
    
    @staticmethod
    def simple_median_aggregation(local_updates: List[Dict]) -> Dict:
        """
        Aggregation sử dụng Median (đơn giản hơn BALANCE)
        Thích hợp cho adversarial robustness
        """
        if not local_updates:
            return {}
        
        result = {}
        
        # Lấy keys từ update đầu tiên
        template = local_updates[0]
        
        for key in template:
            values_list = []
            
            for update in local_updates:
                param = update.get(key)
                if isinstance(param, torch.Tensor):
                    values_list.append(param.cpu())
                else:
                    values_list.append(torch.tensor(param).cpu())
            
            if values_list:
                # Stack và tính median theo axis 0
                stacked = torch.stack(values_list, dim=0)
                median_val = torch.median(stacked, dim=0)[0]
                result[key] = median_val
        
        print(f"[Bypass Byzantine] Median aggregated {len(local_updates)} updates")
        return result


class BypassBlockchain:
    """
    Bypass Blockchain: Lưu model vào RAM (in-memory storage)
    Không cần consensus, chỉ aggregate và update trực tiếp
    """
    
    def __init__(self):
        """
        In-memory model registry
        """
        self.global_models = {}  # {cluster_id: model_state}
        self.history = []        # Lưu lịch sử các model versions
        self.metadata = {}       # Metadata của mỗi model
    
    def store_model(self, cluster_id: int, model_state: Dict, metadata: Dict = None):
        """
        Lưu model vào memory (bypass blockchain)
        
        Args:
            cluster_id: Cluster ID
            model_state: Model state dictionary
            metadata: Metadata (accuracy, loss, etc.)
        """
        # Lưu model
        self.global_models[cluster_id] = copy.deepcopy(model_state)
        
        # Lưu metadata
        if metadata:
            self.metadata[cluster_id] = metadata
        
        # Ghi vào history
        self.history.append({
            "cluster_id": cluster_id,
            "model": copy.deepcopy(model_state),
            "metadata": metadata or {},
            "timestamp": len(self.history)
        })
        
        print(f"[Bypass Blockchain] Stored model for Cluster {cluster_id} (Version {len(self.history)})")
    
    def get_model(self, cluster_id: int) -> Dict:
        """
        Lấy model từ memory
        """
        return self.global_models.get(cluster_id, {})
    
    def get_all_models(self) -> Dict[int, Dict]:
        """
        Lấy tất cả models
        """
        return copy.deepcopy(self.global_models)
    
    def get_history(self, cluster_id: int = None) -> List:
        """
        Lấy lịch sử versions
        """
        if cluster_id is None:
            return self.history
        else:
            return [h for h in self.history if h["cluster_id"] == cluster_id]


class BypassExecutor:
    """
    Orchestrator để chạy các bypass modes
    """
    
    def __init__(self, bypass_mode: int = 0):
        """
        Args:
            bypass_mode: Bitwise combination của BypassConfig flags
        """
        self.bypass_mode = bypass_mode
        self.name = BypassConfig.get_name(bypass_mode)
        self.memory_storage = BypassBlockchain()
        
        print(f"[BypassExecutor] Initialized with mode: {self.name} (flags: {bypass_mode})")
    
    def should_cluster(self) -> bool:
        return BypassConfig.is_clustering_enabled(self.bypass_mode)
    
    def should_apply_privacy(self) -> bool:
        return BypassConfig.is_privacy_enabled(self.bypass_mode)
    
    def should_use_byzantine(self) -> bool:
        return BypassConfig.is_byzantine_enabled(self.bypass_mode)
    
    def should_use_blockchain(self) -> bool:
        return BypassConfig.is_blockchain_enabled(self.bypass_mode)
    
    def execute_clustering_phase(self, workers) -> Dict[int, List]:
        """
        Atur clustering phase dengan bypass logic
        """
        if self.should_cluster():
            # Normal clustering (không bypass)
            return None  # Return None để engine dùng clustering bình thường
        else:
            # Bypass clustering - tất cả vào 1 cụm
            return BypassClustering.cluster_all_nodes(workers, cluster_id=0)
    
    def execute_aggregation(self, local_updates: List[Dict]) -> Dict:
        """
        Execute aggregation với bypass logic
        """
        if self.should_use_byzantine():
            # Dùng BALANCE hoặc robust aggregation bình thường
            return None  # Return None để engine dùng aggregation bình thường
        else:
            # Bypass Byzantine - dùng FedAvg
            return BypassByzantine.fedavg_aggregation(local_updates)
    
    def execute_storage(self, cluster_id: int, model_state: Dict, metadata: Dict = None) -> str:
        """
        Execute storage với bypass logic
        
        Returns:
            Path hoặc ID của model (tùy backend)
        """
        if self.should_use_blockchain():
            # Dùng blockchain storage bình thường
            return None  # Return None để engine dùng blockchain
        else:
            # Bypass blockchain - lưu vào RAM
            self.memory_storage.store_model(cluster_id, model_state, metadata)
            return f"memory://cluster_{cluster_id}_v{len(self.memory_storage.history)}"
    
    def get_report(self) -> Dict:
        """
        Lấy report về bypass status
        """
        return {
            "mode": self.name,
            "bypass_flags": self.bypass_mode,
            "clustering_enabled": self.should_cluster(),
            "privacy_enabled": self.should_apply_privacy(),
            "byzantine_enabled": self.should_use_byzantine(),
            "blockchain_enabled": self.should_use_blockchain(),
            "memory_storage_size": len(self.memory_storage.get_all_models()) if not self.should_use_blockchain() else 0
        }
