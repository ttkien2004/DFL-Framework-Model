# app/scenarios/base_scenario.py
from abc import ABC, abstractmethod

class BaseScenario(ABC):
    def __init__(self, workers, config):
        self.config = config
        self.workers = workers

    @abstractmethod
    def setup_data(self, workers, dataset_name):
        """Phân chia dữ liệu cho các Worker"""
        pass

    @abstractmethod
    def setup_network(self, workers):
        """Cấu hình băng thông, topology (quan trọng cho Kịch bản 3)"""
        pass

    @abstractmethod
    def setup_security(self, workers):
        """Cấu hình tấn công (quan trọng cho Kịch bản 4)"""
        pass

    def apply(self, workers):
        """Hàm template chạy toàn bộ quy trình setup"""
        print(f"Applying Scenario Configuration: {self.__class__.__name__}")
        self.setup_data(workers, self.config.get('dataset', 'cifar10'))
        self.setup_network(workers)
        self.setup_security(workers)