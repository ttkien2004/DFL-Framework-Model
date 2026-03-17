# app/scenarios/base_scenario.py
from abc import ABC, abstractmethod
import numpy as np

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
    
    import numpy as np

    def dirichlet_split_train_test(self, train_labels, test_labels, num_clients, alpha=0.5, num_classes=10):
        """
        Chia đồng bộ tập Train và Test cho các clients dựa trên phân phối Dirichlet.
        
        Args:
            train_labels (numpy array): Danh sách nhãn của tập Train gốc (VD: 50.000 nhãn)
            test_labels (numpy array): Danh sách nhãn của tập Test gốc (VD: 10.000 nhãn)
            num_clients (int): Số lượng workers
            alpha (float): Độ lệch Non-IID. Càng nhỏ càng lệch.
            num_classes (int): Số lượng nhãn (MNIST/CIFAR-10 là 10)
            
        Returns:
            client_train_indices (dict): Mapping client_id -> danh sách index tập Train
            client_test_indices (dict): Mapping client_id -> danh sách index tập Test
        """
        client_train_indices = {i: [] for i in range(num_clients)}
        client_test_indices = {i: [] for i in range(num_clients)}

        # Khởi tạo ma trận Dirichlet: Mỗi hàng là 1 class, mỗi cột là tỷ lệ của 1 client
        # Kích thước: (num_classes, num_clients)
        dirichlet_proportions = np.random.dirichlet(np.repeat(alpha, num_clients), num_classes)

        for c in range(num_classes):
            # 1. Lấy tất cả vị trí (index) của class 'c' trong tập Train và Test gốc
            idx_k_train = np.where(train_labels == c)[0]
            idx_k_test = np.where(test_labels == c)[0]

            # Trộn ngẫu nhiên (Shuffle) để tránh model học theo thứ tự cố định
            np.random.shuffle(idx_k_train)
            np.random.shuffle(idx_k_test)

            # 2. Lấy tỷ lệ chia của class 'c' cho các clients
            proportions = dirichlet_proportions[c]
            
            # Ép tỷ lệ để tổng đúng bằng 1.0 (Tránh lỗi làm tròn của numpy)
            proportions = proportions / proportions.sum()

            # 3. Chuyển tỷ lệ thành số lượng sample cụ thể cho Train và Test
            splits_train = (np.cumsum(proportions) * len(idx_k_train)).astype(int)[:-1]
            splits_test = (np.cumsum(proportions) * len(idx_k_test)).astype(int)[:-1]

            # 4. Cắt mảng index gốc ra làm 'num_clients' phần
            idx_train_splits = np.split(idx_k_train, splits_train)
            idx_test_splits = np.split(idx_k_test, splits_test)

            # 5. Phân phát cho từng client
            for i in range(num_clients):
                client_train_indices[i].extend(idx_train_splits[i].tolist())
                client_test_indices[i].extend(idx_test_splits[i].tolist())

        # Trộn lại index cục bộ của từng client một lần cuối
        for i in range(num_clients):
            np.random.shuffle(client_train_indices[i])
            np.random.shuffle(client_test_indices[i])

        return client_train_indices, client_test_indices