# Cấu hình tham số (Learning rate, Epsilons, Batch size, etc)
import torch
import os

class Config:
    # --- 1. Cấu hình Hệ thống & Docker ---
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = True
    
    # Tự động chọn GPU nếu có
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Đường dẫn lưu Model/Log (Tạo thư mục nếu chưa có)
    SAVE_DIR = "./results"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- 2. Cấu hình Mạng BCFL (Network) ---
    NUM_WORKERS = 5          # Số lượng Worker giả lập
    NUM_CLUSTERS = 2         # Số lượng cụm (Cluster) - K
    NUM_ROUNDS = 50          # Tổng số vòng Global Rounds chạy thử nghiệm

    # --- 3. Cấu hình Huấn luyện (Training Hyperparameters) ---
    DATASET_NAME = 'CIFAR10' # Hoặc 'MNIST'
    MODEL_NAME = 'simple_cnn'
    LOCAL_EPOCHS = 1         # Số epoch train tại mỗi worker mỗi vòng
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    
    # Secure Aggregation / Encryption
    ENABLE_ENCRYPTION = False # (Giai đoạn sau)

    # --- 5. Cấu hình Blockchain & Đồng thuận ---
    COMMITTEE_SIZE = 3       # Số lượng thành viên trong ủy ban
    CONSENSUS_THRESHOLD = 0.66 # Tỷ lệ phiếu bầu cần thiết (2/3)
    MIN_REPUTATION = 50.0    # Điểm uy tín tối thiểu để được làm Block Leader
    
    # --- 6. Cấu hình Cơ chế BALANCE (Lọc độc hại) ---
    BALANCE_THRESHOLD = 10.0 # Ngưỡng sai biệt tối đa để coi là node lành tính

    # --- 7. Cấu hình Smart Contract & Reputation (Có thể sửa lại cấu hình cho đúng) ---
    INITIAL_REPUTATION = 50.0  # Điểm khởi đầu
    MIN_REQ_REPUTATION = 20.0  # Điểm tối thiểu để được tham gia mạng (nếu thấp hơn sẽ bị ban)
    
    # Cơ chế thưởng phạt
    REWARD_SUCCESSFUL_BLOCK = 5.0    # Thưởng cho CH nếu Block được duyệt
    REWARD_COMMITTEE_VOTE = 1.0      # Thưởng cho thành viên Ủy ban vì đã bỏ phiếu
    PENALTY_REJECTED_BLOCK = -10.0   # Phạt nặng CH nếu gửi model rác
    PENALTY_MALICIOUS = -20.0        # Phạt nếu phát hiện tấn công
    w_COM = 40.0
    w_TRAIN = 40.0
    w_TIME = 20.0
    PENALTY = 80.0

    ACCURACY_BONUS_FACTOR = 0.5      # Hệ số nhân thưởng theo độ chính xác (VD: Acc 30% -> Bonus 15 điểm)
    DECAY_FACTOR = 0.99              # Hệ số giảm dần theo thời gian (để khuyến khích đóng góp liên tục)

    # BALANCE
    BALANCE_GAMMA = 0.3
    BALANCE_BETA = 0.5
    BALANCE_LAMBDA = 1.0
    BALANCE_Q0 = 0.01

    # LDP Config
    # Local Differential Privacy (LDP)
    ENABLE_LDP = True        # Bật/Tắt LDP
    LDP_EPSILON = 0.5        # Ngân sách riêng tư (Epsilon càng nhỏ càng bảo mật nhưng nhiễu càng lớn)
    LDP_CLIPPING_THRESHOLD = 1.5
    LDP_SIGMA = 0.5 # Noise multiplier càng cao thì epsilon càng nhỏ (bảo mật hơn)
    LDP_DELTA = 1e-5
    ACC_THRESHOLD = 30.0   # k

    # Cấu hình của Secret Sharing và VIEW-CHANGE
    T_THRESHOLD = 2
    VC_MAX_RETRIES = 2
