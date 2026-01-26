from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import Config

# Hàm này cung cấp tập dữ liệu validation chung cho các Validator 
def get_global_val_loader():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if Config.DATASET_NAME == "CIFAR10":
        dataset = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )

    elif Config.DATASET_NAME == "MNIST":
        dataset = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )

    else:
        raise ValueError("Unsupported dataset")

    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
