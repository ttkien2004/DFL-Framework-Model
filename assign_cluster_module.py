
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader


# -------------------------------------------------
# Simple CNN for CIFAR-10
# -------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# -------------------------------------------------
# AssignCluster for ONE node
# -------------------------------------------------
def assign_cluster(
    models: dict,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str = "cpu"
) -> int:
    """
    c(i) = argmin_j F_client(theta_{i,j}, D_i)
    """

    losses = {}

    with torch.no_grad():
        for j, model in models.items():
            model.eval()
            total_loss = 0.0
            total_samples = 0

            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                loss = loss_fn(preds, y)

                total_loss += loss.item() * X.size(0)
                total_samples += X.size(0)

            losses[j] = total_loss / total_samples

    return min(losses, key=losses.get), losses


# -------------------------------------------------
# TESTING MODULE: Simulate ONE node on CIFAR-10
# -------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- Load CIFAR-10 --------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # -------- Simulate LOCAL data of ONE node --------
    node_indices = list(range(0, 1000))  # node has 1000 samples
    local_dataset = Subset(dataset, node_indices)
    dataloader = DataLoader(local_dataset, batch_size=64, shuffle=False)

    # -------- k personalized models --------
    k = 30
    models = {
        j: SimpleCNN().to(device)
        for j in range(k)
    }

    # -------- Loss function --------
    loss_fn = nn.CrossEntropyLoss()

    # -------- Assign cluster --------
    cluster_id, losses = assign_cluster(
        models=models,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=device
    )

    print("Loss per cluster:")
    for j, l in losses.items():
        print(f"  Cluster {j}: {l:.4f}")

    print("\nAssigned cluster:", cluster_id)
