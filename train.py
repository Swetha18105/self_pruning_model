import os
from urllib.error import URLError

os.environ["PYTORCH_DISABLE_TORCH_COMPILE"] = "1"

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 2
LAMBDAS = [1e-5, 1e-4, 1e-3]
SPARSITY_THRESHOLD = 1e-2


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return torch.matmul(x, pruned_weight.t()) + self.bias


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_sparsity(model):
    total_gates = 0
    pruned_gates = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total_gates += gates.numel()
            pruned_gates += (gates < SPARSITY_THRESHOLD).sum().item()

    return 100.0 * pruned_gates / total_gates


def get_all_gates(model):
    all_gates = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().view(-1).tolist()
            all_gates.extend(gates)

    return all_gates


def gate_l1_loss(model):
    loss = 0.0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            loss = loss + torch.sigmoid(module.gate_scores).abs().sum()

    return loss


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    return 100.0 * correct / total


def train_model(lmbda, trainloader, testloader):
    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for _ in range(EPOCHS):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            classification_loss = criterion(outputs, labels)
            sparsity_loss = gate_l1_loss(model)
            total_loss = classification_loss + lmbda * sparsity_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    accuracy = evaluate(model, testloader)
    sparsity = get_sparsity(model)
    return model, accuracy, sparsity


def load_cifar10(transform):
    dataset_roots = ["./data", "./data_fresh"]

    for root in dataset_roots:
        cifar_dir = os.path.join(root, "cifar-10-batches-py")
        should_download = not os.path.isdir(cifar_dir)

        try:
            trainset = torchvision.datasets.CIFAR10(
                root=root,
                train=True,
                download=should_download,
                transform=transform,
            )
            testset = torchvision.datasets.CIFAR10(
                root=root,
                train=False,
                download=should_download,
                transform=transform,
            )
            return trainset, testset
        except (RuntimeError, PermissionError, URLError):
            continue

    raise RuntimeError(
        "Unable to load CIFAR-10. Place a valid dataset in ./data or enable downloading."
    )


def main():
    os.makedirs("plots", exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    trainset, testset = load_cifar10(transform)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    results = []
    best_model = None
    best_accuracy = -1.0

    for lmbda in LAMBDAS:
        model, accuracy, sparsity = train_model(lmbda, trainloader, testloader)
        results.append((lmbda, accuracy, sparsity))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    gates = get_all_gates(best_model)
    plt.figure(figsize=(8, 5))
    plt.hist(gates, bins=50)
    plt.title("Gate Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plots/gates.png")
    plt.close()

    print("Lambda | Accuracy | Sparsity")
    for lmbda, accuracy, sparsity in results:
        print(f"{lmbda:.0e} | {accuracy:.2f}% | {sparsity:.2f}%")


if __name__ == "__main__":
    main()
