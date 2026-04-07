import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


# import sys

# print(torch.cuda.is_available())


# sys.exit()


class LeafModel(nn.Module):
    def __init__(self, num_classes=2):
        super(LeafModel, self).__init__()
        self.flatten = nn.Flatten()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256x256 (image size) -> 128x128
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64 -> 32x32
        )

        # use adaptive pooling to avoid hardcoding the flattened dimension
        # avoids matmul shape mismatches
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_layer = nn.Dropout(p=0.3)
        self.classify = nn.Sequential(nn.Flatten(), nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.drop_layer(x)
        x = self.classify(x)
        return x


data_transforms = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),  # unsure about these values, prone to change
    ]
)

# temporary flag to switch which dataset is being used
# USE_DATASET: str = "OLID"
USE_DATASET: str = "plant_pathology"
PLANT_PATHOLOGY_DATASET_LOCATION: str = f"data/{USE_DATASET}"
if USE_DATASET == "plant_pathology":
    plants: list[str] = os.listdir(PLANT_PATHOLOGY_DATASET_LOCATION)
    # load each folder as a dataset separately so their labels of ["healthy", "diseased"] are used automatically
    datasets: list[ImageFolder] = []
    for plant in plants:
        dataset = ImageFolder(
            root=f"{PLANT_PATHOLOGY_DATASET_LOCATION}/{plant}",
            transform=data_transforms,
        )
        datasets.append(dataset)

    # combine all datasets into one so there's only healthy n diseased labels
    dataset = ConcatDataset(datasets)

else:
    dataset = ImageFolder(
        root="OLID",
        transform=data_transforms,
    )

# https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.random_split
training_dataset, validation_dataset = torch.utils.data.random_split(
    dataset, [0.8, 0.2]
)

training_dataloader = DataLoader(
    dataset=training_dataset,
    batch_size=24,
    shuffle=True,
    num_workers=8,
)

validation_dataloader = DataLoader(
    dataset=validation_dataset,
    batch_size=24,
    shuffle=False,
    num_workers=8,
)


def train_loop(dataloader, model, loss_fn, optimizer, epoch, device):
    model.train()
    print(f"\n--- Epoch {epoch + 1} ---")

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 2 == 0:
            print(f"  Batch {batch}: Loss = {loss.item():>7f}")
        if batch >= 500:
            break


def evaluate(dataloader, model, loss_fn, device):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            total += y.size(0)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # just evaluate one batch accuracy
            break

    print(f"  Evaluation: Accuracy = {100 * correct / total:>0.1f}%")


def main():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda"
    print(f"{device=}")

    # ConcatDataset does not have classes attribute like a singular ImageFolder
    # so temporarily default to two
    # num_classes = len(dataset.classes)
    num_classes = 2
    model = LeafModel(num_classes=num_classes)
    model.to(device)

    print(model)

    NUM_EPOCHS = 10
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        train_loop(training_dataloader, model, criterion, optimizer, epoch, device)
        evaluate(validation_dataloader, model, criterion, device)


if __name__ == "__main__":
    main()
