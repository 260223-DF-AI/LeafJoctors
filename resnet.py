import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models

# DATA_ROOT = "data"
# TRAIN_DIR = os.path.join(DATA_ROOT, "train")
# TEST_DIR = os.path.join(DATA_ROOT, "test")
# LOG_DIR = "runs/doggo_logs"
# MODEL_PATH = "doggo.pth"

if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: Dataset directory '{TRAIN_DIR}' not found.")
    # print("Please ensure the 'doggo' folder is extracted in the script's directory.")
    sys.exit(1)

data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=data_transforms)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=data_transforms)

print(f"Classes found: {train_dataset.classes}")
print(f"Total training images available: {len(train_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


class PreTrainedModel(nn.Module):
    def __init__(self):
        super(PreTrainedModel, self).__init__()

        # Load ResNet18 with weights pre-trained on ImageNet
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the final layer with one that matches our number of output classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.model(x)


class LeafModel(nn.Module):
    def __init__(self):
        super(LeafModel, self).__init__()
        self.flatten = nn.Flatten()

        self.features = nn.Sequential(
            # nn.Conv2d is a 2D convolution layer, slides a kernel over the input image
            # nn.MaxPool2d is a 2D max pooling layer, reduces the spatial dimensions of the input image
            # nn.ReLU is a rectified linear unit (ReLU) activation function, introduces non-linearity

            # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
            nn.Conv2d(3, 16, kernel_size=3, stride=1),  # Convolute 3x3 kernel, stepping by 1
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256x256 -> 128x128

            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128 -> 64x64
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 62 * 62, 128),  # 32 filters, 62x62 pixels, 128 neurons
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2)  # 128 neurons, 2 outputs
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classify(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer, epoch, best_loss, writer, device):
    print()

    print(f"\n--- Training Epoch {epoch + 1} ---")

    model.train()
    start_time = time.time()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", loss.item(), batch)

        # print(f"Batch {batch}: Loss = {loss.item():>7f}")

        if loss < best_loss:
            best_loss = loss

            print("New best model found! Loss: ", loss.item(), " Saving...")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, MODEL_PATH)

        if batch % 100 == 0:
            print(f"Batch {batch}: Loss = {loss.item():>7f}")

    end_time = time.time()
    print(f"Epoch {epoch + 1} completed: {batch + 1} batches processed")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    return model, best_loss


def evaluate(dataloader, model, loss_fn, writer, device):
    print()
    print("--- Eval Model ---")

    test_loss, correct, total = 0, 0, 0

    model.eval()

    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total += len(y)
            test_loss += loss_fn(pred, y).item()
            correct += int((pred.argmax(1) == y).type(torch.float).sum().item())
            if batch == 9: break

    writer.add_scalar("Loss/test", test_loss / total)

    print("Total Samples: ", total)
    print("Correct Predictions: ", correct)
    print(f"Test Loss: {test_loss / total:.4f}")
    print(f"Evaluation: Accuracy = {int(100 * correct / total)}%")


def main():
    device = torch.device("cuda")  # if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)

    print()
    print("--- Tensorboard Setup---")
    writer = SummaryWriter(LOG_DIR)

    print()
    print("--- Instantiate Model ---")
    #   model = DoggoModel()
    model = PreTrainedModel().to(device)
    best_loss = float('inf')

    print("Adding graph to tensorboard...")
    dummy_data = torch.randn(1, 3, 256, 256).to(device)
    writer.add_graph(model, dummy_data)

    NUM_EPOCHS = 1
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    print("--- Load Best Model ---")
    if os.path.exists(MODEL_PATH):
        best_model = torch.load(MODEL_PATH, weights_only=True)
        model.load_state_dict(best_model['model_state_dict'])
        optimizer.load_state_dict(best_model['optimizer_state_dict'])
        best_loss = best_model['loss']
        print("Loaded best model from ", MODEL_PATH)

    for epoch in range(NUM_EPOCHS):
        model, best_loss = train_loop(train_loader, model, criterion, optimizer, epoch, best_loss, writer, device)
        evaluate(test_loader, model, criterion, writer, device)


if __name__ == "__main__":
    main()
