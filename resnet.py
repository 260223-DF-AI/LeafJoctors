import os
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchvision import models, transforms

import data_handler

device = torch.device("cuda")
MODEL_PATH = "drfrond.pth"
LOG_DIR = "leaf_logs"

# Ensure deterministic results between model runs
torch.manual_seed(55)
torch.backends.cudnn.deterministic = True

data_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # resnet was trained on 224x224 images
        # improve generalizability with random image changes
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomVerticalFlip(),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],  # in line with ResNet's specified `ImageNet` stats
        ),
        # transforms.RandomErasing(),
    ]
)

datasets = []

# --- Load plant_pathology ---
# plants = os.listdir("data/plant_pathology")
plants = [
    p
    for p in os.listdir("data/plant_pathology")
    if os.path.isdir(f"data/plant_pathology/{p}")
]

for plant in plants:
    d = data_handler.BinaryLabelDataset(
        root=f"data/plant_pathology/{plant}",
        transform=data_transforms,
    )
    datasets.append(d)

# --- Load OLID ---
olid_dataset = data_handler.TripleLabelDataset(
    root="data/OLID",
    transform=data_transforms,
)
datasets.append(olid_dataset)

# figure out weights of each label based on how many images of each we have
all_labels = []
for d in datasets:
    all_labels.extend(d.targets)

counts = Counter(all_labels)
total = sum(counts.values())

weights = [total / counts[i] for i in range(3)]
weights = torch.tensor(weights).to(device)

# --- Combine everything ---
dataset = ConcatDataset(datasets)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

training_dataset, validation_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

print(
    f"Train set size: {len(training_dataset)}, Val set size: {len(validation_dataset)}"
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

# print(f"Classes found: {training_dataset.classes}")
# print(f"Total training images available: {len(training_dataset)}")


class PreTrainedModel(nn.Module):
    def __init__(self):
        super(PreTrainedModel, self).__init__()

        # Load ResNet18 with weights pre-trained on ImageNet
        # self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Utilize metrics like F1 score to determine best model for our data
        # self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        # self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # try out unfreezing fourth layer
        # for name, param in self.model.named_parameters():
        #     if "layer4" in name:
        #         param.requires_grad = True

        # Replace the final layer with one that matches our number of output classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 3)

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
            nn.Conv2d(
                3, 16, kernel_size=3, stride=1
            ),  # Convolute 3x3 kernel, stepping by 1
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
            nn.Linear(128, 3),  # 128 neurons, 3 outputs
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classify(x)
        return x


class EarlyStopping:
    def __init__(self, patience=100):
        self.patience = patience  # how many batches without improvement to allow
        self.counter = 0  # num batches w/o improvement
        self.best_loss = float("inf")  # best loss
        self.early_stop = False

    def __call__(self, loss):
        if loss < self.best_loss:  # if loss improved (got smaller)
            self.best_loss = loss  # update best loss
            self.counter = 0  # reset counter
            return self.early_stop, True
        else:  # if loss didn't improve
            self.counter += 1  # increment counter
            if self.counter >= self.patience:  # if counter exceeds patience
                self.early_stop = True  # early stop
        return self.early_stop, False


def train_loop(
    dataloader, model, loss_fn, optimizer, epoch, best_loss, writer, device, early_stop
):
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

        print(f"Batch {batch}: Loss = {loss.item():>7f}")

        # Stop when the model's loss is not improving over many batches
        # Early stopping moved to main after validation

    end_time = time.time()
    print(f"Epoch {epoch + 1} completed: {batch + 1} batches processed")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    return model, None, False


def evaluate(dataloader, model, loss_fn, writer, device):
    print()
    print("--- Eval Model ---")

    test_loss, correct, total = 0, 0, 0

    # Initialize metrics
    accuracy = Accuracy(task="multiclass", num_classes=3).to(device)
    precision = Precision(task="multiclass", num_classes=3, average="macro").to(device)
    recall = Recall(task="multiclass", num_classes=3, average="macro").to(device)
    f1 = F1Score(task="multiclass", num_classes=3, average="macro").to(device)

    all_preds = []
    all_targets = []

    model.eval()

    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total += len(y)
            test_loss += loss_fn(pred, y).item()
            correct += int((pred.argmax(1) == y).type(torch.float).sum().item())

            all_preds.append(pred)
            all_targets.append(y)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    acc = accuracy(preds, targets)
    prec = precision(preds, targets)
    rec = recall(preds, targets)
    f1_score = f1(preds, targets)

    writer.add_scalar("Loss/test", test_loss / total)
    writer.add_scalar("Accuracy/test", acc)
    writer.add_scalar("Precision/test", prec)
    writer.add_scalar("Recall/test", rec)
    writer.add_scalar("F1/test", f1_score)

    print("Total Samples: ", total)
    print("Correct Predictions: ", correct)
    print(f"Test Loss: {test_loss / total:.4f}")
    print(
        f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1_score:.4f}"
    )

    return test_loss / total, acc.item(), f1_score.item()


def main():
    device = torch.device("cuda")  # if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)

    print()
    print("--- Tensorboard Setup---")
    writer = SummaryWriter(LOG_DIR)

    print()
    print("--- Instantiate Model ---")
    # model = LeafModel()
    # model.to(device)
    model = PreTrainedModel().to(device)
    best_loss = float("inf")

    print("Adding graph to tensorboard...")
    dummy_data = torch.randn(1, 3, 256, 256).to(device)
    writer.add_graph(model, dummy_data)

    NUM_EPOCHS = 10
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9
    )
    # optimizer = optim.Adam(
    #     filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001
    # )
    criterion = nn.CrossEntropyLoss(weight=weights)

    early_stop = EarlyStopping()

    LOAD_MODEL: bool = False
    print("--- Load Best Model ---")
    if os.path.exists(MODEL_PATH) and LOAD_MODEL:
        best_model = torch.load(MODEL_PATH, weights_only=True)
        model.load_state_dict(best_model["model_state_dict"])
        optimizer.load_state_dict(best_model["optimizer_state_dict"])
        best_loss = best_model["loss"]
        early_stop.best_loss = best_loss
        print("Loaded best model from ", MODEL_PATH, f" with loss of {best_loss}")

    for epoch in range(NUM_EPOCHS):
        model, _, _ = train_loop(
            training_dataloader,
            model,
            criterion,
            optimizer,
            epoch,
            best_loss,
            writer,
            device,
            early_stop,
        )
        val_loss, val_acc, val_f1 = evaluate(
            validation_dataloader, model, criterion, writer, device
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                },
                MODEL_PATH,
            )
            print(f"New best model found! Val Loss: {val_loss:.4f}, Saving...")

        # Early stopping on validation loss
        if val_loss < early_stop.best_loss:
            early_stop.best_loss = val_loss
            early_stop.counter = 0
        else:
            early_stop.counter += 1
            if early_stop.counter >= early_stop.patience:
                print("Early stopping triggered")
                break


if __name__ == "__main__":
    main()
