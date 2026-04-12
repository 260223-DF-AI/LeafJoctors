import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchvision import models

import utils

# DEFAULT VALUES
SEED = 55
NUM_CLASSES = 3
INPUT_SIZE = 224
NUM_EPOCHS = 100
BATCH_SIZE = 24
NUM_WORKERS = 16
TRAIN_SPLIT = 0.8
CHECKPOINT_DIR = "checkpoints"
LOG_ROOT = "leaf_logs"
LOAD_MODEL = False

MODEL_CONFIGS = {
    "resnet34": {
        "builder": models.resnet34,
        "weights": models.ResNet34_Weights.DEFAULT,
    },
    "resnet50": {
        "builder": models.resnet50,
        "weights": models.ResNet50_Weights.DEFAULT,
    },
    "resnet101": {
        "builder": models.resnet101,
        "weights": models.ResNet101_Weights.DEFAULT,
    },
    "resnet152": {
        "builder": models.resnet152,
        "weights": models.ResNet152_Weights.DEFAULT,
    },
}

BASE_MODELS = MODEL_CONFIGS.keys()


class PreTrainedModel(nn.Module):
    def __init__(
        self,
        base_model,
        trainable_layers=None,
    ):
        super(PreTrainedModel, self).__init__()
        self.model = base_model

        # freeze base model layers
        for param in self.model.parameters():
            param.requires_grad = False

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 3)

        # unfreeze layers named in `trainable_layers`
        if trainable_layers:
            for name, param in self.model.named_parameters():
                if any(layer_name in name for layer_name in trainable_layers):
                    param.requires_grad = True

        # unfreeze final layer
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience  # how many epochs without improvement in validation loss to allow
        self.counter = 0  # num epochs w/o improvement
        self.best_loss = float("inf")  # best loss
        self.early_stop = False

    def __call__(self, loss):
        if loss < self.best_loss:  # if validation loss improved (got smaller)
            self.best_loss = loss  # update best validation loss
            self.counter = 0  # reset counter
            return self.early_stop, True
        else:  # if validation loss didn't improve
            self.counter += 1  # increment counter
            if self.counter >= self.patience:  # if counter exceeds patience
                self.early_stop = True  # early stop
        return self.early_stop, False


def train_loop(
    dataloader, model, loss_fn, optimizer, epoch, best_loss, writer, device, early_stop
):
    print(f"\n\n--- Training Epoch {epoch + 1} ---")

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

        print(f"Batch {batch}: Loss = {loss.item():>7f}")

    end_time = time.time()
    print(f"Epoch {epoch + 1} completed: {batch + 1} batches processed")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    return model, None, False


def evaluate(dataloader, model, loss_fn, writer, device, epoch):
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

    # tensorboard metrics
    writer.add_scalar("Loss/test", test_loss / total, epoch)
    writer.add_scalar("Accuracy/test", acc, epoch)
    writer.add_scalar("Precision/test", prec, epoch)
    writer.add_scalar("Recall/test", rec, epoch)
    writer.add_scalar("F1/test", f1_score, epoch)

    print("Total Samples: ", total)
    print("Correct Predictions: ", correct)
    print(f"Test Loss: {test_loss / total:.4f}")
    print(
        f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1_score:.4f}"
    )

    return test_loss / total, acc.item(), f1_score.item()


def get_model_config(model_name):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported backbone: {model_name}")

    return MODEL_CONFIGS[model_name]


def create_base_model(model_name):
    backbone_config = get_model_config(model_name)
    return backbone_config["builder"](weights=backbone_config["weights"])


def train_model(model_name, device):
    transform = utils.build_data_transforms(input_size=INPUT_SIZE)
    datasets = utils.load_training_datasets(transform=transform)
    weights = utils.compute_class_weights(datasets, NUM_CLASSES, device)
    training_dataloader, validation_dataloader = utils.create_dataloaders(
        datasets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_split=TRAIN_SPLIT,
        seed=SEED,
    )

    print(
        f"Train set size: {len(training_dataloader.dataset)}, "
        f"Val set size: {len(validation_dataloader.dataset)}"
    )

    writer = utils.create_summary_writer(f"{LOG_ROOT}/{model_name}")
    base_model = create_base_model(model_name)
    model = PreTrainedModel(base_model).to(device)
    best_loss = float("inf")
    best_acc = 0.0
    best_f1 = 0.0
    checkpoint_path = f"{CHECKPOINT_DIR}/{model_name}2.pth"

    print(f"\n--- Instantiate Model: {model_name} ---")

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    early_stop = EarlyStopping()

    if LOAD_MODEL:
        checkpoint = utils.load_checkpoint(checkpoint_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_loss = checkpoint["loss"]
        early_stop.best_loss = best_loss

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
            validation_dataloader, model, criterion, writer, device, epoch
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc
            best_f1 = val_f1
            utils.save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                metadata={
                    "model_name": model_name,
                    "num_classes": NUM_CLASSES,
                },
            )
            print(f"New best model found! Val Loss: {val_loss:.4f}, Saving...")

        if val_loss < early_stop.best_loss:
            early_stop.best_loss = val_loss
            early_stop.counter = 0
        else:
            early_stop.counter += 1
            if early_stop.counter >= early_stop.patience:
                print("Early stopping triggered")
                break

    writer.close()
    return {
        "model_name": model_name,
        "loss": best_loss,
        "accuracy": best_acc,
        "f1": best_f1,
        "checkpoint": checkpoint_path,
    }


def print_training_summary(results):
    print("\n--- Training Summary ---")
    for result in results:
        print(
            f"{result['model_name']}: loss={result['loss']:.4f}, "
            f"accuracy={result['accuracy']:.4f}, f1={result['f1']:.4f}, "
            f"checkpoint={result['checkpoint']}"
        )


def main():
    utils.configure_reproducibility(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)
    results = []
    for base_model in BASE_MODELS:
        results.append(train_model(base_model, device))

    print_training_summary(results)


if __name__ == "__main__":
    main()
