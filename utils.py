import os
from collections import Counter, defaultdict

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import data_handler

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def configure_reproducibility(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def build_data_transforms(input_size):

    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(input_size, padding=4),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomVerticalFlip(),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_training_datasets(
    plant_pathology_root="data/plant_pathology",
    olid_root="data/OLID",
    transform=None,
):
    datasets = []

    plants = [
        plant
        for plant in os.listdir(plant_pathology_root)
        if os.path.isdir(os.path.join(plant_pathology_root, plant))
    ]

    for plant in plants:
        datasets.append(
            data_handler.BinaryLabelDataset(
                root=os.path.join(plant_pathology_root, plant),
                transform=transform,
            )
        )

    datasets.append(
        data_handler.TripleLabelDataset(
            root=olid_root,
            transform=transform,
        )
    )

    return datasets


def compute_class_weights(datasets, num_classes, device):
    all_labels = []
    for dataset in datasets:
        all_labels.extend(dataset.targets)

    counts = Counter(all_labels)
    total = sum(counts.values())
    weights = [total / counts[index] for index in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32).to(device)


def create_dataloaders(
    datasets,
    batch_size,
    num_workers,
    train_split=0.8,
    seed=55,
):
    dataset = ConcatDataset(datasets)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)

    training_dataset, validation_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return training_dataloader, validation_dataloader


def create_summary_writer(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)


def save_checkpoint(checkpoint_path, model, optimizer, epoch, loss, metadata=None):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    training_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    if metadata:
        training_data["metadata"] = metadata

    torch.save(training_data, checkpoint_path)


def load_checkpoint(checkpoint_path, device):
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


class OldLeafModel(nn.Module):
    def __init__(self):
        super(OldLeafModel, self).__init__()
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


def get_class_counts_from_concatdataset(concat_dataset):
    class_counts = defaultdict(int)
    class_names = set()
    for ds in concat_dataset.datasets:
        # ds is an ImageFolder
        for idx, class_name in enumerate(ds.classes):
            count = sum(1 for t in ds.targets if t == idx)
            class_counts[class_name] += count
            class_names.add(class_name)
    return sorted(class_names), dict(class_counts)

if __name__ == "__main__":
    transform = build_data_transforms(224)
    plant_pathology_root="data/plant_pathology"
    olid_root="data/OLID"

    olid = data_handler.TripleLabelDataset(
        root=olid_root,
        transform=transform,
    )

    datasets = []

    plants = [
        plant
        for plant in os.listdir(plant_pathology_root)
        if os.path.isdir(os.path.join(plant_pathology_root, plant))
    ]

    for plant in plants:
        datasets.append(
            data_handler.BinaryLabelDataset(
                root=os.path.join(plant_pathology_root, plant),
                transform=transform,
            )
        )

    plantpath = ConcatDataset(datasets)
    classes, counts = get_class_counts_from_concatdataset(plantpath)

    print(f"OLID classes: {olid.classes}")
    print(f"PlantPathology classes: {classes}, counts: {counts}")
