import os

from torchvision import datasets, transforms


# necessary to make sure PyTorch consistently maps 0 and 1 to plant_pathology
class BinaryLabelDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

        # force consistent mapping
        LABEL_MAP = {
            "healthy": 0,
            "diseased": 1,
        }

        new_samples = []
        for path, _ in self.samples:
            folder_name = os.path.basename(os.path.dirname(path)).lower()

            if folder_name in LABEL_MAP:
                new_samples.append((path, LABEL_MAP[folder_name]))
            else:
                raise ValueError(f"Unknown label: {folder_name}")

        self.samples = new_samples
        self.targets = [s[1] for s in new_samples]

        self.classes = ["healthy", "diseased", "pest-infested"]
        self.class_to_idx = {"healthy": 0, "diseased": 1, "pest-infested": 2}


# translate pre-labeled data into more general label categories
OLID_LABELS = {
    "healthy": 0,
    # diseases & nutritional deficiency labels from the OLID dataset
    "PM": 1,
    "K": 1,
    "K_Mg": 1,
    "N": 1,
    "N_K": 1,
    "N_Mg": 1,
    "DM": 1,
    "LS": 1,
    "LM": 1,
    # insects/pests
    "JAS": 2,
    "MIT": 2,
    "JAS_MIT": 2,
    "PLEI": 2,
    "PLEI_MIT": 2,
    "PLEI_IEM": 2,
    "MIT_EB": 2,
    "PC": 2,
    "EB": 2,
    "FB": 2,
}


class TripleLabelDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

        unlabeled = set()
        # override the samples with mapped labels
        new_samples = []
        for path, _ in self.samples:
            # extract the folder name representing the OLID label
            folder_name = os.path.basename(os.path.dirname(path))
            _, label = folder_name.split("__")  # labels are after double underscores

            if label in OLID_LABELS:
                target_label = OLID_LABELS[label]
                new_samples.append((path, target_label))
            else:
                # Optional: handle folders not in your map (like system files)
                # print(f"{label=}")
                unlabeled.add(label)

        print(unlabeled)

        self.samples = new_samples
        self.targets = [s[1] for s in new_samples]
        self.classes = ["healthy", "diseased", "pest-infested"]
        # self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.class_to_idx = {"healthy": 0, "diseased": 1, "pest-infested": 2}


data_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def retrieve_dataset(dataset_folder: str):
    """
    Returns dataset
    """

    DATASET_LOCATION: str = f"data/{dataset_folder}"

    dataset = TripleLabelDataset(root=DATASET_LOCATION, transform=data_transform)

    return dataset


if __name__ == "__main__":
    # dataset = retrieve_dataset("plant_pathology")
    dataset = retrieve_dataset("OLID")

    # dataset = datasets.ImageFolder(root="data/plant_pathology")

    print(f"Total images loaded: {len(dataset)}")
    print(f"Classes: {dataset.classes}")

    from collections import Counter

    label_counts = Counter(dataset.targets)
    print(
        f"Healthy: {label_counts[0]}, Diseased: {label_counts[1]}, Pest: {label_counts[2]}"
    )
