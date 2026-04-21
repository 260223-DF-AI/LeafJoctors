import io
import json
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

CLASS_NAMES = ["healthy", "diseased", "pest-infested"]
IMAGE_CONTENT_TYPES = {
    "application/octet-stream",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


class PreTrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet152(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

    def forward(self, x):
        return self.model(x)


def _build_transform():
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(model_dir, "model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = PreTrainedModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    if request_content_type not in IMAGE_CONTENT_TYPES:
        raise ValueError(f"Unsupported content type: {request_content_type}")

    image = Image.open(io.BytesIO(request_body)).convert("RGB")
    image_tensor = _build_transform()(image).unsqueeze(0)
    return image_tensor


def predict_fn(input_data, model):
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(input_data.to(device))
        probabilities = torch.softmax(logits, dim=1)[0].cpu()

    predicted_index = int(torch.argmax(probabilities).item())
    return {
        "classification": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index].item()),
        "probabilities": {
            class_name: float(probabilities[index].item())
            for index, class_name in enumerate(CLASS_NAMES)
        },
    }


def output_fn(prediction, accept):
    if accept not in (None, "*/*", "application/json"):
        raise ValueError(f"Unsupported accept type: {accept}")

    return json.dumps(prediction)
