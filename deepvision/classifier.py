"""Mode A: ResNet18 transfer-learning classifier (load, predict, build, save)."""
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Preprocessing used for BOTH training (validation) and inference.
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_MODEL_CACHE = {}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet18 with a new final layer. Only the final layer is trainable."""
    if num_classes < 2:
        raise ValueError("At least 2 classes are required.")
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # trainable
    return model


def save_classifier(model: nn.Module, classes: List[str], path) -> None:
    """Save weights + class names in the format load_classifier() expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "classes": list(classes),
        },
        path,
    )


def load_classifier(path) -> Tuple[nn.Module, List[str], torch.device]:
    """Load a trained classifier. Returns (model, classes, device)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Classifier file not found: {path}")
    key = (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    device = get_device()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint or "classes" not in checkpoint:
        raise ValueError("Invalid classifier file: expected keys 'state_dict' and 'classes'.")
    classes = [str(c) for c in checkpoint["classes"]]

    model = build_model(len(classes), pretrained=False)  # no download needed
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    _MODEL_CACHE.clear()  # keep only the latest model in memory
    _MODEL_CACHE[key] = (model, classes, device)
    return model, classes, device


@torch.no_grad()
def predict_classifier(model: nn.Module, classes: List[str], image: Image.Image,
                       device: torch.device) -> Dict:
    """Predict the class of a PIL image."""
    model.eval()
    tensor = eval_transform(image.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    best = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "label": classes[best],
        "confidence": float(probs[best]),
        "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
    }