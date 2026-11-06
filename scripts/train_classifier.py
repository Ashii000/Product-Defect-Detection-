"""Train the Mode A classifier (ResNet18, only the final layer is trained).

Expected data layout (class folder names are your choice, but train and val must match):
    data/classifier/train/<class_name>/*.jpg
    data/classifier/val/<class_name>/*.jpg

Run from the project root:
    python scripts/train_classifier.py
Optional:
    python scripts/train_classifier.py --epochs 15 --batch-size 16 --lr 0.001
"""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so "import deepvision" works from any folder

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from deepvision.classifier import (
    IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE,
    build_model, eval_transform, get_device, save_classifier,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train the DeepVision AI classifier.")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "classifier")
    p.add_argument("--output", type=Path, default=ROOT / "models" / "classifier.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_datasets(data_dir: Path):
    train_dir, val_dir = data_dir / "train", data_dir / "val"
    for d in (train_dir, val_dir):
        if not d.is_dir():
            sys.exit(f"ERROR: folder not found: {d}\nCreate it and add one sub-folder per class.")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    try:
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds = datasets.ImageFolder(val_dir, transform=eval_transform)
    except (FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"ERROR: could not read the dataset ({exc})\n"
                 "Every class folder needs at least one image (jpg, png, bmp, webp...).")

    if len(train_ds.classes) < 2:
        sys.exit("ERROR: at least 2 class folders are required in train/ (e.g. normal, defective).")
    if train_ds.classes != val_ds.classes:
        sys.exit(f"ERROR: train and val classes differ.\n  train: {train_ds.classes}\n  val:   {val_ds.classes}")
    return train_ds, val_ds


def evaluate(model, loader, device, num_classes):
    """Return (accuracy, confusion matrix [true][pred])."""
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(dim=1).cpu()
            for t, p in zip(y.tolist(), pred.tolist()):
                cm[t][p] += 1
    return cm.trace() / max(cm.sum(), 1), cm


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    train_ds, val_ds = load_datasets(args.data_dir)
    classes = train_ds.classes
    counts = np.bincount(train_ds.targets, minlength=len(classes))
    print(f"Classes: {classes}")
    print(f"Train images: {len(train_ds)} {dict(zip(classes, counts.tolist()))} | Val images: {len(val_ds)}")
    if len(train_ds) < 20 or len(val_ds) < 4:
        print("WARNING: very few images - the pipeline will run, but results will not be reliable.")

    device = get_device()
    print(f"Device: {device}")
    model = build_model(len(classes), pretrained=True).to(device)

    # Class weights help when one class has many more images than the other.
    weights = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        # The frozen backbone stays in eval mode so BatchNorm statistics are not changed;
        # only the final linear layer (no BN/dropout) is learned.
        model.eval()
        total_loss, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            seen += x.size(0)
        val_acc, _ = evaluate(model, val_loader, device, len(classes))
        print(f"Epoch {epoch:2d}/{args.epochs} | train loss {total_loss / seen:.4f} | val accuracy {val_acc:.2%}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    save_classifier(model, classes, args.output)

    acc, cm = evaluate(model, val_loader, device, len(classes))
    print("\n=== Final validation results (best epoch) ===")
    print(f"Accuracy: {acc:.2%}")
    print("Confusion matrix (rows = true class, columns = predicted):")
    print("            " + " ".join(f"{c[:10]:>10}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c[:10]:>10}  " + " ".join(f"{v:>10}" for v in cm[i]))
    print("\nPer-class precision / recall / F1:")
    for i, c in enumerate(classes):
        tp = cm[i][i]; fp = cm[:, i].sum() - tp; fn = cm[i].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"  {c}: precision {prec:.2%} | recall {rec:.2%} | F1 {f1:.2%}")
    print(f"\nSaved model to: {args.output}")


if __name__ == "__main__":
    main()