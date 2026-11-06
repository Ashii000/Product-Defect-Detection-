"""Mode B: reference-based anomaly detection with ResNet18 patch features.

Normal reference images -> patch features (memory bank). A test patch's anomaly
value is its distance to the nearest normal patch. A lightweight PatchCore-style
baseline (not the full PatchCore algorithm).
"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

IMAGE_SIZE = 256
MAX_BANK_PATCHES = 60000      # cap on stored normal patches (memory/speed)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_BACKBONE = {}
_DETECTOR_CACHE = {}


def list_images(folder) -> List[Path]:
    """Sorted image files in a folder (empty list if the folder does not exist)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_backbone() -> nn.Module:
    if "model" not in _BACKBONE:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.eval().to(_device())
        for p in model.parameters():
            p.requires_grad = False
        _BACKBONE["model"] = model
    return _BACKBONE["model"]


@torch.no_grad()
def _extract_patches(image: Image.Image) -> torch.Tensor:
    """Return patch features of shape (H*W, C) from ResNet18 layer2 + layer3."""
    model = _get_backbone()
    x = _transform(image.convert("RGB")).unsqueeze(0).to(_device())
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    x = model.layer1(x)
    f2 = model.layer2(x)      # (1,128,32,32)
    f3 = model.layer3(f2)     # (1,256,16,16)
    f2 = F.avg_pool2d(f2, 3, 1, 1)
    f3 = F.avg_pool2d(f3, 3, 1, 1)
    f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    feats = torch.cat([f2, f3], dim=1)                       # (1,384,32,32)
    return feats.squeeze(0).flatten(1).t().contiguous()      # (1024,384)


def _min_distance(queries: torch.Tensor, bank: torch.Tensor, bank_ids: torch.Tensor,
                  exclude_id: int = -1, chunk: int = 20000) -> torch.Tensor:
    """Distance from each query patch to its nearest bank patch."""
    best = torch.full((queries.shape[0],), float("inf"), device=queries.device)
    for start in range(0, bank.shape[0], chunk):
        b = bank[start:start + chunk]
        d = torch.cdist(queries, b)
        if exclude_id >= 0:
            d[:, bank_ids[start:start + chunk] == exclude_id] = float("inf")
        best = torch.minimum(best, d.min(dim=1).values)
    return best


def _gaussian_blur(x: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    radius = int(3 * sigma)
    coords = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), k.view(1, 1, -1, 1))
    return x


class AnomalyDetector:
    def __init__(self, bank: torch.Tensor, bank_ids: torch.Tensor, scale: float):
        self.bank = bank
        self.bank_ids = bank_ids
        self.scale = scale  # typical max-distance of normal images (calibration)

    @classmethod
    def from_reference_images(cls, paths) -> "AnomalyDetector":
        paths = [Path(p) for p in paths]
        if len(paths) < 2:
            raise ValueError("At least 2 normal reference images are required.")
        key = tuple((str(p.resolve()), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
        if key in _DETECTOR_CACHE:
            return _DETECTOR_CACHE[key]

        per_image = []
        for p in paths:
            with Image.open(p) as im:
                per_image.append(_extract_patches(im.convert("RGB")))
        ids = torch.cat([torch.full((f.shape[0],), i, dtype=torch.long) for i, f in enumerate(per_image)])
        bank = torch.cat(per_image, dim=0)

        if bank.shape[0] > MAX_BANK_PATCHES:  # reproducible random subsample
            gen = torch.Generator().manual_seed(0)
            keep = torch.randperm(bank.shape[0], generator=gen)[:MAX_BANK_PATCHES]
            bank = bank[keep.to(bank.device)]
            ids = ids[keep]
        ids = ids.to(bank.device)

        # Calibration: score each reference image against the OTHER references.
        maxima = []
        for i, f in enumerate(per_image):
            d = _min_distance(f, bank, ids, exclude_id=i)
            d = d[torch.isfinite(d)]
            if d.numel():
                maxima.append(float(d.max()))
        scale = max(maxima) if maxima else 1.0
        detector = cls(bank, ids, max(scale, 1e-6))

        _DETECTOR_CACHE.clear()  # keep only the latest reference set
        _DETECTOR_CACHE[key] = detector
        return detector

    @torch.no_grad()
    def score_image(self, image: Image.Image) -> Dict:
        """Return {'score', 'raw_score', 'heatmap'}.

        score   : max patch distance / calibration scale (about 1.0 = as different
                  as normal references are from each other; relative, not a probability)
        heatmap : float32 array (256x256) in [0,1]; 0.5 is the normal-variation level
        """
        feats = _extract_patches(image)
        dist = _min_distance(feats, self.bank, self.bank_ids)
        side = int(round(feats.shape[0] ** 0.5))
        amap = dist.view(1, 1, side, side)
        amap = F.interpolate(amap, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        amap = _gaussian_blur(amap)[0, 0]
        raw = float(amap.max())
        heat = (amap / (2.0 * self.scale)).clamp(0, 1).cpu().numpy().astype(np.float32)
        return {"score": raw / self.scale, "raw_score": raw, "heatmap": heat}


def create_heatmap_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.6) -> Image.Image:
    """Blend a [0,1] heatmap (jet colors) over the image; returns a PIL RGB image."""
    from matplotlib import colormaps

    image = image.convert("RGB")
    h = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    h_img = Image.fromarray((h * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    h = np.asarray(h_img, dtype=np.float32) / 255.0
    colors = colormaps["jet"](h)[..., :3]
    base = np.asarray(image, dtype=np.float32) / 255.0
    a = (alpha * h)[..., None]                      # low values stay transparent
    out = base * (1 - a) + colors * a
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))