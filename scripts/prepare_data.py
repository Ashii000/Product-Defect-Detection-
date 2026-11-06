"""Arrange images into the folders DeepVision AI expects.

Creates (inside the project's data/ folder):
    data/anomaly/reference_normal/        <- normal images for Mode B
    data/classifier/train/{normal,defective}/
    data/classifier/val/{normal,defective}/

MODE 1 - MVTec AD category (e.g. "bottle"):
    python scripts/prepare_data.py --mvtec "C:/path/to/mvtec_anomaly_detection/bottle"

MODE 2 - your own photos (two folders: good photos and defective photos):
    python scripts/prepare_data.py --normal-dir "C:/my/good" --defect-dir "C:/my/bad"

Re-running: add --clean to delete the previously prepared images first.
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEST = {
    "reference": DATA / "anomaly" / "reference_normal",
    "train_normal": DATA / "classifier" / "train" / "normal",
    "train_defective": DATA / "classifier" / "train" / "defective",
    "val_normal": DATA / "classifier" / "val" / "normal",
    "val_defective": DATA / "classifier" / "val" / "defective",
}


def images_in(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS)


def prepare_destinations(clean: bool):
    existing = [p for d in DEST.values() for p in images_in(d)]
    if existing and not clean:
        sys.exit(f"ERROR: {len(existing)} images already exist in data/.\n"
                 "Run again with --clean to replace them.")
    for d in DEST.values():
        d.mkdir(parents=True, exist_ok=True)
        if clean:
            for p in images_in(d):
                p.unlink()
        (d / ".gitkeep").touch()


def copy_files(files, dest: Path, prefix: str = ""):
    for i, src in enumerate(files):
        shutil.copy2(src, dest / f"{prefix}{i:04d}_{src.name}")


def split(files, train_fraction: float, rng: random.Random):
    files = list(files)
    rng.shuffle(files)
    n_train = min(max(1, round(len(files) * train_fraction)), len(files) - 1)  # keep >=1 for val
    return files[:n_train], files[n_train:]


def sample(files, n, rng: random.Random):
    files = list(files)
    rng.shuffle(files)
    return files[:n]


def run_mvtec(src: Path, a, rng):
    train_good = images_in(src / "train" / "good")
    test_good = images_in(src / "test" / "good")
    test_dir = src / "test"
    defect_dirs = sorted(d for d in test_dir.iterdir() if d.is_dir() and d.name != "good") if test_dir.is_dir() else []
    if not train_good or not test_good or not defect_dirs:
        sys.exit(f"ERROR: '{src}' does not look like one MVTec AD category folder.\n"
                 "Expected: train/good/, test/good/ and test/<defect_type>/ inside it\n"
                 "(for example the folder named 'bottle' inside mvtec_anomaly_detection).")
    defect_files = []
    for d in defect_dirs:
        for i, f in enumerate(images_in(d)):
            defect_files.append((d.name, f))
    if len(defect_files) < 2:
        sys.exit("ERROR: fewer than 2 defective images found.")

    prepare_destinations(a.clean)
    ref = sample(train_good, a.max_reference, rng)
    copy_files(ref, DEST["reference"])
    copy_files(sample(train_good, a.max_train_normal, rng), DEST["train_normal"])
    copy_files(test_good, DEST["val_normal"])

    tr, va = split(defect_files, a.train_fraction, rng)
    for dest, items in ((DEST["train_defective"], tr), (DEST["val_defective"], va)):
        for i, (kind, f) in enumerate(items):
            shutil.copy2(f, dest / f"{kind}_{i:04d}_{f.name}")
    return len(ref)


def run_custom(normal_dir: Path, defect_dir: Path, a, rng):
    normal, defect = images_in(normal_dir), images_in(defect_dir)
    if len(normal) < 2 or len(defect) < 2:
        sys.exit(f"ERROR: need at least 2 images in each folder (found {len(normal)} normal, {len(defect)} defective).")
    prepare_destinations(a.clean)
    n_tr, n_va = split(normal, a.train_fraction, rng)
    d_tr, d_va = split(defect, a.train_fraction, rng)
    copy_files(n_tr, DEST["train_normal"]); copy_files(n_va, DEST["val_normal"])
    copy_files(d_tr, DEST["train_defective"]); copy_files(d_va, DEST["val_defective"])
    ref = sample(n_tr, a.max_reference, rng)       # references never come from val images
    copy_files(ref, DEST["reference"])
    return len(ref)


def main():
    p = argparse.ArgumentParser(description="Prepare DeepVision AI data folders.")
    p.add_argument("--mvtec", type=Path, help="path to ONE MVTec AD category folder, e.g. .../bottle")
    p.add_argument("--normal-dir", type=Path, help="folder with your good/normal photos")
    p.add_argument("--defect-dir", type=Path, help="folder with your defective photos")
    p.add_argument("--max-reference", type=int, default=30, help="normal images for Mode B (default 30)")
    p.add_argument("--max-train-normal", type=int, default=100, help="MVTec only: normal training images (default 100)")
    p.add_argument("--train-fraction", type=float, default=0.7, help="share of images used for training (default 0.7)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean", action="store_true", help="delete previously prepared images first")
    a = p.parse_args()
    rng = random.Random(a.seed)

    if not 0.1 <= a.train_fraction <= 0.9:
        sys.exit("ERROR: --train-fraction must be between 0.1 and 0.9.")
    if a.max_reference < 2 or a.max_train_normal < 1:
        sys.exit("ERROR: --max-reference must be >= 2 and --max-train-normal >= 1.")
    custom = a.normal_dir is not None or a.defect_dir is not None
    if a.mvtec and custom:
        sys.exit("ERROR: use EITHER --mvtec OR (--normal-dir and --defect-dir), not both.")
    if a.mvtec:
        if not a.mvtec.is_dir():
            sys.exit(f"ERROR: folder not found: {a.mvtec}")
        n_ref = run_mvtec(a.mvtec, a, rng)
    elif custom:
        if not (a.normal_dir and a.defect_dir):
            sys.exit("ERROR: give BOTH --normal-dir and --defect-dir.")
        for d in (a.normal_dir, a.defect_dir):
            if not d.is_dir():
                sys.exit(f"ERROR: folder not found: {d}")
        n_ref = run_custom(a.normal_dir, a.defect_dir, a, rng)
    else:
        sys.exit("ERROR: choose a mode: --mvtec <category folder>  OR  --normal-dir <dir> --defect-dir <dir>.\nSee the top of this file.")

    print("Done. Images prepared:")
    print(f"  data/anomaly/reference_normal : {len(images_in(DEST['reference']))}  (Mode B)")
    for k, label in (("train_normal", "classifier/train/normal"), ("train_defective", "classifier/train/defective"),
                     ("val_normal", "classifier/val/normal"), ("val_defective", "classifier/val/defective")):
        print(f"  data/{label:<28}: {len(images_in(DEST[k]))}")
    print("\nNext: python scripts/train_classifier.py   then   streamlit run app.py")


if __name__ == "__main__":
    main()