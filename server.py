"""DeepVision AI - web server (Flask).

Start from the project root:
    python server.py
then open http://127.0.0.1:8000  (a browser tab opens automatically).

Options:  --port 8001   --host 127.0.0.1   --no-browser
"""
import argparse
import base64
import io
import logging
import re
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from matplotlib import colormaps
from PIL import Image, ImageOps, UnidentifiedImageError

from deepvision.anomaly import AnomalyDetector, list_images
from deepvision.classifier import get_device, load_classifier, predict_classifier

ROOT = Path(__file__).resolve().parent
CLASSIFIER_PATH = ROOT / "models" / "classifier.pt"
REFERENCE_DIR = ROOT / "data" / "anomaly" / "reference_normal"
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_MB = 16
HEATMAP_MAX_SIDE = 1024

Image.MAX_IMAGE_PIXELS = 60_000_000

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
log = logging.getLogger("deepvision")
_lock = threading.Lock()  # one model computation at a time (thread safety)

# ---------------------------------------------------------------- helpers
_NORMAL_TOKENS = {"normal", "good", "ok", "pass", "passed", "healthy", "clean", "fine", "intact",
                  "nodefect", "nodefects", "nondefect", "nondefective", "notdefective", "defectfree"}
_NEGATORS = {"no", "non", "not", "without", "free"}


def is_defect_class(name: str) -> bool:
    """Class names that look 'normal/good/ok' are OK; every other class counts as a defect."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]
    if any(t in _NORMAL_TOKENS for t in tokens):
        return False
    if any(t in _NEGATORS for t in tokens) and any(t.startswith("defect") for t in tokens):
        return False
    return True


def read_image(file_storage) -> Image.Image:
    name = (file_storage.filename or "").lower()
    if Path(name).suffix not in ALLOWED_EXT:
        raise ValueError("Unsupported file type. Use JPG, PNG, BMP or WEBP.")
    try:
        img = Image.open(file_storage.stream)
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError("This file could not be read as an image.")
    return ImageOps.exif_transpose(img).convert("RGB")


def heatmap_data_url(heat: np.ndarray, size) -> str:
    """Colorised (jet) RGBA PNG; transparent where the image looks normal."""
    w, h = size
    scale = min(1.0, HEATMAP_MAX_SIDE / max(w, h))
    target = (max(1, round(w * scale)), max(1, round(h * scale)))
    hm = np.asarray(Image.fromarray(heat.astype(np.float32)).resize(target, Image.BILINEAR))
    hm = np.clip(hm, 0.0, 1.0)
    rgba = (colormaps["jet"](hm) * 255).astype(np.uint8)
    rgba[..., 3] = (np.clip((hm - 0.25) / 0.5, 0.0, 1.0) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def analyze_classifier(image):
    if not CLASSIFIER_PATH.exists():
        return {"available": False,
                "message": "Classifier not trained yet. Run: python scripts/train_classifier.py"}
    try:
        model, classes, device = load_classifier(CLASSIFIER_PATH)
        res = predict_classifier(model, classes, image, device)
        return {"available": True, "label": res["label"], "confidence": res["confidence"],
                "probabilities": res["probabilities"], "is_defect": is_defect_class(res["label"]),
                "defect_classes": [c for c in classes if is_defect_class(c)]}
    except Exception as exc:  # report, do not crash the server
        log.exception("classifier failed")
        return {"available": False, "message": f"Classifier error: {exc}"}


def analyze_anomaly(image):
    refs = list_images(REFERENCE_DIR)
    if len(refs) < 2:
        return {"available": False,
                "message": "Add at least 2 normal images to data/anomaly/reference_normal/ "
                           "(or run: python scripts/prepare_data.py)."}
    try:
        detector = AnomalyDetector.from_reference_images(refs)
        res = detector.score_image(image)
        return {"available": True, "score": res["score"], "raw_score": res["raw_score"],
                "references": len(refs), "heatmap": heatmap_data_url(res["heatmap"], image.size)}
    except Exception as exc:
        log.exception("anomaly detection failed")
        return {"available": False, "message": f"Anomaly detection error: {exc}"}


def warm_up():
    """Load models in the background so the first request is faster."""
    try:
        with _lock:
            if CLASSIFIER_PATH.exists():
                load_classifier(CLASSIFIER_PATH)
            refs = list_images(REFERENCE_DIR)
            if len(refs) >= 2:
                AnomalyDetector.from_reference_images(refs)
        log.info("Warm-up finished.")
    except Exception as exc:
        log.warning("Warm-up skipped: %s", exc)


# ----------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    info = {"ok": True, "device": str(get_device()), "classifier": {"ready": False, "classes": [], "message": ""},
            "reference_count": len(list_images(REFERENCE_DIR))}
    if CLASSIFIER_PATH.exists():
        try:
            with _lock:
                _, classes, _ = load_classifier(CLASSIFIER_PATH)
            info["classifier"] = {"ready": True, "classes": classes, "message": ""}
        except Exception as exc:
            info["classifier"]["message"] = str(exc)
    else:
        info["classifier"]["message"] = "Not trained yet"
    return jsonify(info)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")
    mode = request.form.get("mode", "combined")
    if file is None or not file.filename:
        return jsonify(ok=False, error="No image was uploaded."), 400
    if mode not in ("combined", "classifier", "anomaly"):
        return jsonify(ok=False, error="Unknown analysis mode."), 400
    try:
        image = read_image(file)
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400

    start = time.perf_counter()
    out = {"classifier": None, "anomaly": None}
    with _lock:
        if mode in ("combined", "classifier"):
            out["classifier"] = analyze_classifier(image)
        if mode in ("combined", "anomaly"):
            out["anomaly"] = analyze_anomaly(image)
    return jsonify(ok=True, mode=mode, width=image.width, height=image.height,
                   elapsed_ms=round((time.perf_counter() - start) * 1000), **out)


@app.errorhandler(413)
def too_large(_):
    return jsonify(ok=False, error=f"File too large (max {MAX_UPLOAD_MB} MB)."), 413


@app.errorhandler(500)
def server_error(_):
    return jsonify(ok=False, error="Internal server error. See the terminal for details."), 500


@app.after_request
def no_cache_api(resp):
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


def main():
    parser = argparse.ArgumentParser(description="DeepVision AI web server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    threading.Thread(target=warm_up, daemon=True).start()
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"\nDeepVision AI running at {url}   (press Ctrl+C to stop)\n")
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except OSError as exc:
        raise SystemExit(f"Could not start the server: {exc}\nTry another port, e.g.: python server.py --port 8001")


if __name__ == "__main__":
    main()