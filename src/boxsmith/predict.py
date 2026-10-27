"""Inference caches identify model bytes, data generation, and decoding settings."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from boxsmith.boxes import postprocess
from boxsmith.data import ShapesDataset, collate
from boxsmith.model import TinyDetector

SCORE_FLOOR = 0.001
MAX_DETECTIONS = 100
DECODER_VERSION = 1


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(checkpoint, device="cpu"):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = TinyDetector(payload["image_size"])
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), payload


@torch.inference_mode()
def predict_dataset(model, dataset, device="cpu", batch_size=64):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    predictions, targets = [], []
    model.eval()
    for images, target in loader:
        predictions.extend(model.decode(model(images.to(device))))
        targets.extend(target)
    return predictions, targets


def cached_predictions(checkpoint, dataset, cache_dir, device="cpu"):
    identity = {"checkpoint": file_hash(checkpoint), "dataset": dataset.identity(),
                "decoder": DECODER_VERSION}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f"{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            predictions = [{k: cache[f"{i}_{k}"] for k in ("boxes", "scores", "labels")}
                           for i in range(len(dataset))]
        targets = [dataset[i][1] for i in range(len(dataset))]
        return predictions, targets, identity
    model, _ = load_model(checkpoint, device)
    predictions, targets = predict_dataset(model, dataset, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"{i}_{k}": v for i, p in enumerate(predictions) for k, v in p.items()}
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    # Bound persistent cache count, retaining the current entry.
    old = sorted(path.parent.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in old[6:]:
        if stale != path:
            stale.unlink()
    return predictions, targets, identity


def suppress(predictions, nms_iou=0.5):
    return [postprocess(p, nms_iou, SCORE_FLOOR, MAX_DETECTIONS) for p in predictions]
