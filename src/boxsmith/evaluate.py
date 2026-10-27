"""Export reproducible validation/test metrics for a checkpoint."""

import argparse
import json
import time
from pathlib import Path

import torch

from boxsmith.data import CLASS_NAMES, SceneConfig, ShapesDataset
from boxsmith.metrics import evaluate, summary
from boxsmith.predict import MAX_DETECTIONS, SCORE_FLOOR, cached_predictions, load_model, suppress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--output", default="outputs/evaluation.json")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("Count must be positive")
    torch.set_num_threads(4)
    _, metadata = load_model(args.checkpoint)
    dataset = ShapesDataset(args.count, args.split, metadata["config"]["seed"],
                            SceneConfig(**metadata["scene_config"]))
    started = time.perf_counter()
    raw, targets, identity = cached_predictions(args.checkpoint, dataset, args.cache, args.device)
    inference_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = suppress(raw, args.nms_iou)
    suppression_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scores = summary(predictions, targets)
    curves = evaluate(predictions, targets)["curves"]
    result = {**scores, "ap50_by_class": {name: c["ap"] for name, c in zip(CLASS_NAMES, curves)},
              "identity": identity, "nms_iou": args.nms_iou, "score_floor": SCORE_FLOOR,
              "max_detections": MAX_DETECTIONS, "checkpoint_epoch": metadata["epoch"],
              "parameters": metadata["parameter_count"], "untrained_map50": metadata["untrained_map50"],
              "inference_or_cache_seconds": inference_seconds, "nms_seconds": suppression_seconds,
              "metric_seconds": time.perf_counter() - started}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
