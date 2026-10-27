"""Transparent no-crowd, all-area, 101-point detection AP."""

import numpy as np

from boxsmith.boxes import iou


def class_curve(predictions, targets, label, matching_iou=0.5):
    if len(predictions) != len(targets) or not 0 <= matching_iou <= 1:
        raise ValueError("Aligned image lists and matching IoU in [0,1] required")
    ground_truth = [t["boxes"][t["labels"] == label] for t in targets]
    total = sum(len(g) for g in ground_truth)
    matched = [np.zeros(len(g), dtype=bool) for g in ground_truth]
    detections = [(float(p["scores"][j]), image, j)
                  for image, p in enumerate(predictions) for j in range(len(p["scores"]))
                  if p["labels"][j] == label]
    detections.sort(key=lambda item: -item[0])  # stable image/candidate ties
    tp = np.zeros(len(detections), dtype=np.int64)
    for rank, (_, image, j) in enumerate(detections):
        overlaps = iou(predictions[image]["boxes"][j:j + 1], ground_truth[image])[0]
        overlaps[matched[image]] = -1
        if len(overlaps):
            best = int(np.argmax(overlaps))
            if overlaps[best] >= matching_iou:
                tp[rank] = 1
                matched[image][best] = True
    cumulative_tp = np.cumsum(tp)
    precision = cumulative_tp / np.arange(1, len(tp) + 1)
    recall = cumulative_tp / total if total else np.zeros(len(tp))
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    sampled = np.zeros(101)
    valid = indices < len(envelope)
    sampled[valid] = envelope[indices[valid]]
    return {"ap": float(sampled.mean()) if total else None,
            "precision": precision, "recall": recall, "tp": tp,
            "scores": np.asarray([d[0] for d in detections]), "ground_truth_count": total}


def evaluate(predictions, targets, matching_iou=0.5, class_count=3):
    curves = [class_curve(predictions, targets, c, matching_iou) for c in range(class_count)]
    aps = [c["ap"] for c in curves if c["ap"] is not None]
    return {"map": float(np.mean(aps)) if aps else None, "curves": curves}


def operating_point(curves, confidence):
    """Micro precision/recall over every class, including predictions of absent classes."""
    tp = sum(int(c["tp"][c["scores"] >= confidence].sum()) for c in curves)
    detections = sum(int((c["scores"] >= confidence).sum()) for c in curves)
    total = sum(c["ground_truth_count"] for c in curves)
    return {"tp": tp, "fp": detections - tp, "fn": total - tp,
            "precision": tp / detections if detections else 0.0,
            "recall": tp / total if total else 0.0}


def summary(predictions, targets):
    values = [evaluate(predictions, targets, float(threshold))["map"]
              for threshold in np.linspace(0.5, 0.95, 10)]
    return {"map50": values[0], "map50_95": float(np.mean([v for v in values if v is not None]))
            if any(v is not None for v in values) else None}
