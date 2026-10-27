"""Pixel xyxy boxes; right and bottom bounds are exclusive."""

import numpy as np


def iou(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                              np.maximum(a[:, None, :2], b[None, :, :2])).prod(-1)
    area_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(-1)
    area_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(-1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def empty_predictions():
    return {"boxes": np.empty((0, 4), dtype=np.float32),
            "scores": np.empty(0, dtype=np.float32), "labels": np.empty(0, dtype=np.int64)}


def nms(boxes, scores, labels, threshold=0.5, max_detections=100):
    """Class-aware greedy NMS. Stable input order breaks equal-score ties."""
    if not 0 <= threshold <= 1 or max_detections < 1:
        raise ValueError("NMS IoU must be in [0,1] and max_detections positive")
    order = np.argsort(-np.asarray(scores), kind="stable")
    keep = []
    while len(order) and len(keep) < max_detections:
        current, rest = order[0], order[1:]
        keep.append(current)
        overlaps = iou(np.asarray(boxes)[current:current + 1], np.asarray(boxes)[rest])[0]
        order = rest[~((np.asarray(labels)[rest] == labels[current]) & (overlaps > threshold))]
    return np.asarray(keep, dtype=np.int64)


def postprocess(prediction, nms_iou=0.5, score_floor=0.001, max_detections=100):
    boxes, scores = prediction["boxes"], prediction["scores"]
    valid = ((scores >= score_floor) & np.isfinite(scores) & np.isfinite(boxes).all(1) &
             (boxes[:, 2:] > boxes[:, :2]).all(1))
    p = {key: np.asarray(value)[valid] for key, value in prediction.items()}
    keep = nms(p["boxes"], p["scores"], p["labels"], nms_iou, max_detections)
    return {key: value[keep] for key, value in p.items()}
