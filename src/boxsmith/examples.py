"""Controlled predictions for teaching; these are explicitly not model outputs."""

import numpy as np

from boxsmith.boxes import empty_predictions

SCENARIOS = ("Mixed errors", "Perfect detections", "Duplicates", "Missed objects", "Wrong classes", "Loose boxes")


def controlled_predictions(target, index, scenario="Mixed errors", image_size=128):
    rng = np.random.default_rng(1000 + index)
    boxes, labels, scores = [], [], []
    for j, (box, label) in enumerate(zip(target["boxes"], target["labels"])):
        if scenario in ("Missed objects", "Mixed errors") and j % 3 == 2:
            continue
        candidate = box.copy()
        if scenario == "Loose boxes" or (scenario == "Mixed errors" and j % 3 == 1):
            center = (box[:2] + box[2:]) / 2
            half = (box[2:] - box[:2]) * 0.35
            candidate = np.concatenate([center - half, center + half])
        if scenario == "Wrong classes":
            label = (int(label) + 1) % 3
        score = float(rng.uniform(0.55, 0.98))
        boxes.append(candidate)
        labels.append(label)
        scores.append(score)
        if scenario in ("Duplicates", "Mixed errors"):
            boxes.append(np.clip(candidate + 1, 0, image_size))
            labels.append(label)
            scores.append(score * 0.85)
    if scenario == "Mixed errors":
        boxes.append([2, 2, 18, 18])
        labels.append(index % 3)
        scores.append(0.8)
    if not boxes:
        return empty_predictions()
    return {"boxes": np.array(boxes, dtype=np.float32), "labels": np.array(labels, dtype=np.int64),
            "scores": np.array(scores, dtype=np.float32)}
