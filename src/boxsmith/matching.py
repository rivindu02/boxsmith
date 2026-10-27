"""Deterministic training assignment; each object gets a unique forced anchor."""

import numpy as np
import torch
from torch.nn import functional as F

from boxsmith.boxes import iou


def assign(anchors, target, positive_iou=0.45, negative_iou=0.25):
    anchors = np.asarray(anchors)
    count = len(anchors)
    # -1 ignore, 0 background, 1 positive.
    state = np.zeros(count, dtype=np.int64)
    labels = np.zeros(count, dtype=np.int64)
    offsets = np.zeros((count, 4), dtype=np.float32)
    if not len(target["boxes"]):
        return state, labels, offsets
    overlaps = iou(anchors, target["boxes"])
    best = overlaps.argmax(1)
    maximum = overlaps.max(1)
    state[maximum >= negative_iou] = -1
    state[maximum >= positive_iou] = 1
    reserved = set()
    for gt in range(len(target["boxes"])):
        for index in np.argsort(-overlaps[:, gt], kind="stable"):
            if int(index) not in reserved:
                reserved.add(int(index))
                best[index], state[index] = gt, 1
                break
        else:
            raise ValueError("More ground-truth objects than anchors")
    labels = target["labels"][best]
    ground_truth = target["boxes"][best]
    centers = (anchors[:, :2] + anchors[:, 2:]) / 2
    sizes = anchors[:, 2:] - anchors[:, :2]
    gt_centers = (ground_truth[:, :2] + ground_truth[:, 2:]) / 2
    gt_sizes = ground_truth[:, 2:] - ground_truth[:, :2]
    offsets[:, :2] = (gt_centers - centers) / sizes
    offsets[:, 2:] = np.log(gt_sizes / sizes)
    return state, labels, offsets


def detection_loss(raw, assignments):
    state, labels, offsets = [torch.stack([torch.as_tensor(a[i]) for a in assignments]).to(raw.device)
                              for i in range(3)]
    positive = state == 1
    objectness = F.binary_cross_entropy_with_logits(raw[..., 0], positive.float(), reduction="none")
    negative_count = torch.minimum((state == 0).sum(1), (3 * positive.sum(1)).clamp(min=16))
    hardness = objectness.detach().masked_fill(state != 0, -torch.inf)
    rank = hardness.argsort(dim=1, descending=True, stable=True).argsort(dim=1)
    selected = positive | ((state == 0) & (rank < negative_count[:, None]))
    obj_loss = objectness[selected].mean()
    if positive.any():
        cls_loss = F.cross_entropy(raw[..., 1:4][positive], labels[positive])
        box_loss = F.smooth_l1_loss(raw[..., 4:8][positive], offsets[positive], beta=0.1)
    else:
        cls_loss = raw.sum() * 0
        box_loss = raw.sum() * 0
    return {"total": obj_loss + cls_loss + 5 * box_loss,
            "objectness": obj_loss, "classification": cls_loss, "box": box_loss}
