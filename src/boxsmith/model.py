"""A small, anchor-based CNN detector trained from scratch."""

import math

import torch
from torch import nn


def anchors_for(image_size=128):
    y, x = torch.meshgrid(torch.arange(image_size // 8), torch.arange(image_size // 8), indexing="ij")
    centers = torch.stack([x, y], -1).float().reshape(-1, 1, 2) * 8 + 4
    sizes = torch.tensor([18.0, 26.0, 36.0]).reshape(1, 3, 1).expand(len(centers), -1, 2)
    centers = centers.expand(-1, 3, -1)
    return torch.cat([centers - sizes / 2, centers + sizes / 2], -1).reshape(-1, 4)


class TinyDetector(nn.Module):
    def __init__(self, image_size=128):
        super().__init__()
        self.image_size = image_size
        layers = []
        previous = 3
        for channels, stride in [(24, 2), (40, 2), (64, 2), (96, 1), (96, 1)]:
            layers.extend([nn.Conv2d(previous, channels, 3, stride, 1, bias=False),
                           nn.BatchNorm2d(channels), nn.SiLU()])
            previous = channels
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(96, 3 * 8, 1)
        self.register_buffer("anchors", anchors_for(image_size))
        with torch.no_grad():
            self.head.bias.reshape(3, 8)[:, 0] = math.log(0.01 / 0.99)

    def forward(self, images):
        raw = self.head(self.backbone(images))
        return raw.permute(0, 2, 3, 1).reshape(len(images), -1, 8)

    def decode(self, raw):
        centers = (self.anchors[:, :2] + self.anchors[:, 2:]) / 2
        sizes = self.anchors[:, 2:] - self.anchors[:, :2]
        predicted_centers = raw[..., 4:6] * sizes + centers
        predicted_sizes = raw[..., 6:8].clamp(-3, 3).exp() * sizes
        boxes = torch.cat([predicted_centers - predicted_sizes / 2,
                           predicted_centers + predicted_sizes / 2], -1).clamp(0, self.image_size)
        class_probability, labels = raw[..., 1:4].softmax(-1).max(-1)
        scores = raw[..., 0].sigmoid() * class_probability
        return [{"boxes": b.detach().cpu().numpy(), "scores": s.detach().cpu().numpy(),
                 "labels": label.detach().cpu().numpy()} for b, s, label in zip(boxes, scores, labels)]
