"""Reproducible shapes generated on demand, without image files."""

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageDraw

from boxsmith.boxes import iou

CLASS_NAMES = ("circle", "square", "triangle")
GENERATOR_VERSION = 1
SPLITS = {"train": 11, "val": 23, "test": 37, "scene": 53}
PALETTE = [(250, 90, 100), (62, 210, 160), (80, 160, 255), (250, 200, 65), (190, 115, 245)]


@dataclass(frozen=True)
class SceneConfig:
    image_size: int = 128
    max_shapes: int = 6
    min_size: int = 16
    max_size: int = 36
    max_overlap: float = 0.10

    def __post_init__(self):
        if not (8 <= self.min_size <= self.max_size < self.image_size):
            raise ValueError("Shape sizes must fit inside the image")
        if self.image_size % 8 or self.max_shapes < 0 or not 0 <= self.max_overlap <= 1:
            raise ValueError("Use image size divisible by 8, nonnegative count, overlap in [0,1]")


def scene(index, split="scene", seed=42, config=SceneConfig()):
    if split not in SPLITS or index < 0 or seed < 0:
        raise ValueError("Unknown split or negative index/seed")
    rng = np.random.default_rng(np.random.SeedSequence([seed, SPLITS[split], index, GENERATOR_VERSION]))
    size = config.image_size
    image = Image.new("RGB", (size, size), (19, 25, 39))
    draw = ImageDraw.Draw(image)
    boxes, labels = [], []
    count = int(rng.integers(0, config.max_shapes + 1))
    for _ in range(count):
        for attempt in range(100):
            width = int(rng.integers(config.min_size, config.max_size + 1))
            height = width  # equal aspect ratios keep the first detector inspectable
            x, y = (int(rng.integers(1, size - width)), int(rng.integers(1, size - height)))
            box = [x, y, x + width, y + height]
            if not boxes or iou([box], boxes).max() <= config.max_overlap:
                break
        else:
            continue
        label = int(rng.integers(3))
        colour = PALETTE[int(rng.integers(len(PALETTE)))]
        pixel_box = [x, y, x + width - 1, y + height - 1]
        if label == 0:
            draw.ellipse(pixel_box, fill=colour)
        elif label == 1:
            draw.rectangle(pixel_box, fill=colour)
        else:
            draw.polygon([(x + width // 2, y), (x + width - 1, y + height - 1),
                          (x, y + height - 1)], fill=colour)
        boxes.append(box)
        labels.append(label)
    return np.asarray(image).copy(), {"boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                                     "labels": np.asarray(labels, dtype=np.int64)}


class ShapesDataset:
    def __init__(self, count, split="train", seed=42, config=SceneConfig()):
        self.count, self.split, self.seed, self.config = count, split, seed, config

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        import torch
        image, target = scene(index, self.split, self.seed, self.config)
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255, target

    def identity(self):
        return {"generator_version": GENERATOR_VERSION, "split": self.split,
                "seed": self.seed, "count": self.count, "config": asdict(self.config)}


def collate(batch):
    import torch
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
