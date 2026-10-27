import numpy as np
import torch

from boxsmith.data import scene
from boxsmith.matching import assign, detection_loss
from boxsmith.model import TinyDetector, anchors_for


def test_assignment_covers_colliding_objects_and_offsets_decode():
    anchors = anchors_for().numpy()
    target = {"boxes": np.array([[30, 30, 50, 50], [31, 31, 51, 51]], dtype=np.float32),
              "labels": np.array([0, 1])}
    state, labels, offsets = assign(anchors, target)
    assert set(labels[state == 1]) == {0, 1}
    centers = (anchors[:, :2] + anchors[:, 2:]) / 2
    sizes = anchors[:, 2:] - anchors[:, :2]
    decoded_centers = offsets[:, :2] * sizes + centers
    decoded_sizes = np.exp(offsets[:, 2:]) * sizes
    decoded = np.concatenate([decoded_centers - decoded_sizes / 2, decoded_centers + decoded_sizes / 2], 1)
    for label in (0, 1):
        np.testing.assert_allclose(decoded[(state == 1) & (labels == label)][0], target["boxes"][label], atol=1e-5)


def test_cpu_forward_backward_and_empty_target():
    torch.set_num_threads(2)
    model = TinyDetector()
    image, target = scene(3, "train")
    empty = {"boxes": np.empty((0, 4)), "labels": np.empty(0, dtype=np.int64)}
    images = torch.stack([torch.from_numpy(image).permute(2, 0, 1).float() / 255] * 2)
    raw = model(images)
    assert raw.shape == (2, 768, 8)
    assert sum(p.numel() for p in model.parameters()) < 1_000_000
    loss = detection_loss(raw, [assign(model.anchors.numpy(), t) for t in [target, empty]])
    loss["total"].backward()
    assert torch.isfinite(loss["total"])
    decoded = model.decode(raw)
    assert decoded[0]["boxes"].min() >= 0
    assert decoded[0]["boxes"].max() <= 128
