import numpy as np
import torch

from boxsmith.data import ShapesDataset, SceneConfig
from boxsmith.model import TinyDetector
from boxsmith.predict import cached_predictions, file_hash


def test_prediction_cache_identity_and_safe_roundtrip(tmp_path):
    torch.set_num_threads(2)
    model = TinyDetector()
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "image_size": 128}, checkpoint)
    dataset = ShapesDataset(3, "val")
    first, targets, identity = cached_predictions(checkpoint, dataset, tmp_path / "cache")
    again, _, cached_identity = cached_predictions(checkpoint, dataset, tmp_path / "cache")
    assert identity == cached_identity
    for a, b in zip(first, again):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    cached_predictions(checkpoint, ShapesDataset(3, "test"), tmp_path / "cache")
    cached_predictions(checkpoint, ShapesDataset(3, "val", config=SceneConfig(max_shapes=0)), tmp_path / "cache")
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 3
    old_hash = file_hash(checkpoint)
    with torch.no_grad():
        model.head.bias.add_(0.1)
    torch.save({"model": model.state_dict(), "image_size": 128}, checkpoint)
    assert file_hash(checkpoint) != old_hash
