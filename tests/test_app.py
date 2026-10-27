import numpy as np
from pathlib import Path
from streamlit.testing.v1 import AppTest

from boxsmith.data import SceneConfig, scene
from boxsmith.examples import controlled_predictions
from boxsmith.metrics import evaluate
from boxsmith.boxes import postprocess


def test_app_confidence_keeps_reference_ap_and_perfect_example(monkeypatch):
    monkeypatch.setenv("BOXSMITH_CHECKPOINT", "/nonexistent/boxsmith.pt")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    assert app.selectbox(key="prediction_source").value == "Controlled examples"
    reference = [metric.value for metric in app.metric[2:4]]
    app.slider(key="confidence").set_value(0.9).run()
    assert not app.exception
    assert [metric.value for metric in app.metric[2:4]] == reference
    app.selectbox(key="error_scenario").set_value("Perfect detections").run()
    assert not app.exception
    assert app.metric[2].value == "100.0%"
    assert app.metric[3].value == "100.0%"
    app.selectbox(key="prediction_source").set_value("Trained tiny CNN").run()
    assert not app.exception
    assert "Choose a trained checkpoint" in app.info[0].value


def test_controlled_empty_scene_and_nms_experiment():
    _, empty = scene(0, config=SceneConfig(max_shapes=0))
    assert not len(controlled_predictions(empty, 0, "Perfect detections")["boxes"])
    target = {"boxes": np.array([[20, 20, 40, 40], [70, 70, 90, 90]], dtype=np.float32),
              "labels": np.array([0, 0])}
    raw = controlled_predictions(target, 3, "Duplicates")
    raw["scores"] = np.array([0.95, 0.94, 0.60, 0.59])
    suppressed = postprocess(raw, 0.5)
    assert len(suppressed["boxes"]) == len(target["boxes"])
    assert evaluate([suppressed], [target])["map"] == 1
    if len(target["boxes"]) > 1:
        assert evaluate([raw], [target])["map"] < 1


def test_trained_checkpoint_view_and_new_empty_scene(monkeypatch, tmp_path):
    import torch
    from dataclasses import asdict
    from boxsmith.model import TinyDetector

    model = TinyDetector()
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"model": model.state_dict(), "image_size": 128, "scene_config": asdict(SceneConfig()),
                "config": {"seed": 42}, "epoch": 0, "parameter_count": 173536}, checkpoint)
    monkeypatch.setenv("BOXSMITH_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("BOXSMITH_CACHE", str(tmp_path / "cache"))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    assert app.selectbox(key="prediction_source").value == "Trained tiny CNN"
    app.radio[0].set_value("New seeded scene").run()
    assert not app.exception
    empty_index = next(i for i in range(100) if not len(scene(i, "scene")[1]["boxes"]))
    app.number_input[0].set_value(empty_index).run()
    assert not app.exception
    assert any("This scene: 0 TP · 0 FP · 0 missed objects" in c.value for c in app.caption)
