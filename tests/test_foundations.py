import contextlib
import io

import numpy as np
import pytest

from boxsmith.boxes import empty_predictions, iou, nms, postprocess
from boxsmith.data import SceneConfig, scene
from boxsmith.metrics import evaluate, operating_point, summary


def target(boxes, labels):
    return {"boxes": np.array(boxes, dtype=float).reshape(-1, 4), "labels": np.array(labels)}


def prediction(boxes, scores, labels):
    return {**target(boxes, labels), "scores": np.array(scores)}


def test_iou_and_degenerate_boxes():
    assert iou([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0] == pytest.approx(1 / 3)
    assert iou([[0, 0, 0, 0]], [[0, 0, 0, 0]])[0, 0] == 0
    assert iou([], []).shape == (0, 0)


def test_nms_ties_classes_and_exact_boundary():
    boxes = np.array([[0, 0, 10, 10]] * 3)
    np.testing.assert_array_equal(nms(boxes, [0.9] * 3, [0, 0, 1], 0.5), [0, 2])
    np.testing.assert_array_equal(nms(boxes, [0.9] * 3, [0, 0, 1], 1), [0, 1, 2])
    p = prediction([[0, 0, 0, 4], [0, 0, 4, 4]], [0.9, 0.5], [0, 0])
    assert len(postprocess(p)["scores"]) == 1


def test_ap_duplicates_missing_and_confidence_independence():
    t = [target([[0, 0, 10, 10], [20, 0, 30, 10]], [0, 0])]
    p = [prediction([[0, 0, 10, 10]] * 2 + [[20, 0, 30, 10]], [0.9, 0.8, 0.7], [0, 0, 0])]
    result = evaluate(p, t)
    assert result["map"] == pytest.approx((51 + 50 * 2 / 3) / 101)
    assert operating_point(result["curves"], 0.85) == dict(tp=1, fp=0, fn=1, precision=1, recall=0.5)
    assert operating_point(result["curves"], 0.65)["fp"] == 1
    assert evaluate(p, t)["map"] == result["map"]
    assert evaluate([empty_predictions()], t)["map"] == 0
    assert evaluate([empty_predictions()], [target([], [])])["map"] is None


def test_wrong_class_localization_and_threshold():
    t = [target([[0, 0, 10, 10]], [0])]
    assert evaluate([prediction([[0, 0, 10, 10]], [0.9], [1])], t)["map"] == 0
    p = [prediction([[0, 0, 5, 10]], [0.9], [0])]
    assert evaluate(p, t, 0.5)["map"] == 1
    assert evaluate(p, t, 0.51)["map"] == 0
    assert summary(p, t)["map50_95"] == pytest.approx(0.1)


def test_scenes_reproducible_disjoint_and_valid():
    for index in range(40):
        image, t = scene(index, "val")
        again, other = scene(index, "val")
        np.testing.assert_array_equal(image, again)
        np.testing.assert_array_equal(t["boxes"], other["boxes"])
        assert image.shape == (128, 128, 3)
        assert (t["boxes"][:, 2:] > t["boxes"][:, :2]).all()
        assert ((t["boxes"] >= 0) & (t["boxes"] <= 128)).all()
    assert not np.array_equal(scene(3, "train")[0], scene(3, "test")[0])
    assert len(scene(0, config=SceneConfig(max_shapes=0))[1]["boxes"]) == 0


def test_ap_matches_coco_on_nontrivial_no_crowd_fixture():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    targets = [target([[0, 0, 10, 10], [30, 30, 50, 50]], [0, 1]),
               target([[10, 10, 30, 30]], [0])]
    predictions = [prediction([[0, 0, 10, 10], [0, 0, 10, 10], [30, 30, 48, 48]],
                              [0.95, 0.8, 0.7], [0, 0, 1]),
                   prediction([[12, 12, 30, 30], [60, 60, 80, 80]], [0.6, 0.5], [0, 1])]
    annotations, results = [], []
    for image, (t, p) in enumerate(zip(targets, predictions)):
        for box, label in zip(t["boxes"], t["labels"]):
            x, y, x2, y2 = box
            annotations.append(dict(id=len(annotations) + 1, image_id=image, category_id=int(label),
                                    bbox=[x, y, x2 - x, y2 - y], area=(x2 - x) * (y2 - y), iscrowd=0))
        for box, score, label in zip(p["boxes"], p["scores"], p["labels"]):
            x, y, x2, y2 = box
            results.append(dict(image_id=image, category_id=int(label),
                                bbox=[x, y, x2 - x, y2 - y], score=float(score)))
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = dict(images=[dict(id=i) for i in range(2)], annotations=annotations,
                            categories=[dict(id=i, name=str(i)) for i in range(3)], info={})
        coco.createIndex()
        evaluator = COCOeval(coco, coco.loadRes(results), "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    ours = summary(predictions, targets)
    assert ours["map50_95"] == pytest.approx(precision[precision >= 0].mean())
    assert ours["map50"] == pytest.approx(precision[0][precision[0] >= 0].mean())
