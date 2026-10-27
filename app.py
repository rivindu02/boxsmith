"""Boxsmith: run with `streamlit run app.py`."""

import json
import os
from pathlib import Path
import time

import numpy as np
import streamlit as st
import torch

from boxsmith.boxes import postprocess
from boxsmith.data import CLASS_NAMES, SceneConfig, ShapesDataset, scene
from boxsmith.examples import SCENARIOS, controlled_predictions
from boxsmith.metrics import evaluate, operating_point, summary
from boxsmith.predict import (MAX_DETECTIONS, SCORE_FLOOR, cached_predictions,
                              file_hash, load_model, suppress)
from boxsmith.visualization import overlay, pr_figure

st.set_page_config(page_title="Boxsmith · Detection Lab", page_icon="◈", layout="wide")
torch.set_num_threads(4)
st.markdown("""<style>
.stApp {background: #0b1020;}
h1,h2,h3 {letter-spacing:-.035em;}
[data-testid="stMetric"] {background:#151e32;border:1px solid #263249;border-radius:12px;padding:16px;}
[data-testid="stSidebar"] {background:#10182a;}
.eyebrow {color:#38bdf8;font-size:12px;letter-spacing:.18em;font-weight:700;}
</style>""", unsafe_allow_html=True)

DEFAULT_CHECKPOINT = os.environ.get("BOXSMITH_CHECKPOINT", "checkpoints/best.pt")
CACHE = os.environ.get("BOXSMITH_CACHE", "cache")


@st.cache_resource(max_entries=2)
def model_resource(path, identity):
    return load_model(path)


@st.cache_data(max_entries=4, show_spinner=False)
def dataset_predictions(path, identity, count):
    _, metadata = model_resource(path, identity)
    dataset = ShapesDataset(count, "val", metadata["config"]["seed"], SceneConfig(**metadata["scene_config"]))
    return cached_predictions(path, dataset, CACHE)


@st.cache_data(max_entries=6, show_spinner=False)
def example_dataset(count, scenario):
    dataset = ShapesDataset(count, "val")
    targets = [dataset[i][1] for i in range(count)]
    predictions = [controlled_predictions(t, i, scenario) for i, t in enumerate(targets)]
    return predictions, targets, {"source": "controlled examples", "scenario": scenario, "dataset": dataset.identity()}


@st.cache_data(max_entries=8, show_spinner=False)
def metric_bundle(raw, targets, nms_iou, matching_iou):
    started = time.perf_counter()
    predictions = suppress(raw, nms_iou)
    selected = evaluate(predictions, targets, matching_iou)
    scores = summary(predictions, targets)
    return predictions, selected, scores, time.perf_counter() - started


with st.sidebar:
    st.markdown("### ◈ BOXSMITH")
    st.caption("An experiment in seeing clearly.")
    available = Path(DEFAULT_CHECKPOINT).is_file()
    source = st.selectbox("Prediction source", ["Trained tiny CNN", "Controlled examples"], index=0 if available else 1, key="prediction_source")
    checkpoint = st.text_input("Checkpoint path", DEFAULT_CHECKPOINT) if source == "Trained tiny CNN" else None
    scenario = st.selectbox("Error experiment", SCENARIOS, key="error_scenario") if source == "Controlled examples" else None
    count = st.select_slider("Validation scenes", options=[32, 100, 500], value=100)
    st.divider()
    confidence = st.slider("Confidence", 0.0, 1.0, 0.30, 0.01, key="confidence", help="Selects visible detections and the PR operating point. Reference AP uses the full score sweep.")
    nms_iou = st.slider("NMS IoU", 0.0, 1.0, 0.50, 0.05, key="nms_iou", help="Boxes of the same class suppress lower-scored boxes when their overlap exceeds this value.")
    matching_iou = st.slider("Evaluation IoU", 0.50, 0.95, 0.50, 0.05, key="matching_iou", help="Minimum overlap to match a prediction to ground truth.")
    st.caption("Three independent controls. Confidence selects a point; NMS and evaluation IoU can reshape the curve.")

st.markdown('<div class="eyebrow">COMPUTER VISION / INTERACTIVE LAB</div>', unsafe_allow_html=True)
st.title("Small shapes. Big detection ideas.")
st.markdown("Train a tiny CNN. Follow every box from prediction to precision–recall.")

if source == "Trained tiny CNN":
    if not Path(checkpoint).is_file():
        st.info("Choose a trained checkpoint, or switch to Controlled examples to explore metrics immediately.")
        st.code("python -m boxsmith.train --device cuda:0\n# Then set the checkpoint path to runs/baseline/best.pt")
        st.stop()
    try:
        identity = file_hash(checkpoint)
        with st.spinner("Loading validation predictions…"):
            raw, targets, run_identity = dataset_predictions(checkpoint, identity, count)
        model, metadata = model_resource(checkpoint, identity)
        scene_config = SceneConfig(**metadata["scene_config"])
        scene_seed = metadata["config"]["seed"]
        st.caption(f"Trained CNN · {metadata['parameter_count']:,} parameters · epoch {metadata['epoch']} · validation split")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        st.error(f"Unable to load this checkpoint: {error}")
        st.stop()
else:
    raw, targets, run_identity = example_dataset(count, scenario)
    scene_config, scene_seed = SceneConfig(), 42
    st.info("Controlled examples: these predictions illustrate known errors and are not outputs of a trained model.")

with st.spinner("Evaluating detections…"):
    predictions, result, scores, elapsed = metric_bundle(raw, targets, nms_iou, matching_iou)
point = operating_point(result["curves"], confidence)


def percentage(value):
    return "—" if value is None else f"{value:.1%}"


cards = st.columns(4)
for column, label, value in zip(cards,
                               ["Precision at confidence", "Recall at confidence", "Reference mAP@0.50", "Reference mAP@0.50:0.95"],
                               [point["precision"], point["recall"], scores["map50"], scores["map50_95"]]):
    column.metric(label, percentage(value))

playground, concepts, training = st.tabs(["Detection playground", "How the metrics work", "Training & reproducibility"])
with playground:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("Inspect a scene")
        mode = st.radio("Scene source", ["Validation scene", "New seeded scene"], horizontal=True)
        if mode == "Validation scene":
            first_nonempty = next((i for i, t in enumerate(targets) if len(t["boxes"])), 0)
            index = int(st.number_input("Validation index", 0, count - 1, first_nonempty, step=1))
            image, target = scene(index, "val", scene_seed, scene_config)
            scene_raw, scene_predictions = raw[index], predictions[index]
            scene_split = "val"
        else:
            index = int(st.number_input("Scene seed index", 0, 100000, 7, step=1))
            image, target = scene(index, "scene", scene_seed, scene_config)
            scene_split = "scene"
            if source == "Trained tiny CNN":
                with torch.inference_mode():
                    tensor = torch.from_numpy(image).permute(2, 0, 1).float()[None] / 255
                    scene_raw = model.decode(model(tensor))[0]
            else:
                scene_raw = controlled_predictions(target, index, scenario, scene_config.image_size)
            scene_predictions = postprocess(scene_raw, nms_iou)
        stage = st.selectbox("Prediction layer", ["Final detections", "After NMS", "Raw candidates", "Ground truth only"])
        show_truth = st.checkbox("Show ground truth (green)", value=True)
        selected_prediction = scene_raw if stage == "Raw candidates" else scene_predictions
        display_floor = confidence if stage in ("Final detections", "Raw candidates") else SCORE_FLOOR
        if stage == "Ground truth only":
            selected_prediction = None
        st.image(overlay(image, target if show_truth or stage == "Ground truth only" else None,
                         selected_prediction, display_floor), width="stretch")
        scene_result = evaluate([scene_predictions], [target], matching_iou)
        scene_point = operating_point(scene_result["curves"], confidence)
        st.caption(f"This scene: {scene_point['tp']} TP · {scene_point['fp']} FP · {scene_point['fn']} missed objects. "
                   "Raw overlays show at most 100 boxes at the display confidence.")
    with right:
        st.subheader("Follow the precision–recall curve")
        st.caption(f"Entire validation set · {count} scenes · matching IoU {matching_iou:.2f}. Dots mark the confidence setting.")
        st.plotly_chart(pr_figure(result["curves"], confidence), width="stretch")
        st.dataframe([{"Class": name, "Ground truth": curve["ground_truth_count"], "AP at selected IoU": percentage(curve["ap"])}
                      for name, curve in zip(CLASS_NAMES, result["curves"])], hide_index=True, width="stretch")
        st.caption(f"Dataset totals at confidence {confidence:.2f}: {point['tp']} TP · {point['fp']} FP · {point['fn']} FN. "
                   f"Uncached evaluation: {elapsed:.2f}s. Reference AP stays fixed when only confidence changes.")
        export = {"identity": run_identity, "confidence": confidence, "nms_iou": nms_iou,
                  "matching_iou": matching_iou, "score_floor": SCORE_FLOOR, "max_detections": MAX_DETECTIONS,
                  "scene": {"split": scene_split, "index": index, "seed": scene_seed}, "scene_counts": scene_point,
                  "dataset_operating_point": point, **scores,
                  "ap_at_selected_iou": {name: c["ap"] for name, c in zip(CLASS_NAMES, result["curves"])}}
        st.download_button("↓ Export experiment JSON", json.dumps(export, indent=2), "boxsmith-experiment.json", "application/json")

with concepts:
    st.subheader("One box, three decisions")
    st.markdown("""
**Confidence** ranks model predictions. Raising it usually removes weak detections;
it moves the highlighted operating point along each curve. It does not change reference AP.

**NMS IoU** controls duplicate removal. Overlapping boxes of the same class compete;
the lower-scored box is suppressed if IoU exceeds the NMS threshold. A higher threshold
keeps more overlapping boxes. Different classes do not suppress one another.

**Evaluation IoU** decides whether a surviving prediction is well localized enough
to match a ground-truth object of the same class. Each object can be matched once.
Duplicates, wrong classes, and unmatched boxes are false positives; unmatched objects are misses.

**IoU** is intersection area divided by union area. **Precision** is TP / (TP + FP);
**recall** is TP / (TP + FN). A zero denominator is displayed as zero for operating points.

**AP** averages the precision envelope at 101 recall levels over a full confidence sweep.
**mAP@0.50** averages AP across classes with ground truth at IoU 0.50.
**mAP@0.50:0.95** also averages across ten matching IoUs. Absent classes show undefined AP.

These are educational COCO-style, all-area, no-crowd metrics. Candidate score floor is
0.001 and the post-NMS cap is 100 detections per image; finite floors/caps can limit recall.
Single-scene counts explain a scene; the curves and reference metrics use the full selected dataset.
""")
    st.markdown("Try **Duplicates** with NMS IoU 1.0, **Wrong classes**, and **Loose boxes** in Controlled examples.")
with training:
    st.subheader("A detector you can reproduce")
    st.markdown("Scenes are generated from split-specific seeds. Colour varies independently of shape class. "
                "The tiny CNN predicts objectness, class, and offsets for three anchors at each feature location. "
                "Validation scenes guide development; the test split is reserved for the final report.")
    st.code("python -m boxsmith.train --config configs/tiny-shapes.yaml --device cuda:0\n"
            "python -m boxsmith.evaluate runs/baseline/best.pt --split test --count 500\n"
            "streamlit run app.py --server.address 127.0.0.1")
    st.markdown("Source and setup: [github.com/rivindu02/boxsmith](https://github.com/rivindu02/boxsmith)")
