# Boxsmith implementation plan

First release implemented on 2026-10-28: data/metrics, the tiny CNN, baseline training,
interactive app, clean CPU setup, and validation are complete. See [results](RESULTS.md)
and [environment notes](ENVIRONMENT.md) for the measured implementation and setup.
The sections below record the original design and acceptance criteria.

## Goal and scope

Build an educational detector whose predictions and evaluation are inspectable.
The first release uses synthetic coloured shapes, a tiny CNN trained from scratch,
and a local Streamlit app served from the training server through an SSH tunnel.
Real-world datasets, cloud hosting, authentication, and distributed training are
outside the first release.

## Proposed user flow

1. Open a prepared validation scene or generate a reproducible scene from a seed.
2. Compare ground-truth boxes with the trained model's predictions.
3. Move the confidence slider and observe the chosen precision/recall point and
   visible detections.
4. Change class-aware NMS IoU to inspect duplicate removal and its effect on AP.
5. Change evaluation IoU to see when a localization becomes a true or false positive.
6. Inspect the full validation-set PR curve, per-class AP, and aggregate mAP.
7. Switch to controlled error examples to understand misses, duplicates, incorrect
   classes, background detections, and poorly localized boxes.

Keep single-scene counts distinct from dataset-level PR/AP. A single scene is an
explanation view; it is not a representative benchmark.

## Data

- Start with 128 × 128 RGB scenes and 0–6 shapes. Include empty scenes explicitly.
- Classes are circle, square, and triangle. Sample colours independently of class
  so colour alone cannot solve classification.
- Generate annotations directly from geometry; clip boxes to image bounds and
  validate positive area. Render without antialiasing initially so geometric and
  pixel bounds remain easy to reconcile.
- Use pixel `xyxy` boxes with exclusive right/bottom bounds at dataset boundaries;
  convert explicitly for model encoding. Record dimensions and class IDs.
- Use bounded overlap initially; add harder crowded/occluded scenes later. Keep
  the annotation policy explicit: boxes describe complete shapes before occlusion.
- Sample deterministic scenes on demand instead of saving thousands of images.
- Start with 4,000 training, 500 validation, and 500 test scenes, using distinct
  split seed namespaces and a recorded generator version/configuration.
- Tune on validation only; reserve test scenes for the final report. Training
  augmentations must never enter validation or test generation.

## Tiny detector

Implement a small one-stage anchor-based detector in PyTorch:

- A few convolution blocks, producing a stride-8 feature map (16 × 16 at 128 px).
- Three initial box priors per location, selected for the generated size range.
- Heads predict objectness, three class logits, and four encoded box offsets.
- Assign ground truth by anchor IoU, ensure every object has a positive anchor,
  resolve anchor conflicts deterministically, and ignore ambiguous anchors.
  Verify matching visually before attempting a full training run.
- Use binary objectness loss, positive-anchor class cross entropy, and positive
  box regression loss with bounded negative sampling. Record component losses.
- Define confidence as objectness probability × class probability. Decode, clip,
  reject invalid boxes, then apply class-aware NMS.
- Retain raw decoded candidates for every validation image before display filtering
  or NMS, so slider changes never require retraining or another model forward pass.

Target fewer than one million parameters; confirm the actual count after building.
Start with a small-batch CPU smoke run, then one available GPU. Device selection
must be explicit on the shared server. Use only one GPU for the initial model.
A TorchVision detector is an optional comparison after the tiny model works.

## Metrics contract

Keep three independent controls: display confidence, NMS IoU, and matching IoU.

1. Decode a fixed prediction set for the entire evaluation split. Use a documented
   low score floor, deterministic tie ordering, and a per-image detection cap.
2. Apply class-aware NMS with the selected NMS IoU. Use a deterministic CPU path
   for reproducible evaluation when scores tie.
3. For each class, rank predictions across images by decreasing score. Match each
   prediction to the highest-IoU unmatched ground truth of the same class in its
   own image, provided IoU meets the evaluation threshold. A ground truth can be
   matched only once. Duplicates count as false positives.
4. Compute cumulative precision and recall. Handle no predictions, no ground truth,
   empty images, degenerate boxes, and exact-threshold comparisons explicitly.
5. Use the precision envelope sampled at 101 recall levels for educational AP.
   An absent class has undefined AP and is excluded from the class mean; a class
   with ground truth and zero detections has AP = 0.
6. Report AP per class, mAP@0.50, and mAP averaged over IoUs 0.50, 0.55, …, 0.95.
   The latter is COCO-style: do not claim full COCO compliance until compared with
   pycocotools using identical caps, categories, area settings, and annotations.
7. The confidence slider changes visible boxes and the highlighted PR point.
   Reference AP always uses the same score floor and the full confidence sweep.
   If an advanced mode computes AP after confidence truncation, label it
   "AP of filtered predictions" and keep reference AP alongside it.
8. Changing NMS or matching IoU recomputes curves/AP from cached raw predictions.
   Changing confidence alone must leave reference AP unchanged.

Use `maxDets=100` per image after NMS for the reference view. Explain that a finite
candidate floor/cap limits measured recall. Store both settings with metric exports.

## Application architecture

Proposed files, to be created during implementation:

```text
app.py                         Streamlit views and controls
pyproject.toml                 package metadata and validated dependency versions
configs/tiny-shapes.yaml       reproducible baseline settings
src/boxsmith/
  data.py                     deterministic scenes and annotations
  model.py                    tiny CNN, heads, and box decoding
  matching.py                 anchor assignment for training
  train.py                    CLI, checkpoints, and training logs
  predict.py                  inference and raw-candidate cache
  boxes.py                    IoU, clipping, class-aware NMS
  metrics.py                  evaluation matching, PR, AP, mAP
  evaluate.py                 validation/test reports and exports
  visualization.py            box overlays and PR plots
tests/                        hand-computed metrics and integration checks
```

Cache the loaded model separately from prediction arrays. Prediction cache keys
must include checkpoint hash, generator version, split seeds, and decoding config.
Metric cache keys must also include NMS IoU, matching IoU, score floor, and cap.
Display confidence does not invalidate model inference or reference AP caches.
Bound cache sizes; keep inference in evaluation mode with gradients disabled.

Show an image panel, threshold controls, TP/FP/FN counts, a PR curve with the
selected operating point, and per-class/aggregate metrics. Offer ground-truth,
raw-candidate, post-NMS, and final-detection overlay toggles. Save scene seed,
thresholds, checkpoint identity, and metrics with JSON exports.

## Milestones and completion checks

### 0. Repository and environment — this setup

- Public GitHub repository, initial plan, ignore rules, and MIT license.
- Local commits through SSHFS; SSH connection and server resources checked.
- Document storage and environment limitations before installation.

### 1. Data and evaluation foundations

- Implement deterministic scenes, annotation visualization, IoU, NMS, and metrics.
- Hand-computed tests: perfect predictions, duplicates, wrong classes, localization
  errors, empty scenes, missing predictions, absent classes, ties, threshold edges.
- Verify confidence changes the selected point while preserving reference AP.
- Compare controlled no-crowd fixtures against pycocotools with identical settings.

### 2. Tiny CNN and training

- Implement anchor matching and verify coverage/conflicts on generated scenes.
- Overfit a tiny fixed dataset before a baseline run; verify finite losses and
  sensible decoded boxes. Record model size, settings, seed, and checkpoint.
- Run baseline training on one available GPU after storage/environment readiness.
- Evaluate against a random/untrained baseline. Aim for validation mAP@0.50 ≥ 0.85
  on easy scenes; this is a target to measure, not a promised result.

### 3. Interactive playground

- Build overlays, threshold controls, dataset curves, and explanatory examples.
- Use cached candidates so thresholds update without network/model inference.
- Target sub-second interactions on the 500-image validation set; measure actual
  timings and bound cache memory before declaring that target met.
- Smoke-check app launch, empty-scene behavior, and exported configuration.

### 4. Reproducibility and presentation

- Pin tested dependency versions; document setup, training, evaluation, and launch.
- Provide a compact checkpoint or release asset and a reproducible generation path.
- Report held-out test metrics, timings, model size, and known failure cases.
- Add screenshots and short explanations of IoU, NMS, PR, AP, and mAP.
- Verify a fresh environment can reproduce the workflow using documented commands.

## Storage and dependency policy

The server root filesystem currently has very little free space. Before installing
packages or training, identify a writable volume with sufficient capacity or have
space made available. Plan for at least 5 GB free if reusing an existing compatible
PyTorch install; a fresh CUDA/PyTorch environment may require substantially more.
Measure the actual environment footprint before downloading.

Prefer a project environment and a verified compatible existing PyTorch build when
possible. Do not modify system Python or another project's environment. Generate
scenes on demand and retain only best/latest checkpoints. Keep all environments,
datasets, caches, logs, and weights out of Git. Do not delete unrelated server files.

## Reference documentation

- [PyTorch detection tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
  for detection data interfaces and an optional comparison model.
- [TorchVision batched NMS](https://docs.pytorch.org/vision/main/generated/torchvision.ops.batched_nms.html)
  for class-aware suppression semantics.
- [COCO evaluation implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
  for 101-point interpolation and reference metric conventions.
- [Streamlit data caching](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data)
  for caching prediction arrays and metric results.
