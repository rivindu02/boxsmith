# Baseline results and practical limits

## Training and selection

The first baseline uses a 173,536-parameter CNN with three anchors per 16 × 16
feature map location, trained from scratch on 128 × 128 images. The model has no
pretrained weights and no real-world dataset dependency.

- Generator version 1, seed 42, independent train/validation/test seed namespaces.
- 4,000 training scenes, 500 validation scenes, 500 test scenes; 0–6 objects per scene.
- Circles, squares, triangles; colours sampled independently of class.
- Batch size 64, AdamW, initial learning rate 0.002, cosine schedule, 32 epochs.
- GPU 0 of a Quadro RTX 5000 server; about 83 seconds for the epoch loop including
  periodic validation. Dataset preparation and the initial baseline evaluation
  are outside that epoch-loop timing.
- Checkpoint chosen by validation mAP@0.50 at epoch 24, then evaluated on test once.
- A 64-scene overfit diagnostic reached 96.28% mAP@0.50 on 32 training scenes;
  this diagnostic is not a held-out metric.
- Untrained validation mAP@0.50 was 0.0136%.

The original [configuration](../configs/tiny-shapes.yaml) and full
[training history](reports/training.jsonl) are included.

## Detection metrics

| Metric | Validation (500 scenes) | Test (500 scenes) |
| --- | ---: | ---: |
| mAP@0.50 | 99.8935% | 99.5708% |
| mAP@0.50:0.95 | 95.3292% | 95.5514% |
| Circle AP@0.50 | 99.8035% | 99.9866% |
| Square AP@0.50 | 99.8894% | 99.7402% |
| Triangle AP@0.50 | 99.9875% | 98.9855% |

NMS IoU = 0.50, score floor = 0.001, post-NMS cap = 100 per image. AP uses a
101-point interpolated precision envelope. Absent classes are excluded from mAP.
These are all-area, no-crowd, COCO-style metrics; the implementation agrees with
pycocotools on the included controlled fixture. It does not implement COCO's
full crowd/ignore/area-range machinery.

Machine-readable reports: [validation](reports/validation.json), [test](reports/test.json).
Both identify the generator, data split, model hash, epoch, and evaluation settings.

## Model artifact

The checkpoint is approximately 703 KiB and is provided with the
[v0.1.0 release](https://github.com/rivindu02/boxsmith/releases/tag/v0.1.0).

SHA-256:

```text
61adda73503b24aeaaf0db8466ce3ef38ec2872457f65e6d66ec42d6c6d541fc
```

The download command verifies this digest. Checkpoints are loaded using
`torch.load(..., weights_only=True)`.

## Validation performed

- 13 tests pass in both the CUDA-enabled project environment and an independent
  CPU-only environment. The clean CPU environment has no inherited system-site packages.
- Hand-calculated IoU/AP cases, duplicates, localization and class errors, empty
  scenes, missing predictions, absent classes, ties, and exact IoU thresholds.
- Comparison with pycocotools on a multi-image/multi-class fixture with duplicates
  and imperfect localization.
- CPU model backward pass, CUDA forward/backward smoke check, forced-anchor
  coverage for colliding objects, and box encoding/decoding consistency.
- Prediction cache round trips and invalidation by model bytes, split, and generator config.
- App interactions: confidence preserves reference AP; perfect controlled predictions
  yield 100% AP; missing checkpoints, trained model inference, and new empty scenes work.
- A real Chromium rendering of the app with the trained checkpoint completed with
  no browser JavaScript errors. The [preview](preview.png) is captured from that app.

## Interaction timings

Measured Streamlit AppTest reruns with the trained checkpoint and 500 validation
scenes in the independent CPU environment:

| Control change | Rerun time |
| --- | ---: |
| Confidence | 0.29 s |
| Evaluation IoU | 0.30 s |
| NMS IoU (new setting) | 1.58 s |

Confidence and evaluation IoU reuse cached predictions and curves. A new NMS
setting recomputes suppression and the ten-IoU sweep; this misses the original
sub-second target at 500 scenes. Smaller datasets and previously cached NMS
settings update faster. Timings include Streamlit's Python rerun but exclude
browser/network latency. Reproduce with `python scripts/benchmark_interactions.py`.
Raw measurements are in [interactions.json](reports/interactions.json).

## Limitations

This benchmark deliberately uses simple shapes, a fixed background, a small size
range, axis-aligned geometry, and bounded overlap. Its near-perfect mAP says
nothing about photographs, unfamiliar shapes, heavy occlusion, tiny objects,
crowded scenes, or background variation. These are useful next experiments.

At low display confidence, the model can retain incorrect classes or additional
overlapping boxes. Class-aware NMS preserves boxes from different classes, so
raising NMS suppression alone cannot remove every false positive. The playground
makes these errors visible even when reference AP is high.

Ranking ties are deterministic. Full COCO equivalence is not claimed for arbitrary
crowd/ignore annotations or category-specific maximum-detection conventions.
The finite score floor and detection cap can limit recall. Run timings depend on
shared-server load; GPU and CPU numerical results need not be bitwise identical.

Long-term storage under the server's temporary directory is not guaranteed. The
release checkpoint, pinned CPU environment, and generation commands provide a
reproducible path independent of that directory.
