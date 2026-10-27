# Boxsmith

**Forge detections. Understand every AP point.**

A working object detection and mAP playground. Train a tiny CNN from scratch on
colourful shapes, inspect its boxes, and explore how confidence, NMS, and evaluation
IoU affect precision, recall, and average precision.

![Boxsmith running with the trained tiny CNN](docs/preview.png)

## Explore

- Reproducible circles, squares, and triangles with automatically generated boxes.
- A **173,536-parameter CNN**, trained from scratch without pretrained weights.
- Ground truth, raw candidates, post-NMS boxes, and confidence-filtered detections.
- Independent confidence, NMS IoU, and evaluation IoU controls.
- Dataset precision–recall curves, per-class AP, mAP@0.50, and mAP@0.50:0.95.
- Controlled duplicate, missed-object, wrong-class, and localization experiments.
- Seeded new scenes, TP/FP/FN counts, and JSON experiment downloads.

Confidence selects a PR operating point. **Reference AP uses the full confidence
sweep and stays fixed when only the confidence slider changes.** NMS can change
the curve and AP; evaluation IoU changes what counts as a correct localization.

## Run locally with CPU

Use Python 3.12. The public checkpoint is about 703 KiB; a GPU is optional for
inference. These commands create a complete environment with pinned CPU packages:

```sh
git clone https://github.com/rivindu02/boxsmith.git
cd boxsmith
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-cpu.lock.txt
python -m pip install --no-deps -e .
python -m boxsmith.fetch_checkpoint
python -m streamlit run app.py --server.address 127.0.0.1
```

Open <http://localhost:8501>. Without a checkpoint, the app opens in **Controlled
examples** mode so the metric experiments work immediately.

The [v0.1.0 release](https://github.com/rivindu02/boxsmith/releases/tag/v0.1.0)
contains the baseline checkpoint. The downloader verifies its pinned SHA-256.

## Train and evaluate

For a GPU environment, install an appropriate PyTorch build first, then install
`requirements-dev.txt` and the package. The existing server environment is
documented in [environment notes](docs/ENVIRONMENT.md).

```sh
python -m boxsmith.train --config configs/tiny-shapes.yaml --device cuda:0
python -m boxsmith.evaluate runs/baseline/best.pt --split val --count 500
python -m boxsmith.evaluate runs/baseline/best.pt --split test --count 500
BOXSMITH_CHECKPOINT=runs/baseline/best.pt python -m streamlit run app.py
```

Use `--device cpu` to train without a GPU. Training retains best/latest checkpoints
and refuses to overwrite an existing run; choose a fresh `--output` directory.
`--overfit --train-count 64 --val-count 32 --epochs 160` is a diagnostic that
evaluates training scenes, not a held-out performance report.

## Measured baseline

4,000 training scenes, seed 42; checkpoint selected by validation mAP@0.50 at epoch
24 of a 32-epoch run. Test scenes were not used to select it.

| Split | Scenes | mAP@0.50 | mAP@0.50:0.95 |
| --- | ---: | ---: | ---: |
| Validation | 500 | 99.89% | 95.33% |
| Test | 500 | 99.57% | 95.55% |

These are **synthetic-shape results**, not real-world detection performance.
See the [results and limitations](docs/RESULTS.md), machine-readable
[reports](docs/reports), and [implementation plan](docs/PLAN.md).

## Develop on the server through SSHFS

Source: `~/mnt/entc/Rivindu/boxsmith` locally, `~/Rivindu/boxsmith` on `entc`.
Edit and commit through SSHFS; run training and the app over SSH.

```sh
ssh entc
cd ~/Rivindu/boxsmith
/mnt/minura/tmp/boxsmith-ravindu/venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

From a second local terminal, forward the app port:

```sh
ssh -N -L 8501:127.0.0.1:8501 entc
```

The root disk is nearly full, so environments, training outputs, and caches use
the larger volume. See [environment notes](docs/ENVIRONMENT.md) for the exact paths.

## Verify

```sh
python -m pytest -q
python -m pip check
```

The suite checks hand-calculated AP, parity with `pycocotools` on a nontrivial
no-crowd fixture, deterministic data, IoU/NMS edge cases, anchor matching,
forward/backward execution, cache identity, verified downloads, and Streamlit
interactions with both controlled predictions and a model checkpoint.

GitHub Actions runs the CPU suite. Optional browser capture and timing scripts
live in `scripts/`; browser capture additionally requires Playwright and Chromium.

## License

[MIT](LICENSE).
