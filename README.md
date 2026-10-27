# Boxsmith

**Forge detections. Understand every AP point.**

An object detection and mAP playground: train a tiny CNN on colourful synthetic
shapes, adjust confidence and non-maximum suppression (NMS), and see how boxes,
precision, recall, and average precision relate.

**Status: planning and repository setup. The detector and app are not implemented yet.**

## The experience

- Generate scenes of circles, squares, and triangles with automatic bounding boxes.
- Train a small detector from scratch, with reproducible dataset seeds.
- Inspect ground truth, raw candidates, suppressed boxes, and final detections.
- Adjust confidence, NMS IoU, and evaluation IoU independently.
- Explore precision–recall curves, AP per class, mAP@0.50, and mAP@0.50:0.95.
- Explain true positives, duplicate detections, false positives, and missed objects.

Confidence selects an operating point on a precision–recall curve. Reference AP
uses a confidence sweep over a fixed candidate set; the display confidence slider
does not silently redefine that metric. NMS and evaluation IoU can change AP.

## Proposed stack

Python 3.12, PyTorch, NumPy/Pillow for synthetic data, Streamlit for the playground,
Plotly for curves, and pytest for metric and pipeline checks. Exact compatible
versions will be pinned after the server environment is validated.

## Plan and setup

- [Implementation plan](docs/PLAN.md): architecture, metric definitions, milestones,
  and acceptance checks.
- [Environment notes](docs/ENVIRONMENT.md): development workflow, server checks,
  and storage constraints.

Development files live in `~/mnt/entc/Rivindu/boxsmith` through SSHFS. Run Python,
training, and the eventual app on the server using `ssh entc`.

No dependencies, model weights, or datasets have been installed or generated as
part of this initial setup. See the plan for the next implementation step.

## Learning goals

CNN detection heads, box regression, intersection over union (IoU), confidence
ranking, class-aware NMS, matching predictions to ground truth, precision–recall,
AP, and mAP.

## License

[MIT](LICENSE).
