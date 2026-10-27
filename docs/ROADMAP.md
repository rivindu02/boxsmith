# Improvement review and roadmap

Reviewed against the v0.1.0 implementation on 2026-10-28. This is a prioritized
backlog: items below describe work still to do, not features already implemented.
The current release works for the documented single-user synthetic-shape workflow.

## Recommended order

| Order | Improvement | Why it matters | Completion check |
| --- | --- | --- | --- |
| 1 | Make prediction caches safe under concurrent requests | Two sessions can write the same temporary filename | Concurrent writers/readers complete without corrupting or losing a cache entry |
| 2 | Validate checkpoint format, generator version, and training settings | Invalid inputs can produce an exception or misleading evaluation context | Bad inputs produce an actionable error before a run starts; incompatible generators are rejected |
| 3 | Add harder, separately reported scene benchmarks | Current near-perfect scores cover an intentionally narrow distribution | Publish held-out results for background, scale, overlap, and colour shifts |
| 4 | Explain each prediction's evaluation status in the image | Aggregate counts do not show which box was wrong or why | Selecting a box reveals its class, matched object, IoU, and TP/FP reason |
| 5 | Reduce uncached NMS rerun latency | A new NMS setting takes about 1.58 s on 500 scenes | Preserve reference metrics and stable ties while reaching measured sub-second reruns |
| 6 | Add resumable training and stronger provenance | Interrupted runs must currently restart | A saved run resumes with model, optimizer, scheduler, RNG, and configuration restored |
| 7 | Expand regression coverage and packaging checks | The 13 tests cover the core path, not all malformed inputs or platforms | Cover races, malformed checkpoints/configs, empty training sets, and release installation |

## 1. Shared cache robustness

Evidence: [predict.py](../src/boxsmith/predict.py), `cached_predictions`, writes to
one deterministic `<cache-key>.tmp.npz` path, then replaces the final path. Two
processes with the same cache key can collide at the write or replace step.
Eviction also lists, stats, and deletes files without coordinating with other writers.
This is a source-inspection finding; the current tests do not exercise concurrency.

Use a unique temporary file in the destination directory, publish it atomically,
clean up only that writer's temporary file, and tolerate files disappearing during
eviction. Add locking or another coordination mechanism where necessary. Handle
an incomplete or invalid cache by rebuilding it instead of failing the app.
Include a multi-process regression test; same-process caching alone is insufficient.

The cache identity includes the checkpoint hash, dataset identity, and decoder
version, which is a good foundation. Keep those guarantees when changing storage.
Python's [temporary-file documentation](https://docs.python.org/3.12/library/tempfile.html)
describes unique temporary files and their lifecycle.

## 2. Input validation and honest evaluation context

Evidence: [train.py](../src/boxsmith/train.py), [predict.py](../src/boxsmith/predict.py),
[evaluate.py](../src/boxsmith/evaluate.py), and [app.py](../app.py).

- **All-empty training configuration:** `max_shapes: 0` is a valid scene config and
  no-ground-truth mAP is intentionally `None`. Training compares that value with
  numeric `best`, which raises `TypeError`. Reject an all-empty training task or
  explicitly support it with a different checkpoint-selection policy.
- **Invalid YAML settings:** validate the full schema, positive batch size and
  evaluation frequency, nonnegative seed, valid threads, and finite positive
  learning rate. Currently only a subset is checked before execution.
- **Checkpoint compatibility:** saved `generator_version` is not checked when the
  app/evaluator creates current-generation scenes. A future generator change could
  silently change the benchmark. Validate a schema/generator version before use,
  or retain a versioned generation path.
- **Malformed checkpoint handling:** a non-checkpoint file raises
  `pickle.UnpicklingError` in the checked environment. The app's loader error handler
  catches several exceptions but not this one. Wrap recognized load/validation
  failures in an actionable project exception; preserve `weights_only=True`.
- **Overfit diagnostics:** training records `evaluation_split: train`, but labels
  log results as `val_map50`; the app always says validation split. Surface that
  diagnostic status and distinguish it from a normal held-out run.

Reproduced during review: empty-target evaluation returns `None`, malformed
checkpoint loading raises `UnpicklingError`, and a one-epoch all-empty training run
fails at `record["val_map50"] > best`. These are edge-case findings; the released
baseline uses nonempty, valid scene settings and a verified checkpoint.

## 3. Generalization benchmarks

Evidence: [data.py](../src/boxsmith/data.py) uses one dark background, five colours,
equal-width/height shape bounds, a 16–36 pixel default size range, and bounded overlap.
[Existing results](RESULTS.md) explicitly describe this narrow task.

Add controlled difficulty presets for variable backgrounds, noise, smaller/larger
objects, rotation, crowding, and occlusion. Version the generator/configuration and
retain the original easy benchmark as a regression check. Keep train/validation/test
namespaces separate and report each difficulty separately. Run multiple training
seeds to measure variance rather than presenting one seed as a general guarantee.

A larger detector or a real-photo dataset should come after these controlled
benchmarks. Better test coverage of the existing learning goal is more informative
than simply replacing the tiny model.

## 4. Teaching and interface improvements

Evidence: [visualization.py](../src/boxsmith/visualization.py) colours predictions by
class and draws green ground truth. [metrics.py](../src/boxsmith/metrics.py) returns
TP flags for PR calculations, but does not expose match IDs or per-detection reasons.

Return structured match details from the evaluator and reuse them for an optional
TP/FP/FN overlay, box inspector, and matched-IoU table. Explain the differences
between duplicate detections, class errors, localization failures, and background
false positives. Avoid performing a different matching algorithm just for the UI.

Add import/replay of the existing experiment JSON and shareable settings if useful.
The current download records settings but does not restore them in the app. Add
keyboard/mobile checks and ensure status is conveyed by text as well as colour.

## 5. Performance

Evidence: [recorded interaction timings](reports/interactions.json): confidence
0.29 s, matching IoU 0.30 s, and a new NMS IoU 1.58 s for 500 validation scenes.

Profile NMS and metric matching separately on both easy and crowded scenes. Remove
duplicate work in the CLI's `summary` plus AP@0.50 evaluation, avoid constructing
image tensors merely to regenerate targets on cache hits, and consider a compiled
NMS implementation only if it preserves the documented tie and threshold behavior.
Record browser-visible timings as well as Python rerun timings. Keep numerical
parity fixtures around any optimization.

## 6. Training and reproducibility

The training checkpoint currently contains model parameters/configuration and
epoch information, not optimizer/scheduler state or a resumable RNG snapshot.
Save resumable state separately from the compact inference release checkpoint.
Record source revision, dependency versions, device details, and dataset identity.
Publish multiple-seed comparisons if making stronger model-quality claims.

The existing server environments and run artifacts live under a temporary volume.
Move active environments/runs to a durable, writable location when one is available.
The public release checkpoint already provides a recovery path for inference.

## 7. Tests and package portability

Add regressions for the issues above, broader random/no-crowd pycocotools parity
fixtures, successful release download/loading, and checkpoint cache eviction under
concurrent access. Add a short training CLI smoke test and validate wheel installation
from outside the repository root. Keep browser checks small and purposeful.

The complete CPU lock was validated on Linux x86-64 with Python 3.12. Explicitly
test other supported platforms before claiming portability. The broad lower
dependency bounds in `pyproject.toml` should be checked against APIs actually used
or tightened to tested compatible versions.

## Completed in this review pass

- A more useful README with an illustrated header, architecture diagram, guided
  experiments, setup choices, evaluation examples, and troubleshooting.
- Relevant GitHub repository topics for discovery.
- This evidence-based improvement backlog with completion checks.

The backlog does not alter the published model, metrics, or release artifact.
