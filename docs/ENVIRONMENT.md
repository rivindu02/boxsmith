# Environment and development workflow

Validated on 2026-10-28. This is a shared server; check current resources before training.

## Source and Git

- Local SSHFS project: `~/mnt/entc/Rivindu/boxsmith`.
- Remote project: `~/Rivindu/boxsmith` using `ssh entc`.
- Public repository: <https://github.com/rivindu02/boxsmith>.

All implementation commits were made from the local SSHFS directory. Python,
tests, training, and browser validation run on the server. Avoid simultaneous Git
operations through SSH and SSHFS. The existing mount is writable; restricted agents
need filesystem permission because it is outside their normal writable workspace.

## Working server environment

| Item | Validated value |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| NumPy | 2.5.4 |
| Pillow | 12.3.0 |
| Streamlit | 1.66.1 |
| Plotly | 7.1.0 |
| PyYAML | 6.0.1 |
| pytest / pycocotools | 9.1.1 / 2.0.11 |
| GPUs | 3 × Quadro RTX 5000, 16,384 MiB each |
| NVIDIA driver | 580.178.04 |
| Training device | cuda:0; other GPUs were occupied |
| Git | 2.43.0 |

The isolated training environment uses `--system-site-packages` to reuse the
existing CUDA PyTorch install. Missing project packages were installed into this
environment only; system Python and other projects were not modified.

TorchVision is not required by the implemented tiny model. Training and NMS are
implemented in the project, with deterministic CPU NMS during evaluation.

## Storage locations

The root/home filesystem has roughly 612 MB free and reports 100% utilization.
The larger `/mnt/minura` volume has roughly 341 GB free. Its sticky, world-writable
`tmp` directory allowed creation of an owned Boxsmith directory without modifying
another user's files. The shared NFS mount root was not writable.

| Purpose | Server path |
| --- | --- |
| CUDA training environment | `/mnt/minura/tmp/boxsmith-ravindu/venv` |
| Independent CPU environment | `/mnt/minura/tmp/boxsmith-ravindu/clean-env` |
| Baseline run | `/mnt/minura/tmp/boxsmith-ravindu/baseline` |
| Prediction caches | `/mnt/minura/tmp/boxsmith-ravindu/cache` |
| Browser validation tools | `/mnt/minura/tmp/boxsmith-ravindu/browsers` |

`checkpoints/best.pt` in the project is an ignored symlink to the baseline's best
checkpoint. Model weights are published as a release asset, not committed to Git.
Temporary-volume contents can be removed by server maintenance; the release and
documented generation commands provide a recovery path.

Generated scenes are kept on demand or cached in RAM during training. No dataset
image files are needed. Best/latest are the only retained per-run checkpoints.
Persistent inference caches retain at most six entries; Streamlit caches also
have entry limits. Dataset, weights, environments, and caches are ignored by Git.

## Run the existing setup

```sh
ssh entc
cd ~/Rivindu/boxsmith
export BOXSMITH_CACHE=/mnt/minura/tmp/boxsmith-ravindu/cache
/mnt/minura/tmp/boxsmith-ravindu/venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

From your local machine:

```sh
ssh -N -L 8501:127.0.0.1:8501 entc
```

Visit <http://localhost:8501>. If an app is already listening on 8501, use that
instance or choose another port for both the app and tunnel.

Train another baseline in a new output directory on the larger volume:

```sh
cd ~/Rivindu/boxsmith
/mnt/minura/tmp/boxsmith-ravindu/venv/bin/python -m boxsmith.train \
  --device cuda:0 --output /mnt/minura/tmp/boxsmith-ravindu/another-run
```

Recheck `nvidia-smi` first. Device selection is explicit; the initial run used only
GPU 0. No unrelated server files were deleted.

## Reproduce a clean CPU environment

`requirements-cpu.lock.txt` pins the complete tested Python 3.12 CPU dependency
set, including PyTorch's CPU build. It includes test dependencies. Run:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-cpu.lock.txt
python -m pip install --no-deps -e .
python -m boxsmith.fetch_checkpoint
python -m pytest -q
python -m pip check
```

A separate environment without system-site packages passed all 13 tests, dependency
checks, and CPU inference on the trained checkpoint. The CUDA environment also
passed all 13 tests and a real forward/backward check before training.

For CUDA, `requirements.txt` pins direct project dependencies without pinning a
particular CUDA wheel flavor. Install the appropriate PyTorch wheel first.
`requirements-dev.txt` adds the test dependencies. Do not install large environments
on the nearly full root disk.

Optional screenshot capture:

```sh
python -m pip install playwright
python -m playwright install chromium --only-shell
python scripts/capture_preview.py --output docs/preview.png
```

Use an external `PLAYWRIGHT_BROWSERS_PATH` on this server to keep browser downloads
off the root disk. Playwright is not needed to use Boxsmith or run its test suite.
