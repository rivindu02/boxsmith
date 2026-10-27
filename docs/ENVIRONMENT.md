# Environment and development workflow

Checked on 2026-10-28. Resource availability can change on this shared server.

## Development locations

- Local SSHFS mount: `~/mnt/entc`.
- Mounted project directory: `~/mnt/entc/Rivindu/boxsmith`.
- Remote project directory relative to the SSH user's home: `~/Rivindu/boxsmith`.
- SSH host alias: `entc`.
- GitHub: <https://github.com/rivindu02/boxsmith> (public).

Edit and commit through the mounted project directory. Execute Python and training
on the server, where the GPUs and data reside. Do not run a second Git operation
on the same repository simultaneously over SSH and SSHFS.

```sh
cd ~/mnt/entc/Rivindu/boxsmith
git status
ssh entc
cd ~/Rivindu/boxsmith
```

The existing SSHFS mount is writable outside the restricted workspace sandbox.
No remount was necessary. A restricted agent may need elevated filesystem access
to write here because the mount is outside its writable workspace.

## Server inspection

| Check | Observation |
| --- | --- |
| OS | Linux 6.8.0-147-generic |
| Python | 3.12.3 |
| pip | 24.0 |
| Git | 2.43.0 |
| GPUs | 3 × Quadro RTX 5000, 16,384 MiB each |
| NVIDIA driver | 580.178.04 |
| Root/home free disk | Approximately 614 MB; filesystem reports 100% used |
| PyTorch package metadata | 2.13.0 |
| PyTorch import and CUDA check | Imports as 2.13.0+cu130; CUDA available; three devices visible |
| TorchVision package metadata | 0.28.0 |
| Other proposed dependencies | NumPy, Streamlit, Plotly, pytest not found in the inspected Python environment |

PyTorch imports and detects CUDA, but warns that NumPy is missing. Package metadata
alone does not prove a working training environment. Confirm remaining imports,
PyTorch/TorchVision compatibility, and a small model forward/backward pass in the
selected project environment before installation is considered complete.

Alternative mounts inspected: `/mnt/gpu-nfs-share` reports approximately 37 GB free
and `/mnt/minura` reports approximately 341 GB free. These are shared locations;
neither mount root is writable by the current SSH user. A project-specific writable
directory still needs to be identified; no directories were created on either mount.

At inspection, GPU 2 was almost fully occupied (~15.5 GB), while GPUs 0 and 1
each used ~0.5 GB. Recheck before choosing a device; do not assume availability.
No training jobs were started during setup.

## Before implementation

1. Identify sufficient writable storage; the current free space is insufficient
   for a normal fresh GPU environment. Alternative mounted paths still need
   capacity and permission checks before use.
2. Inspect existing compatible environments without modifying them. Create a
   project environment on the chosen storage and pin only versions actually tested.
3. Verify imports and a minimal forward/backward pass, then build data/metrics.

When the app exists, bind it to remote loopback and forward its port locally:

```sh
# On your local machine; run the app in another SSH session after implementation.
ssh -L 8501:127.0.0.1:8501 entc
# Future server command (app.py does not exist yet):
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Do not install dependencies or allocate model caches on the nearly full root
filesystem until storage is resolved. Generated data, caches, and model weights
belong outside version control.
