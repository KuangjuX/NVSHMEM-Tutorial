# NVSHMEM‑Tutorial: Build a DeepEP‑like GPU Buffer

NVSHMEM‑Tutorial is a hands‑on guide to GPU‑to‑GPU communication with NVSHMEM. By building a simplified, DeepEP‑inspired Buffer, you will learn how to initialize NVSHMEM, allocate symmetric memory, perform one‑sided put/get, implement intra‑/inter‑node collectives, and engineer compute‑communication overlap.

DeepEP (by DeepSeek) is a high‑performance communication library for MoE and Expert‑Parallel. This tutorial does not reimplement DeepEP. Instead, it mirrors several core ideas in a minimal, readable form so you can understand and reproduce the techniques.


## Repository layout
- `csrc/` — C++/CUDA extension code (NVSHMEM bootstrap, primitives, kernels, bindings)
- `nvshmem_tutorial/` — Thin Python wrapper exposing `NvshmemBuffer`
- `tests/` — Unit/integration tests for intra‑ and inter‑node primitives
- `benchmarks/` — Performance benchmarks for communication primitives and end-to-end workflows
- `scripts/` — Convenience scripts for launching local multi‑GPU tests


## Install
```bash
# Editable install of the CUDA extension + Python package
pip install -e .
```


## Quickstart
Launch a 2‑GPU intra‑node test with torchrun (adjust `--nproc_per_node` as needed):
```bash
python -m torch.distributed.run --nproc_per_node=2 tests/test_intranode_nvshmem.py
```

Inter‑node runs follow the same pattern but require proper rendezvous environment (`MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`) and a fabric supported by NVSHMEM.

You can also use the convenience scripts in `scripts/`:
```bash
bash scripts/run_intranode_nvshmem.sh 2
bash scripts/run_internode_nvshmem.sh 0 # set rendezvous env first
```


## Performance

All benchmarks were conducted on NVIDIA H20 GPUs with NVLink connectivity:
- **NVLink Bidirectional Bandwidth**: 900 GB/s (theoretical)
- **NVLink Unidirectional Bandwidth**: 450 GB/s (theoretical)

### Point-to-Point Communication

| Type      | Data Size | NCCL P2P (GB/s) | CUDA IPC (GB/s) | NVSHMEM (GB/s) |
| --------- |-----------|-----------------|-----------------|----------------|
| Intranode | 16 KB     | 0.90            |    1.81         | 1.98           |
| Intranode | 256 KB    | 14.02           |    27.57        | 29.56          |
| Intranode | 1 MB      | 50.46           |    90.86        | 97.17          |
| Intranode | 16 MB     | 305.47          |    321.68       | 322.24         |
| Intranode | 64 MB     | 343.01          |    366.67       | 374.51         |
| Intranode | 256 MB    | 362.09          |    384.58       | 390.88         |
| Intranode | 1 GB      | 335.65          |    389.74       | 395.54         |
| Internode | 16 KB     | 0.49            |      X          | 0.58           |
| Internode | 256 KB    | 4.04            |      X          | 6.65           |
| Internode | 1 MB      | 10.81           |      X          | 14.60          |
| Internode | 16 MB     | 19.48           |      X          | 23.49          |
| Internode | 64 MB     | 19.65           |      X          | 24.21          |
| Internode | 256 MB    | 19.67           |      X          | 24.36          |
| Internode | 1 GB      | 19.67           |      X          | 24.21          |

### All Gather Communication

| Type      | Data Size | NCCL (GB/s)     | Hybrid (GB/s)   |
| --------- |-----------|-----------------|-----------------|
| Intranode | 4 KB      | 0.18            |    0.54         |
| Intranode | 16 KB     | 1.00            |    2.12         |
| Intranode | 64 KB     | 3.63            |    8.10         |
| Intranode | 256 KB    | 11.13           |    29.30        |
| Intranode | 1024 KB   | 53.61           |    73.43        |
| Intranode | 2 MB      | 76.51           |    87.27        |
| Intranode | 4 MB      | 107.46          |    92.57        |
| Intranode | 8 MB      | 170.90          |    99.19        |
| Intranode | 16 MB     | 210.44          |    93.75        |
| Intranode | 32 MB     | 230.63          |    92.16        |
| Intranode | 64 MB     | 239.52          |    93.24        |
| Internode | 4 KB      | 0.29            |    0.16         |
| Internode | 16 kB     | 0.92            |    0.62         |
| Internode | 64 KB     | 5.40            |    2.39         |
| Internode | 256 KB    | 21.23           |    8.10         |
| Internode | 1024 KB   | 63.34           |    17.41        |




## Acknowledgements
- DeepEP (DeepSeek): core inspiration for the buffer architecture and optimization ideas.


