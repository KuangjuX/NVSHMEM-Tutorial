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
| Intranode | 1 KB      | 0.06            |    0.12         | 0.13           | 
| Intranode | 16 KB     | 0.90            |    1.81         | 1.98           |
| Intranode | 64 KB     | 3.62            |    7.39         | 7.90           |
| Intranode | 256 KB    | 14.02           |    27.57        | 29.56          |
| Intranode | 1 MB      | 50.46           |    90.86        | 97.17          |
| Intranode | 4 MB      | 201.49          |    215.72       | 221.30         |
| Intranode | 16 MB     | 305.47          |    321.68       | 322.24         |
| Intranode | 64 MB     | 343.01          |    366.67       | 374.51         |
| Intranode | 256 MB    | 362.09          |    384.58       | 390.88         |
| Intranode | 1 GB      | 335.65          |    389,74       | 395.54         |




## Acknowledgements
- DeepEP (DeepSeek): core inspiration for the buffer architecture and optimization ideas.


