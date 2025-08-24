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
python -m pip install -e .

# Or use the provided Makefile
make build
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



## Acknowledgements
- DeepEP (DeepSeek): core inspiration for the buffer architecture and optimization ideas.


