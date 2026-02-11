<p align="center">
  <h1 align="center">NVSHMEM-Tutorial</h1>
  <p align="center">
    <em>Build a DeepEP-like GPU Communication Buffer from Scratch</em>
  </p>
</p>

<p align="center">
  <a href="https://github.com/KuangjuX/NVSHMEM-Tutorial/stargazers">
    <img src="https://img.shields.io/github/stars/KuangjuX/NVSHMEM-Tutorial?style=social" alt="GitHub Stars">
  </a>
  <a href="https://github.com/KuangjuX/NVSHMEM-Tutorial/issues">
    <img src="https://img.shields.io/github/issues/KuangjuX/NVSHMEM-Tutorial" alt="GitHub Issues">
  </a>
  <a href="https://github.com/KuangjuX/NVSHMEM-Tutorial/pulls">
    <img src="https://img.shields.io/github/issues-pr/KuangjuX/NVSHMEM-Tutorial" alt="Pull Requests">
  </a>
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/CUDA-12.x-green" alt="CUDA 12.x">
  <img src="https://img.shields.io/badge/arch-sm__90a%20(Hopper)-orange" alt="SM90a">
</p>

---

NVSHMEM-Tutorial is a hands-on guide to GPU-to-GPU communication with **NVSHMEM**. By building a simplified, [DeepEP](https://github.com/deepseek-ai/DeepEP)-inspired buffer, you will learn how to:

- **Initialize** NVSHMEM with Unique ID bootstrapping via `torch.distributed`
- **Allocate** symmetric memory across multiple GPUs
- **Perform** one-sided `put`/`get` operations without target GPU involvement
- **Implement** intra-node (NVLink/IPC) and inter-node (RDMA) all-gather collectives
- **Engineer** compute-communication overlap with async operations
- **Leverage** Hopper TMA (Tensor Memory Access) for efficient data movement

> [!NOTE]
> [DeepEP](https://github.com/deepseek-ai/DeepEP) (by DeepSeek) is a high-performance communication library for MoE and Expert-Parallel workloads. This tutorial does **not** reimplement DeepEP — it mirrors several core ideas in a minimal, readable form so you can understand and reproduce the techniques.

## Table of Contents

- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Benchmarks](#benchmarks)
  - [Test Environment](#test-environment)
  - [Point-to-Point Communication](#point-to-point-communication)
  - [All-Gather Communication](#all-gather-communication)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Repository Layout

```
NVSHMEM-Tutorial/
├── csrc/                        # C++/CUDA extension
│   ├── buffer.cu / buffer.cuh   #   NvshmemBuffer core (alloc, sync, IPC)
│   ├── intranode.cu             #   Intra-node collectives (NVLink/IPC)
│   ├── internode.cu             #   Inter-node collectives (NVSHMEM/RDMA)
│   ├── put.cu / put.cuh         #   One-sided put primitives
│   ├── kernels/
│   │   └── copy.cu / copy.cuh   #   Hopper TMA copy kernels
│   ├── pybind.cu                #   PyBind11 bindings
│   └── *.cuh                    #   PTX wrappers, sync, sym helpers
├── nvshmem_tutorial/            # Python package
│   ├── __init__.py              #   Public API surface
│   └── buffer.py                #   NvshmemBuffer Python wrapper
├── benchmarks/                  # Performance benchmarks
│   ├── bench_p2p.py             #   Point-to-point: NVSHMEM vs NCCL vs CUDA IPC
│   └── bench_allgather.py       #   All-gather: NVSHMEM (hybrid) vs NCCL
├── tests/                       # Integration tests
│   ├── test_intranode_nvshmem.py
│   ├── test_internode_nvshmem.py
│   ├── test_intranode_allgather.py
│   ├── test_internode_allgather.py
│   └── test_tma_copy.py
├── scripts/                     # Launch helpers
│   ├── install.sh
│   ├── run_bench_intranode_p2p.sh
│   ├── run_bench_internode_p2p.sh
│   ├── run_bench_intranode_allgather.sh
│   └── run_bench_internode_allgather.sh
├── docs/                        # Tutorial documents
│   ├── 00-introduction.md       #   NVSHMEM concepts & memory model
│   └── 01-initialization.md     #   Bootstrap & team creation
└── setup.py                     # Build configuration
```

## Prerequisites

| Dependency | Version |
|------------|---------|
| Python     | ≥ 3.8   |
| PyTorch    | ≥ 2.0 (with CUDA support) |
| CUDA Toolkit | 12.x |
| NVSHMEM    | 2.x or 3.x (set `NVSHMEM_HOME`, default `/opt/nvshmem`) |
| GPU Architecture | NVIDIA Hopper (sm_90a), e.g. H100 / H20 |
| InfiniBand (inter-node) | Mellanox ConnectX / IBGDA-capable HCA |

## Installation

```bash
# 1. Make sure NVSHMEM is installed and NVSHMEM_HOME is set
export NVSHMEM_HOME=/opt/nvshmem  # adjust to your path

# 2. Editable install of the CUDA extension + Python package
pip install -e .

# Or use the convenience script
bash scripts/install.sh
```

## Quick Start

### Intra-node (single machine, multi-GPU)

Launch a 2-GPU intra-node test with `torchrun`:

```bash
torchrun --nproc_per_node=2 tests/test_intranode_nvshmem.py
```

Or use the convenience script:

```bash
bash scripts/run_intranode_nvshmem.sh 2
```

### Inter-node (multi-machine)

Inter-node runs require proper rendezvous environment variables (`MASTER_ADDR`, `MASTER_PORT`):

```bash
# On each node (adjust --node_rank accordingly)
export MASTER_ADDR=<master-ip>
export MASTER_PORT=29500

bash scripts/run_internode_nvshmem.sh 0   # node 0
bash scripts/run_internode_nvshmem.sh 1   # node 1
```

## Benchmarks

### Test Environment

All benchmarks were conducted with the following hardware and software configuration:

| Item | Specification |
|------|---------------|
| GPU  | NVIDIA H20 × 8 per node |
| Intra-node Interconnect | NVLink (900 GB/s bidirectional, 450 GB/s unidirectional) |
| Inter-node Interconnect | InfiniBand (8× Mellanox HCAs) |
| Warmup Iterations | 10 |
| Benchmark Iterations | 50–100 |

### Point-to-Point Communication

Bandwidth comparison of **NCCL Send/Recv**, **CUDA IPC** (NvshmemBuffer intra-node send/recv), and **NVSHMEM Put** for various data sizes.

#### Intra-node P2P

| Data Size | NCCL P2P (GB/s) | CUDA IPC (GB/s) | NVSHMEM (GB/s) |
|-----------|:----------------:|:----------------:|:--------------:|
| 16 KB     | 0.90             | 1.81             | **1.98**       |
| 256 KB    | 14.02            | 27.57            | **29.56**      |
| 1 MB      | 50.46            | 90.86            | **97.17**      |
| 16 MB     | 305.47           | 321.68           | **322.24**     |
| 64 MB     | 343.01           | 366.67           | **374.51**     |
| 256 MB    | 362.09           | 384.58           | **390.88**     |
| 1 GB      | 335.65           | 389.74           | **395.54**     |

> **Key takeaway:** NVSHMEM consistently outperforms NCCL P2P by **1.1×–2.2×** across all data sizes, reaching **395.54 GB/s** (≈88% of theoretical NVLink unidirectional bandwidth) at 1 GB. CUDA IPC is close but NVSHMEM still edges ahead due to its optimized one-sided put semantics.

#### Inter-node P2P

| Data Size | NCCL P2P (GB/s) | NVSHMEM (GB/s) |
|-----------|:----------------:|:--------------:|
| 16 KB     | 0.49             | **0.58**       |
| 256 KB    | 4.04             | **6.65**       |
| 1 MB      | 10.81            | **14.60**      |
| 16 MB     | 19.48            | **23.49**      |
| 64 MB     | 19.65            | **24.21**      |
| 256 MB    | 19.67            | **24.36**      |
| 1 GB      | 19.67            | **24.21**      |

> **Key takeaway:** Inter-node NVSHMEM achieves up to **1.6×** higher bandwidth than NCCL at small message sizes and maintains a **~24%** advantage at large transfers. CUDA IPC is not applicable (marked ✗) for inter-node scenarios.

### All-Gather Communication

Bandwidth comparison of **NCCL `all_gather_into_tensor`** and **Hybrid NVSHMEM** (NvshmemBuffer-based) all-gather.

> The "Hybrid" approach uses CUDA IPC for intra-node data movement and NVSHMEM RDMA for inter-node transfers.

#### Intra-node All-Gather (8 GPUs)

| Data Size (total) | NCCL (GB/s) | Hybrid (GB/s) |
|-------------------|:-----------:|:-------------:|
| 4 KB              | **1.02**    | 0.18          |
| 16 KB             | **2.48**    | 0.74          |
| 64 KB             | **13.87**   | 2.96          |
| 256 KB            | **57.33**   | 11.71         |
| 1 MB              | **126.60**  | 41.55         |
| 2 MB              | **198.55**  | 73.40         |
| 4 MB              | **234.00**  | 118.41        |
| 8 MB              | **280.46**  | 172.19        |
| 16 MB             | **308.44**  | 188.47        |
| 32 MB             | **328.32**  | 204.16        |
| 64 MB             | **336.90**  | 216.75        |
| 128 MB            | **345.00**  | 225.75        |
| 256 MB            | **352.18**  | 230.39        |
| 512 MB            | **356.45**  | 232.82        |
| 1 GB              | **359.75**  | 231.51        |

> **Key takeaway:** For pure intra-node all-gather, NCCL's highly optimized ring/tree algorithms outperform the hybrid approach. The hybrid method reaches **~65%** of NCCL throughput at large sizes — it is designed primarily for **multi-node** scenarios where the NVSHMEM RDMA path provides an advantage.

#### Inter-node All-Gather (2 nodes × 8 GPUs)

| Data Size (total) | NCCL (GB/s) | Hybrid (GB/s) |
|-------------------|:-----------:|:-------------:|
| 4 KB              | **0.29**    | 0.16          |
| 16 KB             | **0.92**    | 0.62          |
| 64 KB             | **5.40**    | 2.39          |
| 256 KB            | **21.23**   | 8.10          |
| 1 MB              | **63.34**   | 17.41         |

> **Key takeaway:** At the current implementation stage, NCCL retains an advantage in inter-node all-gather. The hybrid approach is a work-in-progress — further optimizations (pipelining, kernel fusion, multi-rail RDMA) are expected to close the gap, especially for MoE-style workloads where overlap with computation is critical.

## Documentation

Step-by-step tutorial documents are available in the [`docs/`](docs/) directory:

| Document | Topic |
|----------|-------|
| [00-introduction.md](docs/00-introduction.md) | NVSHMEM concepts, memory model, core primitives (`put`/`get`), synchronization |
| [01-initialization.md](docs/01-initialization.md) | Unique ID bootstrap, `nvshmemx_init_attr`, team creation |

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/my-feature`)
3. **Commit** your changes (`git commit -am 'Add my feature'`)
4. **Push** to the branch (`git push origin feature/my-feature`)
5. **Open** a Pull Request

For bug reports and feature requests, please use [GitHub Issues](https://github.com/KuangjuX/NVSHMEM-Tutorial/issues).

## Acknowledgements

- [**DeepEP**](https://github.com/deepseek-ai/DeepEP) (DeepSeek) — Core inspiration for the buffer architecture and optimization ideas.
- [**NVSHMEM**](https://developer.nvidia.com/nvshmem) (NVIDIA) — The underlying PGAS communication library.
- [**NCCL**](https://github.com/NVIDIA/nccl) (NVIDIA) — Used as the baseline for performance comparison.

## Citation

If you find this tutorial helpful, please consider giving it a ⭐ on [GitHub](https://github.com/KuangjuX/NVSHMEM-Tutorial)!

```bibtex
@misc{nvshmem-tutorial,
  title  = {NVSHMEM-Tutorial: Build a DeepEP-like GPU Buffer},
  author = {Chengxiang Qi},
  year   = {2025},
  url    = {https://github.com/KuangjuX/NVSHMEM-Tutorial}
}
```
