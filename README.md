# NVSHMEM Tutorial

A hands-on course on GPU-to-GPU communication using NVSHMEM for system engineers. The codebase focuses on low-level NVSHMEM APIs and CUDA kernels without high-level abstractions, so that we can build efficient multi-GPU communication patterns from scratch and understand the underlying optimizations.

The goal is to learn the techniques behind efficiently implementing inter-GPU communication in distributed deep learning systems (e.g., AllReduce, AllGather, communication buffers).

In week 1, you will implement the necessary components to establish NVSHMEM communication between GPUs (e.g., symmetric memory allocation, one-sided put/get operations, barriers). In week 2, you will implement advanced communication patterns used in distributed training systems like DeepSpeed and Megatron (e.g., communication buffers, overlapped communication, multi-node setup). In week 3, we will cover performance optimizations and integration with real-world distributed training frameworks.

**Why NVSHMEM**: it enables direct GPU-to-GPU communication without CPU involvement, which is crucial for scaling distributed training to hundreds of GPUs.

**Why focus on communication patterns**: understanding efficient inter-GPU communication is essential for building scalable distributed AI systems, and NVSHMEM provides the lowest-level primitives for maximum performance.

## Book
The NVSHMEM tutorial book is available at [https://nvshmem-tutorial.github.io/](https://nvshmem-tutorial.github.io/). You can follow the guide and start building.

## Community
You may join the NVSHMEM Tutorial Discord server and study with the community.

[Join NVSHMEM Tutorial Discord Server](https://discord.gg/nvshmem-tutorial)

## Roadmap
Week 1 is complete. Week 2 is in progress.

| Week + Chapter | Topic | Code | Test | Doc |
|----------------|-------|------|------|-----|
| 1.1 | NVSHMEM Initialization | ✅ | ✅ | ✅ |
| 1.2 | Symmetric Memory Allocation | ✅ | ✅ | ✅ |
| 1.3 | One-sided Put/Get Operations | ✅ | ✅ | ✅ |
| 1.4 | Barriers and Synchronization | ✅ | ✅ | ✅ |
| 1.5 | Multi-GPU Ring Communication | ✅ | ✅ | ✅ |
| 1.6 | PyTorch Tensor Integration | ✅ | ✅ | ✅ |
| 1.7 | Error Handling and Debugging | ✅ | ✅ | ✅ |
| 2.1 | Communication Buffers | ✅ | ✅ | 🚧 |
| 2.2 | AllReduce Implementation | ✅ | ✅ | 🚧 |
| 2.3 | AllGather Implementation | ✅ | ✅ | 🚧 |
| 2.4 | Overlapped Communication | ✅ | ✅ | 🚧 |
| 2.5 | Multi-node NVSHMEM Setup | ✅ | ✅ | 🚧 |
| 2.6 | Performance Profiling | ✅ | 🚧 | 🚧 |
| 2.7 | Memory Optimization | ✅ | 🚧 | 🚧 |
| 3.1 | Advanced Topology Awareness | 🚧 | 🚧 | 🚧 |
| 3.2 | Custom Collective Operations | 🚧 | 🚧 | 🚧 |
| 3.3 | NCCL vs NVSHMEM Comparison | 🚧 | 🚧 | 🚧 |
| 3.4 | Integration with DeepSpeed | 🚧 | 🚧 | 🚧 |
| 3.5 | Integration with Megatron-LM | 🚧 | 🚧 | 🚧 |
| 3.6 | Multi-tenant GPU Sharing | 🚧 | 🚧 | 🚧 |
| 3.7 | Cross-platform Communication | 🚧 | 🚧 | 🚧 |

*Other topics not covered: RDMA optimization, InfiniBand tuning, fault tolerance, dynamic topology changes*