# NVSHMEM Introduction

NVSHMEM(NVIDIA SHared MEMory) 是一个基于 OpenSHMEM 标准的并行编程库。核心思想是实现一个分区全局地址空间(Partitioned Global Address Space)模型。

在 PGAS 模型中，所有参与计算的处理单元（在这里是 GPU）共享一个全局的、逻辑上的地址空间，但这个空间在物理上是分布在各个 GPU 的显存中的。这使得任何一个 GPU 都可以通过简单的内存操作（如 `put` 和 `get`）直接、单边地访问另一个 GPU 上的数据，而无需目标 GPU 的显式参与。

NVSHMEM 的特性:
- 单边通信：与传统 MPI 双边通信的最本质区别。在 NVSHMEM 中，一个 GPU 可以直接对另一个 GPU 的内存进行读写，无需目标 GPU 进行任何匹配操作，减少同步开销和通信延迟。
- GPU 原生通信：可以直接在 CUADA kernel 内部发起，消除了 CPU-GPU 上下文切换，实现了计算与通信的重叠。
- 硬件卸载：NVSHMEM 可以直接卸载到硬件执行。通过 NVLink 和 RDMA，数据传输可以绕过 CPU 和内核，直接在 GPU 网络接口和内存控制器进行。

## Memory Model

NVSHMEM 的内存模型有两个核心概念：对称堆(Symmetric Heap) 和分区全局地址空间(Partitioned Global Address Space, PGAS)。

对称堆事 NVSHMEM 在每个 PE（处理单元）的显存中划分出来的一块特殊内存区域。这块区域由 NVSHMEM 库管理，并通过专门的 API（如 `nvshmem_malloc`）进行分配。如果在对称堆上分配了一个对象，那么这个指针变量 `ptr` 在所有 PE 上都具有相同的逻辑地址值。然而，它在每个 PE 上指向的物理地址是不同的，是位于各自 PE 的本地显存中的。

当所有 PE 的对称堆集合在一起时，它们就共同构成了一个分区的全局地址空间 (PGAS)：
- 全局 (Global): 从任何一个 PE 的角度来看，它都可以访问这个地址空间中的任何位置，就好像在访问一个巨大的、统一的内存池一样。
- 分区 (Partitioned): 这个全局地址空间在物理上是分散的（分区的），每个 PE “拥有”并管理其中的一部分（即它自己的对称堆）。

## 核心通信原语

`put` 和 `get` 是最基本的单边操作原语。

`void nvshmem_putmem(void *dest, const void *source, size_t nelems, int pe)`

将本地内存中的数据，直接写入到目标 PE 的对称堆上的指定地址。

工作流程:

1. 源 PE (例如 PE 0) 准备好要发送的数据。这些数据可以位于其本地的任何地方（包括对称堆或普通的 cudaMalloc 内存）。
2. 源 PE 指定目标地址。这个目标地址必须是位于目标 PE (例如 PE 1) 的对称堆中的一个有效地址。由于对称性，源 PE 知道这个地址。
3. 源 PE 调用 `nvshmem_putmem(void *dest, const void *source, size_t nelems, int pe)`。
4. NVSHMEM 运行时库和硬件接管一切：
    - 它将 `dest` 解释为 PE 1 上的逻辑地址。
    - 通过 NVLink/NVSwitch 等高速互连，直接将 `source` 指向的数据传输到 PE 1 的物理显存中与 `dest` 对应的位置。
    - 整个过程，PE 1 的 CPU 或 GPU 内核完全不参与。它甚至不知道这次写入的发生。这就是“单边”的含义。


`void nvshmem_getmem(void *dest, const void *source, size_t nelems, int pe)`

从目标 PE 的对称堆上的指定地址，直接读取数据到本地内存中。

工作流程:

1. 源 PE (例如 PE 0) 在自己的本地内存中准备好一块缓冲区，用于存放即将读取的数据。
2. 源 PE 指定要从哪个目标 PE (例如 PE 1) 的哪个对称堆地址读取数据。
3. 源 PE 调用 `nvshmem_getmem(void *dest, const void *source, size_t nelems, int pe)`。
4. 硬件将数据从 PE 1 的对称堆中抓取，并通过互连网络传输回 PE 0，存入 local_buffer。
5. 同样，这个过程对 PE 1 来说是完全透明和无感的。

## 同步与内存一致性

既然通信是单边的，那么就会引出一个重要问题：PE 0 向 PE 1 put 了一个数据后，PE 1 什么时候才能安全地读取这个新数据呢？

直接的内存操作不是瞬间完成的，网络传输需要时间。因此，NVSHMEM 提供了一套同步原语来保证内存一致性。

- `nvshmem_quiet()`: 这是一个关键函数。当一个 PE 调用 `nvshmem_quiet()` 时，它会暂停执行，直到由该 PE 发起的所有远程内存操作（put, get, 原子操作等）都已经在目标 PE 上全部完成。

- `nvshmem_barrier_all()`: 这是一个更强的全局同步。它会阻塞所有 PE，直到所有 PE 发起的所有远程操作都已完成，并且所有 PE 都到达了这个屏障点。它隐含了 quiet 的功能。

## NVSHMEM Collective APIs

`int nvshmem_broadcastmem(shmem_team_t team, void *dest, const void *source, size_t nelems, int PE_root)`

将数据从一个指定的根 PE 拷贝到同一个 team 的其他 PE。

`int nvshmem_alltoallmem(shmem_team_t team, void *dest, const void *source, size_t nelems)`

每个 PE 都有一个数据块，它将这个数据块分割成 N 份，然后将第 i 份发送给第 i 个 PE。最终，每个 PE 都从其他 PE（包含自己）那里接收到了一小份数据，并将它们组成一个新的数据块。


