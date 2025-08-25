# NVSHMEM 初始化


**基于 Unique ID 的初始化**

核心思想:

1. 由一个指定的“根”进程（通常是 rank 0）创建一个全局唯一的标识符 (Unique ID)。
2. 根进程通过某种外部带外机制（out-of-band mechanism）将这个 Unique ID 分发给所有其他参与的进程。
3. 所有进程（包括根进程）使用这个相同的 Unique ID 以及各自的 rank 信息来初始化自己的 NVSHMEM 环境。

```cpp
// get_unique_id()
// 作用：由 rank 0 进程调用，生成一个全局唯一的 ID。
std::vector<uint8_t> get_unique_id() {
    nvshmemx_uniqueid_t unique_id;
    // 调用 NVSHMEM 扩展 API 来获取一个 ID 结构体
    nvshmemx_get_uniqueid(&unique_id);

    // 将 ID 序列化到一个字节向量中，以便于传输
    std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
    return result;
}
```

- `nvshmemx_get_uniqueid`: 这是 NVSHMEM 扩展 API (nvshmemx) 中的一个函数，用于生成一个唯一的标识符。
- `std::vector<uint8_t>`: 将这个 ID 拷贝到一个 uint8_t 的 vector 中是一个很好的实践。这使得 ID 变成了一个可序列化的字节流，可以轻松地通过各种方式传输。

```cpp
int init(const std::vector<uint8_t> &root_unique_id_val, int rank, int num_ranks, bool low_latency_mode) {
    nvshmemx_uniqueid_t root_unique_id;
    nvshmemx_init_attr_t attr;
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(nvshmemx_uniqueid_t));
    nvshmemx_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

    // Create sub-RDMA teams
    // NOTES: if `num_ranks <= NUM_MAX_NVL_PEERS` then only low-latency kernels are used
    if (low_latency_mode and num_ranks > NUM_MAX_NVL_PEERS) {
        EP_HOST_ASSERT(cpu_rdma_team == NVSHMEM_TEAM_INVALID);
        EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
        EP_HOST_ASSERT(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, rank % NUM_MAX_NVL_PEERS, NUM_MAX_NVL_PEERS,
                                                  num_ranks / NUM_MAX_NVL_PEERS, &cpu_rdma_team_config, 0, &cpu_rdma_team) == 0);
        EP_HOST_ASSERT(cpu_rdma_team != NVSHMEM_TEAM_INVALID);
    }

    nvshmem_barrier_all();
    return nvshmem_my_pe();
}
```

- `nvshmemx_set_attr_unique_args`: 将进程的身份信息(`rank`，`num_ranks`) 和通信组的标识(`root_unique_id`) 打包到一个属性结构体 `attr` 中。
- `nvshmemx_init_attr`: 初始化调用，NVSHMEM 运行时使用我们的 unique_id 寻找并连接到其他持有相同 Unique ID 的对等线程。
- 创建通信子组(Teams)：在一个多节点 GPU 集群中，节点内部的 GPU 通信速度远快于节点间通信。`nvshmem_team_split_strided` 将全局的通信组分割成更小的子组。创建这些基于物理拓扑的 team 后，可以在 Team 内部使用高度优化的、基于 NVLink 设计的集合通信算法。而在 Team 之间使用针对 RDMA 优化的算法。从而实现全局最优的通信性能。

## NVSHMEM Team

NVSHMEM Team 是处理单元的一个有序子集，想象一下整个程序有 16 个 GPUs，它们的全局 Rank 是 0 到 15。这是一个大的、默认的通信组，在 NVSHMEM 被称为 `NVSHMEM_TEAM_WORLD`。在每个 `Team` 内部，成员们都有一个新的、从 0 开始的本地 Rank，并且这个 `Team` 有自己的大小，这使得这个子集内部通信非常方便。

使用 NVSHMEM TEAM 的原因：

- 匹配硬件拓扑
- 实现算法分区：模型的不同层分布在不同的 GPU 组上，每一层可以是一个 `Team`，层内的 GPU 协同计算，层间的 GPU 完成数据传递。
- 提高性能和减少同步开销：在 `NVSHMEM_TEAM_WORLD` 上执行一个同步操作。例如 `nvshmem_barrier_all` 需要所有 PE 都参与同步，这是一个全局同步点，开销很大；如果程序逻辑只需要一小部分 PEs 同步，那么在一个只包含这些 PEs 的 `Team` 上执行 `nvshmem_barrier_on_team` 会快得多。

在 DeepEP 中的高吞吐模式和低延迟模式采用了不同的初始化方式。

- 高吞吐模式：为 `rdma_rank == 0` 的所有 GPU 生成 unique_id，并广播到其他 rdma_rank，使用 `rdma_rank` 和 `num_rdma_ranks` 进行初始化。
- 低延迟模式：仅仅为 `rank == 0` 的 GPU 生成 unique_id，广播到其他 rank，初始化后紧接着调用 `nvshmem_team_split_strided`。
