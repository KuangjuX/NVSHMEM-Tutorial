import os
import torch.distributed as dist
import _nvshmem_pybind_cpp as nvshmem_ops # The name comes from setup.py

class NvshmemBuffer:
    def __init__(self, group: dist.ProcessGroup, rank: int, num_ranks: int, num_nvl_bytes: int, num_rdma_bytes: int):
        os.environ["NVSHMEM_REMOTE_TRANSPORT"] = "none"
        
        if rank == 0:
            # Rank 0 创建 unique_id
            unique_id = nvshmem_ops.get_unique_id()
        else:
            unique_id = None

        unique_ids = [None] * num_ranks
        dist.all_gather_object(unique_ids, unique_id, group = dist.group.WORLD)

        # 5. 使用接收到的 unique_id 初始化 NVSHMEM
        nvshmem_ops.init_with_unique_id(unique_ids[0], rank, num_ranks)
        
        print(f"[Rank {rank}] NVSHMEM initialized successfully.")
        
        # 确保所有进程都完成了初始化再继续
        dist.barrier()

        self.buffer = nvshmem_ops.Buffer(rank, num_ranks, num_nvl_bytes, num_rdma_bytes)
