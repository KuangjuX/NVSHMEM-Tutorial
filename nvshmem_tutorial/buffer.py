import os
import torch
import torch.distributed as dist
import _nvshmem_pybind_cpp as nvshmem_runtime  # The name comes from setup.py


class NvshmemBuffer:
    def __init__(
        self,
        group: dist.ProcessGroup,
        rank: int,
        num_ranks: int,
        num_nvl_bytes: int,
        num_rdma_bytes: int,
    ):
        os.environ["NVSHMEM_REMOTE_TRANSPORT"] = "none"

        self.group = group
        self.group_size = group.size()
        self.rank = rank
        self.num_ranks = num_ranks
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes

        # Initialize nvshmem with unique id
        if rank == 0:
            unique_id = nvshmem_runtime.get_unique_id()
        else:
            unique_id = None

        unique_ids = [None] * num_ranks
        dist.all_gather_object(unique_ids, unique_id, group=dist.group.WORLD)
        nvshmem_runtime.init_with_unique_id(unique_ids[0], rank, num_ranks)

        print(f"[Rank {rank}] NVSHMEM initialized successfully.")
        dist.barrier()

        self.runtime = nvshmem_runtime.Buffer(
            rank, num_ranks, num_nvl_bytes, num_rdma_bytes
        )

        # Synchronize device IDs
        device_ids = [None] * self.group_size
        local_device_id = self.runtime.get_local_device_id()
        dist.all_gather_object(device_ids, local_device_id, group)

        # Synchronize IPC handles
        ipc_handles = [None] * self.group_size
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        dist.all_gather_object(ipc_handles, local_ipc_handle, group)

    def __del__(self):
        pass
