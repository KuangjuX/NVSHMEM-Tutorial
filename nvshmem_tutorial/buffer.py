import os
import torch
import torch.distributed as dist
import _nvshmem_tutorial as nvshmem_runtime  # The name comes from setup.py


class NvshmemBuffer:
    """Buffer class for NVSHMEM communication."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        rank: int,
        num_ranks: int,
        num_nvl_bytes: int,
        num_rdma_bytes: int,
        low_latency_mode: bool = False,
    ):
        """Initialize the NVSHMEM buffer."""

        self.group = group
        self.group_size = group.size()
        self.rank = rank
        self.num_ranks = num_ranks
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes

        self.runtime = nvshmem_runtime.Buffer(
            rank, num_ranks, num_nvl_bytes, num_rdma_bytes
        )

        print(f"[DEBUG] Rank {self.rank} initialized runtime buffer.")

        # Synchronize device IDs
        device_ids = [None] * self.group_size
        local_device_id = self.runtime.get_local_device_id()
        dist.all_gather_object(device_ids, local_device_id, group)

        # Synchronize IPC handles
        ipc_handles = [None] * self.group_size
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        dist.all_gather_object(ipc_handles, local_ipc_handle, group)

        root_unique_id = None
        if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
            os.environ["NVSHMEM_IB_ENABLE_IBGDA"] = "1"
            os.environ["NVSHMEM_DISABLE_P2P"] = "1"
            os.environ["NVSHMEM_MAX_TEAMS"] = "7"
            os.environ["NVSHMEM_DISABLE_NVLS"] = "1"

            # Synchronize using the root ID.
            nvshmem_unique_ids = [None] * self.group_size
            if (low_latency_mode and self.rank == 0) or (
                not low_latency_mode and self.runtime.get_rdma_rank() == 0
            ):
                root_unique_id = self.runtime.get_local_nvshmem_unique_id()

            dist.all_gather_object(nvshmem_unique_ids, root_unique_id, group)
            root_unique_id = nvshmem_unique_ids[
                0 if low_latency_mode else self.runtime.get_root_rdma_rank(True)
            ]

        if root_unique_id is None:
            raise RuntimeError(f"Rank {self.rank} Root unique ID is None")

        self.runtime.sync(device_ids, ipc_handles, root_unique_id)

    def __del__(self):
        self.runtime.destroy()
        self.runtime = None

    def get_num_device_sms(self):
        """Get the number of SMs on the device."""
        return self.runtime.get_num_device_sms()

    def intranode_all_gather(self, tensor_list, tensor, async_op=False):
        """Perform intra-node all-gather communication using NVLink and CUDA IPC."""
        if not tensor.is_cuda:
            raise ValueError("Tensor must be CUDA tensor")
        if len(tensor_list) != self.group_size:
            raise ValueError("Tensor list must match group size")
        self.runtime.intranode_all_gather(tensor_list, tensor, async_op)
