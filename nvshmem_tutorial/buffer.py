import os
import torch
import torch.distributed as dist
import _nvshmem_pybind_cpp as nvshmem_runtime  # The name comes from setup.py


class NvshmemBuffer:
    """Buffer class for NVSHMEM communication."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        rank: int,
        num_ranks: int,
        num_nvl_bytes: int,
        num_rdma_bytes: int,
    ):
        """Initialize the NVSHMEM buffer."""
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

    def get_num_device_sms(self):
        """Get the number of SMs on the device."""
        return self.runtime.get_num_device_sms()

    def intranode_all_to_all(
        self, input_tensor, output_tensor, input_split_sizes, output_split_sizes
    ):
        """Perform intra-node all-to-all communication using NVLink and CUDA IPC."""
        if not input_tensor.is_cuda() or not output_tensor.is_cuda():
            raise ValueError("Input and output must be CUDA tensors")
        if not input_split_sizes.is_cuda() or not output_split_sizes.is_cuda():
            raise ValueError("Split sizes must be CUDA tensors")
        if (
            input_split_sizes.numel() != self.group_size
            or output_split_sizes.numel() != self.group_size
        ):
            raise ValueError("Split sizes must match group size")
        self.runtime.intranode_all_to_all(
            input_tensor, output_tensor, input_split_sizes, output_split_sizes
        )
