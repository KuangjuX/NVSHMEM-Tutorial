#include "buffer.cuh"
#include "nvshmem.hpp"
#include "sync.cuh"
#include "utils.hpp"

namespace nvshmem_tutorial {

void Buffer::internode_all_gather(std::vector<torch::Tensor>& tensor_list,
                                  const torch::Tensor& tensor, bool async_op) {
  if (!tensor.is_cuda()) {
    throw std::runtime_error("internode_all_gather expects CUDA tensor");
  }

  if (buffer_ptrs_[nvl_rank_] == nullptr) {
    throw std::runtime_error("Local NVLink buffer not allocated");
  }

  // 0. Copy local tensor to NVLink buffer
  const int send_rank = 0;
  void* send_nvl_buffer = buffer_ptrs_[send_rank];
  void* slot_in_send_nvl_buffer =
      static_cast<char*>(send_nvl_buffer) + tensor.nbytes() * nvl_rank_;

  CUDA_CHECK(cudaMemcpyAsync(slot_in_send_nvl_buffer, tensor.data_ptr(),
                             tensor.nbytes(), cudaMemcpyDeviceToDevice,
                             comm_stream_));

  sync::barrier(barrier_signal_ptrs_gpu_, nvl_rank_, NUM_MAX_NVL_PEERS,
                comm_stream_);

  // 1. Rank 0 sends all local tensors to rdma buffer.
  if (nvl_rank_ == send_rank) {
    const int send_bytes = tensor.nbytes() * NUM_MAX_NVL_PEERS;
    void* send_rdma_buffer_ptr =
        static_cast<char*>(rdma_buffer_ptr_) + rdma_rank_ * send_bytes;

    CUDA_CHECK(cudaMemcpyAsync(send_rdma_buffer_ptr, send_nvl_buffer,
                               send_bytes, cudaMemcpyDeviceToDevice,
                               comm_stream_));

    nvshmem::barrier();

    for (int rdma_rank = 0; rdma_rank < num_rdma_ranks_; rdma_rank++) {
      if (rdma_rank == rdma_rank_) {
        continue;
      }

      void* local_rdma_buffer_ptr =
          static_cast<char*>(send_nvl_buffer) + rdma_rank * send_bytes;
      void* remote_rdma_buffer_ptr =
          static_cast<char*>(rdma_buffer_ptr_) + rdma_rank * send_bytes;

      nvshmem::get_mem(local_rdma_buffer_ptr, remote_rdma_buffer_ptr,
                       send_bytes, rdma_rank);
    }
  }

  nvshmem::barrier();

  for (int rank = 0; rank < num_ranks_; rank++) {
    void* local_buffer_ptr =
        static_cast<char*>(send_nvl_buffer) + rank * tensor.nbytes();

    CUDA_CHECK(cudaMemcpyAsync(tensor_list[rank].data_ptr(), local_buffer_ptr,
                               tensor.nbytes(), cudaMemcpyDeviceToDevice,
                               comm_stream_));
  }

  if (!async_op) {
    CUDA_CHECK(cudaStreamSynchronize(comm_stream_));
  }
}

}  // namespace nvshmem_tutorial
