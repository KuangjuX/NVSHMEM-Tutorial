#include "buffer.cuh"
#include "nvshmem.hpp"
#include "sym.cuh"
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

  int total_bytes = tensor.nbytes() * NUM_MAX_NVL_PEERS;
  auto rdma_buffer =
      SymLayout<uint8_t>(rdma_buffer_ptr_, total_bytes, num_rdma_ranks_);

  auto send_rdma_buffer = rdma_buffer.send_buffer(rdma_rank_);
  auto recv_rdma_buffer = rdma_buffer.recv_buffer(rdma_rank_);

  int leader_rank = 0;

  // 0. Copy local tensor to NVLink buffer
  void* leader_nvl_buffer = buffer_ptrs_[leader_rank];
  void* slot_in_leader_nvl_buffer =
      static_cast<char*>(leader_nvl_buffer) + tensor.nbytes() * nvl_rank_;

  CUDA_CHECK(cudaMemcpyAsync(slot_in_leader_nvl_buffer, tensor.data_ptr(),
                             tensor.nbytes(), cudaMemcpyDeviceToDevice,
                             comm_stream_));

  sync::barrier(barrier_signal_ptrs_gpu_, nvl_rank_, NUM_MAX_NVL_PEERS,
                comm_stream_);

  // 1. Leader rank sends all local tensors to rdma buffer.
  if (nvl_rank_ == leader_rank) {
    CUDA_CHECK(cudaMemcpyAsync(send_rdma_buffer, leader_nvl_buffer, total_bytes,
                               cudaMemcpyDeviceToDevice, comm_stream_));

    nvshmem::barrier();

    for (int rdma_rank = 0; rdma_rank < num_rdma_ranks_; rdma_rank++) {
      if (rdma_rank == rdma_rank_) {
        continue;
      }
      auto dst_send_rdma_buffer = rdma_buffer.send_buffer(rdma_rank);
      nvshmem::get_mem_async(recv_rdma_buffer, dst_send_rdma_buffer,
                             total_bytes, rdma_rank, comm_stream_);
    }
  }

  // nvshmem::barrier();
  cudaStreamSynchronize(comm_stream_);

  for (int rank = 0; rank < num_ranks_; rank++) {
    if (is_same_rdma_rank(rank)) {
      auto ptr =
          send_rdma_buffer + (rank % NUM_MAX_NVL_PEERS) * tensor.nbytes();
      CUDA_CHECK(cudaMemcpyAsync(tensor_list[rank].data_ptr(), ptr,
                                 tensor.nbytes(), cudaMemcpyDeviceToDevice,
                                 comm_stream_));
    } else {
      auto ptr =
          recv_rdma_buffer + (rank % NUM_MAX_NVL_PEERS) * tensor.nbytes();
      CUDA_CHECK(cudaMemcpyAsync(tensor_list[rank].data_ptr(), ptr,
                                 tensor.nbytes(), cudaMemcpyDeviceToDevice,
                                 comm_stream_));
    }
  }

  if (!async_op) {
    CUDA_CHECK(cudaStreamSynchronize(comm_stream_));
  }
}

}  // namespace nvshmem_tutorial
