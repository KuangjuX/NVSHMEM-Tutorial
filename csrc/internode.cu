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

  int same_node_total_bytes = tensor.nbytes() * NUM_MAX_NVL_PEERS;
  auto rdma_buffer =
      SymLayout<uint8_t>(rdma_buffer_ptr_, same_node_total_bytes, num_rdma_ranks_);

  auto send_rdma_buffer = rdma_buffer.send_buffer(rdma_rank_);
  auto base_recv_rdma_buffer = rdma_buffer.recv_buffer(0);

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
    CUDA_CHECK(cudaMemcpyAsync(send_rdma_buffer, leader_nvl_buffer, same_node_total_bytes,
                               cudaMemcpyDeviceToDevice, comm_stream_));

    cudaStreamSynchronize(comm_stream_);

    for (int rdma_rank = 0; rdma_rank < num_rdma_ranks_; rdma_rank++) {
      if (rdma_rank == rdma_rank_) {
        continue;
      }
      auto dst_send_rdma_buffer = rdma_buffer.send_buffer(rdma_rank);
      auto src_recv_rdma_buffer = rdma_buffer.recv_buffer(rdma_rank);
      nvshmem::get_mem_async(src_recv_rdma_buffer, dst_send_rdma_buffer,
                             same_node_total_bytes, rdma_rank, comm_stream_);
    }
  }

  cudaStreamSynchronize(comm_stream_);

  for (int rank = 0; rank < num_ranks_; rank++) {
    if (is_same_rdma_rank(rank)) {
      auto ptr =
          send_rdma_buffer + (rank % NUM_MAX_NVL_PEERS) * tensor.nbytes();
      CUDA_CHECK(cudaMemcpyAsync(tensor_list[rank].data_ptr(), ptr,
                                 tensor.nbytes(), cudaMemcpyDeviceToDevice,
                                 comm_stream_));
    } else {
      auto rdma_rank = rank / NUM_MAX_NVL_PEERS;
      auto ptr = 
        rdma_buffer.recv_buffer(rdma_rank) + (rank % NUM_MAX_NVL_PEERS) * tensor.nbytes();
      CUDA_CHECK(cudaMemcpyAsync(tensor_list[rank].data_ptr(), ptr,
                                 tensor.nbytes(), cudaMemcpyDeviceToDevice,
                                 comm_stream_));
    }
  }

  cudaStreamSynchronize(comm_stream_);

  // 2. Copy from leader_rank back to other ranks on the same node.
  if (nvl_rank_ == leader_rank) {
    for (int rank = 0; rank < NUM_MAX_NVL_PEERS; ++rank) {
      if (rank == leader_rank) {
        continue;
      }
      
      // Async copy tensor by tensor to avoid calling cudaStreamSynchronize.
      void* dst_nvl_buffer = buffer_ptrs_[rank];
      for (int i = 0; i < num_ranks_; ++i) {
        void* slot_in_dst_nvl_buffer = static_cast<char*>(dst_nvl_buffer) + i * tensor.nbytes();
        CUDA_CHECK(cudaMemcpyAsync(slot_in_dst_nvl_buffer, tensor_list[i].data_ptr(), 
                                   tensor.nbytes(), cudaMemcpyDeviceToDevice, 
                                   comm_stream_));
      }
    }
  }

  cudaStreamSynchronize(comm_stream_);

  // 3. Non-leader_rank copy from buffer into tensor_list.
  if (nvl_rank_ != leader_rank) {
    void* src_nvl_buffer =  buffer_ptrs_[nvl_rank_];
    for (int i = 0; i < num_ranks_; ++i) {
      void* slot_in_src_nvl_buffer = static_cast<char*>(src_nvl_buffer) + i * tensor.nbytes();
      CUDA_CHECK(cudaMemcpyAsync(tensor_list[i].data_ptr(), slot_in_src_nvl_buffer, 
                      tensor.nbytes(), cudaMemcpyDeviceToDevice, 
                      comm_stream_));
    }
  }

  if (!async_op) {
    CUDA_CHECK(cudaStreamSynchronize(comm_stream_));
  }
}

}  // namespace nvshmem_tutorial
