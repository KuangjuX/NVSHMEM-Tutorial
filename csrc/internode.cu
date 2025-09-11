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
  auto rdma_buffer = SymLayout<uint8_t>(rdma_buffer_ptr_, same_node_total_bytes,
                                        num_rdma_ranks_);

  auto send_rdma_buffer = rdma_buffer.send_buffer(rdma_rank_);

  int leader_rank = 0;

  // 0. Copy local tensor to NVLink buffer
  void* leader_nvl_buffer = buffer_ptrs_[leader_rank];
  void* slot_in_leader_nvl_buffer =
      static_cast<char*>(leader_nvl_buffer) + tensor.nbytes() * nvl_rank_;

  CUDA_CHECK(cudaMemcpyAsync(slot_in_leader_nvl_buffer, tensor.data_ptr(),
                             tensor.nbytes(), cudaMemcpyDeviceToDevice,
                             comm_streams_[nvl_rank_]));

  sync::barrier(barrier_signal_ptrs_gpu_, nvl_rank_, NUM_MAX_NVL_PEERS,
                comm_streams_[nvl_rank_]);

  if (nvl_rank_ == leader_rank) {
    // 1. Leader rank sends all local tensors to rdma buffer.
    CUDA_CHECK(cudaMemcpyAsync(send_rdma_buffer, leader_nvl_buffer,
                               same_node_total_bytes, cudaMemcpyDeviceToDevice,
                               comm_streams_[nvl_rank_]));

    // cudaStreamSynchronize + nvshmem_barrier_all together
    // to ensure nvshmem_get happens after data is ready in 
    // rdma buffer.
    cudaStreamSynchronize(comm_streams_[nvl_rank_]);
    nvshmem::barrier();

    // All leader PEs conduct internode communications.
    for (int rdma_rank = 0; rdma_rank < num_rdma_ranks_; rdma_rank++) {
      if (rdma_rank == rdma_rank_) {
        continue;
      }
      auto dst_send_rdma_buffer = rdma_buffer.send_buffer(rdma_rank);
      auto src_recv_rdma_buffer = rdma_buffer.recv_buffer(rdma_rank);
      nvshmem::get_mem_async(src_recv_rdma_buffer, dst_send_rdma_buffer,
                             same_node_total_bytes, rdma_rank, comm_streams_[nvl_rank_]);
    }

    // Ensure leader rank itself has finished its own internode part.
    cudaStreamSynchronize(comm_streams_[nvl_rank_]);
  
    // Copy from buffer into tensor_list and ranks on the same node
    // simultaneously to avoid calling cudaStreamSynchronize.
    for (int rank = 0; rank < num_ranks_; rank++) {
      void* ptr = nullptr;
      if (is_same_rdma_rank(rank)) {
        ptr =
            send_rdma_buffer + (rank % NUM_MAX_NVL_PEERS) * tensor.nbytes();
      } else {
        auto rdma_rank = rank / NUM_MAX_NVL_PEERS;
        ptr = rdma_buffer.recv_buffer(rdma_rank) +
                   (rank % NUM_MAX_NVL_PEERS) * tensor.nbytes();
      }

      CUDA_CHECK(cudaMemcpyAsync(tensor_list[rank].data_ptr(), ptr,
                                   tensor.nbytes(), cudaMemcpyDeviceToDevice,
                                   comm_streams_[nvl_rank_]));

      // 2. Copy from leader rank back to other ranks on the same node.
      for (int nvl_rank = 0; nvl_rank < NUM_MAX_NVL_PEERS; ++nvl_rank) {
        if (nvl_rank == leader_rank) {
          continue;
        }
        void* dst_nvl_buffer = buffer_ptrs_[nvl_rank];
        void* slot_in_dst_nvl_buffer =
            static_cast<char*>(dst_nvl_buffer) + rank * tensor.nbytes();
        CUDA_CHECK(cudaMemcpyAsync(slot_in_dst_nvl_buffer, ptr, 
                                   tensor.nbytes(), cudaMemcpyDeviceToDevice,
                                   comm_streams_[nvl_rank_]));
      }
    }
  }

  // Ensure data is finished copying from leader rank 
  // before other ranks start reading.
  sync::barrier(barrier_signal_ptrs_gpu_, nvl_rank_, NUM_MAX_NVL_PEERS,
                comm_streams_[nvl_rank_]);

  // 3. Non-leader ranks copy from buffer into tensor_list.
  if (nvl_rank_ != leader_rank) {
    void* src_nvl_buffer = buffer_ptrs_[nvl_rank_];
    for (int i = 0; i < num_ranks_; ++i) {
      void* slot_in_src_nvl_buffer =
          static_cast<char*>(src_nvl_buffer) + i * tensor.nbytes();
      CUDA_CHECK(cudaMemcpyAsync(tensor_list[i].data_ptr(),
                                 slot_in_src_nvl_buffer, tensor.nbytes(),
                                 cudaMemcpyDeviceToDevice, comm_streams_[nvl_rank_]));
    }
  }

  if (!async_op) {
    CUDA_CHECK(cudaStreamSynchronize(comm_streams_[nvl_rank_]));
  }
}

}  // namespace nvshmem_tutorial
