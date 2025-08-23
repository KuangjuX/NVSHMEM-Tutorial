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

  CUDA_CHECK(cudaMemcpyAsync(buffer_ptrs_[nvl_rank_], tensor.data_ptr(),
                             tensor.nbytes(), cudaMemcpyDeviceToDevice,
                             comm_stream_));

  CUDA_CHECK(cudaMemcpyAsync(rdma_buffer_ptr_, tensor.data_ptr(),
                             tensor.nbytes(), cudaMemcpyDeviceToDevice,
                             comm_stream_));
  // intranode barrier
  sync::barrier(barrier_signal_ptrs_gpu_, rank_, num_ranks_, comm_stream_);

  // internode barrier
  nvshmem::barrier();

  for (int rank = 0; rank < num_ranks_; ++rank) {
    // TODO(KuangjuX): Implement this
    bool is_same_node = true;

    if (is_same_node) {
      // Intranode: CUDA IPC
      CUDA_CHECK(cudaMemcpyAsync(tensor_list[rank].data_ptr(),
                                 buffer_ptrs_[rank], tensor.nbytes(),
                                 cudaMemcpyDeviceToDevice, comm_stream_));
    } else {
      // Internode: RDMA
      nvshmem::get_mem(tensor_list[rank].data_ptr(), rdma_buffer_ptr_,
                       tensor.nbytes(), rank);
    }

    if (!async_op) {
      cudaStreamSynchronize(comm_stream_);
    }
  }
}

}  // namespace nvshmem_tutorial
