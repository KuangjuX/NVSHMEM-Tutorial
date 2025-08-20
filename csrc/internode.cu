#include "buffer.cuh"
#include "utils.hpp"

namespace nvshmem_tutorial {

void Buffer::internode_all_gather(std::vector<torch::Tensor>& tensor_list,
                                  const torch::Tensor& tensor, bool async_op) {
  // TODO(KuangjuX): Implement this
  if (!tensor.is_cuda()) {
    throw std::runtime_error("internode_all_gather expects CUDA tensor");
  }

  if (buffer_ptrs_[nvl_rank_] == nullptr) {
    throw std::runtime_error("Local NVLink buffer not allocated");
  }

  CUDA_CHECK(cudaMemcpyAsync(tensor_list[nvl_rank_].data_ptr(),
                             tensor.data_ptr(), tensor.nbytes(),
                             cudaMemcpyDeviceToDevice, comm_stream_));
}
}  // namespace nvshmem_tutorial
