#pragma once

#include "utils.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <optional>
#include <vector>

namespace py = pybind11;

namespace nvshmem_tutorial {
class Buffer {
  STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "NUM_MAX_NVL_PEERS must be equal to 8");

 public:
  // DeepEP-like constructor: with rank topology and RDMA bytes
  Buffer(int rank, int num_ranks, int64_t num_nvl_bytes,
         int64_t num_rdma_bytes);

  ~Buffer();

  // Allocate Symmetric Memory.
  torch::Tensor alloc_symmetric(int64_t size_bytes);

  // Free Symmetric Memory.
  void free_symmetric(torch::Tensor t);

  // py::bytearray get_local_nvshmem_unique_id() const;

  void sync(
      const std::vector<int>& device_ids,
      const std::vector<std::optional<py::bytearray>>& all_gathered_handles,
      const std::optional<py::bytearray>& root_unique_id_opt);

  // Intra-node (NVLink) helpers
  py::bytearray get_local_ipc_handle() const;

  torch::Tensor get_local_buffer_u8() const;

  // Intra-node communication methods

  void intranode_all_gather(std::vector<torch::Tensor>& tensor_list,
                            const torch::Tensor& tensor, bool async_op);

  void intranode_all_to_all(torch::Tensor input, torch::Tensor output,
                            torch::Tensor input_split_sizes,
                            torch::Tensor output_split_sizes, bool async_op);

  void internode_all_gather(std::vector<torch::Tensor>& tensor_list,
                            const torch::Tensor& tensor, bool async_op);

  // DeepEP-like: view buffers as typed tensor
  torch::Tensor get_local_buffer_tensor(const py::object& dtype, int64_t offset,
                                        bool use_rdma_buffer) const;

  // Lifecycle
  void destroy();

  // Introspection
  bool is_available() const { return available_; }
  int get_local_pe() const;           // NVSHMEMX_TEAM_NODE local PE index
  int get_num_local_pes() const;      // NVSHMEMX_TEAM_NODE size
  int get_local_device_id() const;    // CUDA device id
  int64_t get_num_nvl_bytes() const;  // NVLink buffer bytes

  int get_num_device_sms() const;

 private:
  // Topology
  int rank_{0};
  int num_ranks_{1};
  int rdma_rank_{0};
  int nvl_rank_{0};
  int num_rdma_ranks_{1};
  int num_nvl_ranks_{1};

  // Device & teams
  int device_id_{-1};
  int local_pe_{0};
  int num_local_pes_{1};
  int num_device_sms_{1};

  // NVLink local buffers
  int64_t num_nvl_bytes_{0};
  void* buffer_ptrs_[NUM_MAX_NVL_PEERS] = {nullptr};
  void** buffer_ptrs_gpu_{nullptr};

  // Barrier signals
  int* barrier_signal_ptrs_[NUM_MAX_NVL_PEERS] = {nullptr};
  int** barrier_signal_ptrs_gpu_{nullptr};

  // IPC handles
  cudaIpcMemHandle_t ipc_handles_[NUM_MAX_NVL_PEERS]{};

  // NVSHMEM RDMA buffer
  int64_t num_rdma_bytes_{0};
  void* rdma_buffer_ptr_{nullptr};

  bool local_allocated_{false};
  bool available_{false};

  at::cuda::CUDAStream comm_stream_;
};
}  // namespace nvshmem_tutorial
