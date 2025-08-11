#pragma once

#include "utils.hpp"

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <optional>
#include <vector>

namespace py = pybind11;

class Buffer {
  STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "NUM_MAX_NVL_PEERS must be equal to 8");

 public:
  // DeepEP-like constructor: with rank topology and RDMA bytes
  Buffer(int rank, int num_ranks, int64_t num_nvl_bytes,
         int64_t num_rdma_bytes);

  ~Buffer();

  // NVSHMEM symmetric memory helpers
  torch::Tensor alloc_symmetric(int64_t size_bytes);
  void free_symmetric(torch::Tensor t);

  // DeepEP-like: export NVSHMEM unique id (only on rdma root, i.e., rank %
  // NUM_MAX_NVL_PEERS == 0)
  // py::bytearray get_local_nvshmem_unique_id() const;

  // DeepEP-like: open CUDA IPC handles and/or initialize NVSHMEM and allocate
  // RDMA buffer
  // void sync(
  //     const std::vector<int>& device_ids,
  //     const std::vector<std::optional<py::bytearray>>& all_gathered_handles,
  //     const std::optional<py::bytearray>& root_unique_id_opt);

  // Intra-node (NVLink) helpers
  py::bytearray get_local_ipc_handle() const;
  void open_ipc_handles(
      const std::vector<std::optional<py::bytearray>>& all_handles);
  void intranode_memcpy_to(int dst_local_pe, int64_t dst_offset_bytes,
                           torch::Tensor src);
  torch::Tensor get_local_buffer_u8() const;

  // Inter-node (NVSHMEM) helpers
  void internode_put(torch::Tensor dst_symmetric, torch::Tensor src,
                     int dst_pe);
  void internode_get(torch::Tensor dst, torch::Tensor src_symmetric,
                     int src_pe);

  // DeepEP-like: view buffers as typed tensor
  torch::Tensor get_local_buffer_tensor(const py::object& dtype, int64_t offset,
                                        bool use_rdma_buffer) const;

  // Lifecycle
  void destroy();

  // Introspection
  bool is_available() const { return available_; }
  int get_local_pe() const;           // NVSHMEMX_TEAM_NODE local PE index
  int get_num_local_pes() const;      // NVSHMEMX_TEAM_NODE size
  int get_device_id() const;          // CUDA device id
  int64_t get_num_nvl_bytes() const;  // NVLink buffer bytes

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

  // NVLink local buffers and IPC
  int64_t num_nvl_bytes_{0};
  void* buffer_ptrs_[NUM_MAX_NVL_PEERS] = {nullptr};
  cudaIpcMemHandle_t ipc_handles_[NUM_MAX_NVL_PEERS]{};

  // NVSHMEM RDMA buffer
  int64_t num_rdma_bytes_{0};
  void* rdma_buffer_ptr_{nullptr};

  bool local_allocated_{false};
  bool available_{false};
};