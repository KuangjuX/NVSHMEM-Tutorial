#include "buffer.cuh"
#include "nvshmem.hpp"
#include "sync.cuh"

#include <cstring>

namespace nvshmem_tutorial {
// Helper to get NVSHMEM local pe info
static inline void query_local_pe(int& local_pe, int& num_local_pes) {
  local_pe = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  num_local_pes = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
}

Buffer::Buffer(int rank, int num_ranks, int64_t num_nvl_bytes,
               int64_t num_rdma_bytes)
    : rank_{rank},
      num_ranks_{num_ranks},
      num_nvl_bytes_{num_nvl_bytes},
      num_rdma_bytes_{num_rdma_bytes},
      comm_stream_(at::cuda::getStreamFromPool(true)) {
  CUDA_CHECK(cudaGetDevice(&device_id_));
  query_local_pe(local_pe_, num_local_pes_);

  rdma_rank_ = rank / NUM_MAX_NVL_PEERS;
  nvl_rank_ = rank % NUM_MAX_NVL_PEERS;
  num_rdma_ranks_ = std::max(1, num_ranks / NUM_MAX_NVL_PEERS);
  num_nvl_ranks_ = std::min(num_ranks, NUM_MAX_NVL_PEERS);

  int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);
  int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);
  int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);

  // Get ranks
  CUDA_CHECK(cudaGetDevice(&device_id_));

  // Get device info
  cudaDeviceProp device_prop = {};
  CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id_));
  num_device_sms_ = device_prop.multiProcessorCount;

  if (num_nvl_bytes_ > 0) {
    // Local IPC: alloc local memory and set local IPC handles.
    /**
     * Buffer layout:
     * | num_nvl_bytes_ | barrier_signal_bytes | buffer_ptr_bytes |
     * barrier_signal_ptr_bytes |
     */
    CUDA_CHECK(cudaMalloc(&buffer_ptrs_[nvl_rank_],
                          num_nvl_bytes_ + barrier_signal_bytes +
                              barrier_signal_ptr_bytes + buffer_ptr_bytes));
    local_allocated_ = true;
    CUDA_CHECK(
        cudaIpcGetMemHandle(&ipc_handles_[nvl_rank_], buffer_ptrs_[nvl_rank_]));
    buffer_ptrs_gpu_ = reinterpret_cast<void**>(
        static_cast<uint8_t*>(buffer_ptrs_[nvl_rank_]) + num_nvl_bytes_ +
        barrier_signal_bytes);

    // Set barrier signals
    barrier_signal_ptrs_[nvl_rank_] = reinterpret_cast<int*>(
        static_cast<uint8_t*>(buffer_ptrs_[nvl_rank_]) + num_nvl_bytes_);
    barrier_signal_ptrs_gpu_ = reinterpret_cast<int**>(
        static_cast<uint8_t*>(buffer_ptrs_[nvl_rank_]) + num_nvl_bytes_ +
        barrier_signal_bytes + buffer_ptr_bytes);

    // Do not synchronize here, will be synchronized in sync()
    CUDA_CHECK(cudaMemsetAsync(barrier_signal_ptrs_[nvl_rank_], 0,
                               barrier_signal_bytes, comm_stream_));
  }
}

Buffer::~Buffer() {
  if (!destroyed_) {
    destroy();
  }
}

void Buffer::sync(
    const std::vector<int>& device_ids,
    const std::vector<std::optional<py::bytearray>>& all_gathered_handles,
    const std::optional<py::bytearray>& root_unique_id_opt) {
  // Open CUDA IPC peers
  if (num_nvl_bytes_ > 0) {
    if (static_cast<int>(device_ids.size()) != num_ranks_) {
      throw std::runtime_error("sync: device_ids size mismatch");
    }
    if (static_cast<int>(all_gathered_handles.size()) != num_ranks_) {
      throw std::runtime_error("sync: handles size mismatch");
    }
    // Map only peers within our NVL group
    int offset = rdma_rank_ * num_nvl_ranks_;
    for (int i = 0; i < num_nvl_ranks_; ++i) {
      auto peer_rank = offset + i;
      auto handle_str = std::string(
          all_gathered_handles[peer_rank].value().cast<std::string>());
      if (peer_rank != rank_) {
        if (handle_str.size() != sizeof(cudaIpcMemHandle_t)) {
          throw std::runtime_error("CUDA IPC handle size mismatch");
        }
        std::memcpy(ipc_handles_[i].reserved, handle_str.data(),
                    sizeof(cudaIpcMemHandle_t));
        CUDA_CHECK(cudaIpcOpenMemHandle(&buffer_ptrs_[i], ipc_handles_[i],
                                        cudaIpcMemLazyEnablePeerAccess));
        barrier_signal_ptrs_[i] = reinterpret_cast<int*>(
            static_cast<uint8_t*>(buffer_ptrs_[i]) + num_nvl_bytes_);
      }
    }
    // Copy all buffer and barrier signal pointers to GPU
    CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu_, buffer_ptrs_,
                          sizeof(void*) * NUM_MAX_NVL_PEERS,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(barrier_signal_ptrs_gpu_, barrier_signal_ptrs_,
                          sizeof(int*) * NUM_MAX_NVL_PEERS,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Initialize NVSHMEM and allocate RDMA buffer
  if (num_rdma_bytes_ > 0) {
    if (!root_unique_id_opt.has_value()) {
      throw std::runtime_error("sync: missing root NVSHMEM unique id");
    }

    std::vector<uint8_t> root_unique_id(root_unique_id_opt->size());
    auto root_unique_id_str = root_unique_id_opt->cast<std::string>();
    std::memcpy(root_unique_id.data(), root_unique_id_str.c_str(),
                root_unique_id_opt->size());
    auto nvshmem_rank = low_latency_mode_ ? rank_ : rdma_rank_;
    auto num_nvshmem_ranks = low_latency_mode_ ? num_ranks_ : num_rdma_ranks_;
    HOST_ASSERT(nvshmem::init_with_unique_id(
                    root_unique_id, nvshmem_rank, num_nvshmem_ranks,
                    low_latency_mode_) == nvshmem_rank);

    // Barrier
    nvshmem::barrier();

    rdma_buffer_ptr_ =
        nvshmem::alloc(num_rdma_bytes_, NUM_BUFFER_ALIGNMENT_BYTES);

    // Clean buffer.
    CUDA_CHECK(cudaMemset(rdma_buffer_ptr_, 0, num_rdma_bytes_));

    // Barrier
    nvshmem::barrier();
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  available_ = true;
}

py::bytearray Buffer::get_local_ipc_handle() const {
  if (!local_allocated_ || num_nvl_bytes_ == 0) {
    throw std::runtime_error("No local NVLink buffer allocated");
  }
  return py::bytearray(
      reinterpret_cast<const char*>(ipc_handles_[nvl_rank_].reserved),
      sizeof(cudaIpcMemHandle_t));
}

torch::Tensor Buffer::get_local_buffer_u8() const {
  if (!local_allocated_ || num_nvl_bytes_ == 0) {
    throw std::runtime_error("No local NVLink buffer allocated");
  }
  return torch::from_blob(buffer_ptrs_[nvl_rank_], {num_nvl_bytes_},
                          torch::dtype(torch::kUInt8).device(torch::kCUDA));
}

// intranode communication kernel

torch::Tensor Buffer::get_local_buffer_tensor(const py::object& dtype,
                                              int64_t offset,
                                              bool use_rdma_buffer) const {
  auto scalar_type = torch::python::detail::py_object_to_dtype(dtype);
  int64_t elem_size = c10::elementSize(scalar_type);
  void* base_ptr = use_rdma_buffer ? rdma_buffer_ptr_ : buffer_ptrs_[nvl_rank_];
  int64_t num_bytes = use_rdma_buffer ? num_rdma_bytes_ : num_nvl_bytes_;
  if (base_ptr == nullptr || num_bytes == 0) {
    throw std::runtime_error("Requested buffer is not available");
  }
  auto ptr = static_cast<uint8_t*>(base_ptr) + offset;
  return torch::from_blob(
      ptr, num_bytes / elem_size,
      torch::TensorOptions().dtype(scalar_type).device(torch::kCUDA));
}

pybind11::bytearray Buffer::get_local_nvshmem_unique_id() const {
  auto unique_id = nvshmem::get_unique_id();
  return {reinterpret_cast<const char*>(unique_id.data()), unique_id.size()};
}

void Buffer::destroy() {
  HOST_ASSERT(destroyed_ == false);
  // Synchronize
  CUDA_CHECK(cudaDeviceSynchronize());
  // Close CUDA IPC
  if (is_available() && num_nvl_bytes_ > 0) {
    sync::barrier(barrier_signal_ptrs_gpu_, nvl_rank_, num_nvl_ranks_,
                  comm_stream_);

    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < num_nvl_ranks_; i++) {
      if (i != nvl_rank_) CUDA_CHECK(cudaIpcCloseMemHandle(buffer_ptrs_[i]));
    }

    // Free local buffer and error flag
    CUDA_CHECK(cudaFree(buffer_ptrs_[nvl_rank_]));
  }

  // Free NVSHMEM
  if (is_available() && num_rdma_bytes_ > 0) {
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_quiet();
    nvshmem::barrier();
    nvshmem_free(rdma_buffer_ptr_);
    nvshmem_finalize();
  }
  available_ = false;
  destroyed_ = true;
}

int Buffer::get_local_pe() const { return local_pe_; }
int Buffer::get_num_local_pes() const { return num_local_pes_; }
int Buffer::get_local_device_id() const { return device_id_; }
int64_t Buffer::get_num_nvl_bytes() const { return num_nvl_bytes_; }
int Buffer::get_num_device_sms() const { return num_device_sms_; }

int Buffer::get_rdma_rank() const { return rdma_rank_; }
int Buffer::get_num_rdma_ranks() const { return num_rdma_ranks_; }
int Buffer::get_root_rdma_rank(bool global) const {
  return global ? nvl_rank_ : 0;
}

bool Buffer::is_same_rdma_rank(int rank) const {
  return rank / NUM_MAX_NVL_PEERS == rdma_rank_;
}

}  // namespace nvshmem_tutorial
