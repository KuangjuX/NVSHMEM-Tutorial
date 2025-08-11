#include "buffer.cuh"

#include <cstring>

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
      num_rdma_bytes_{num_rdma_bytes} {
  CUDA_CHECK(cudaGetDevice(&device_id_));
  query_local_pe(local_pe_, num_local_pes_);

  rdma_rank_ = rank / NUM_MAX_NVL_PEERS;
  nvl_rank_ = rank % NUM_MAX_NVL_PEERS;
  num_rdma_ranks_ = std::max(1, num_ranks / NUM_MAX_NVL_PEERS);
  num_nvl_ranks_ = std::min(num_ranks, NUM_MAX_NVL_PEERS);

  if (num_nvl_bytes_ > 0) {
    CUDA_CHECK(cudaMalloc(&buffer_ptrs_[nvl_rank_], num_nvl_bytes_));
    local_allocated_ = true;
    CUDA_CHECK(
        cudaIpcGetMemHandle(&ipc_handles_[nvl_rank_], buffer_ptrs_[nvl_rank_]));
  }
}

Buffer::~Buffer() {
  // Close peer handles
  for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i) {
    if (i != nvl_rank_ && buffer_ptrs_[i] != nullptr) {
      cudaIpcCloseMemHandle(buffer_ptrs_[i]);
      buffer_ptrs_[i] = nullptr;
    }
  }
  if (local_allocated_ && buffer_ptrs_[nvl_rank_] != nullptr) {
    CUDA_CHECK(cudaFree(buffer_ptrs_[nvl_rank_]));
    buffer_ptrs_[nvl_rank_] = nullptr;
  }
}

torch::Tensor Buffer::alloc_symmetric(int64_t size_bytes) {
  void* ptr = nvshmem_malloc(size_bytes);
  if (ptr == nullptr) {
    throw std::runtime_error("nvshmem_malloc returned nullptr");
  }
  return torch::from_blob(ptr, {size_bytes},
                          torch::dtype(torch::kUInt8).device(torch::kCUDA));
}

void Buffer::free_symmetric(torch::Tensor t) { nvshmem_free(t.data_ptr()); }

// py::bytearray Buffer::get_local_nvshmem_unique_id() const {
//   // Mimic DeepEP: only RDMA root (rdma_rank==0) allowed to export unique id
//   if (rdma_rank_ != 0) {
//     throw std::runtime_error("Only RDMA rank 0 can get NVSHMEM unique ID");
//   }
//   auto uid = get_unique_id();
//   return py::bytearray(reinterpret_cast<const char*>(uid.data()),
//   uid.size());
// }

// void Buffer::sync(
//     const std::vector<int>& device_ids,
//     const std::vector<std::optional<py::bytearray>>& all_gathered_handles,
//     const std::optional<py::bytearray>& root_unique_id_opt) {
//   // Open CUDA IPC peers
//   if (num_nvl_bytes_ > 0) {
//     if (static_cast<int>(device_ids.size()) != num_ranks_) {
//       throw std::runtime_error("sync: device_ids size mismatch");
//     }
//     if (static_cast<int>(all_gathered_handles.size()) != num_ranks_) {
//       throw std::runtime_error("sync: handles size mismatch");
//     }
//     // Map only peers within our NVL group
//     int offset = rdma_rank_ * num_nvl_ranks_;
//     for (int i = 0; i < num_nvl_ranks_; ++i) {
//       auto peer_rank = offset + i;
//       auto handle_str = std::string(
//           all_gathered_handles[peer_rank].value().cast<std::string>());
//       if (peer_rank != rank_) {
//         if (handle_str.size() != sizeof(cudaIpcMemHandle_t)) {
//           throw std::runtime_error("CUDA IPC handle size mismatch");
//         }
//         std::memcpy(ipc_handles_[i].reserved, handle_str.data(),
//                     sizeof(cudaIpcMemHandle_t));
//         CUDA_CHECK(cudaIpcOpenMemHandle(&buffer_ptrs_[i], ipc_handles_[i],
//                                         cudaIpcMemLazyEnablePeerAccess));
//       }
//     }
//     CUDA_CHECK(cudaDeviceSynchronize());
//   }

//   // Initialize NVSHMEM and allocate RDMA buffer
//   if (num_rdma_bytes_ > 0) {
//     if (!root_unique_id_opt.has_value()) {
//       throw std::runtime_error("sync: missing root NVSHMEM unique id");
//     }
//     auto uid_str = root_unique_id_opt->cast<std::string>();
//     std::vector<int8_t> uid_vec(uid_str.begin(), uid_str.end());

//     int nvshmem_rank = rdma_rank_;
//     int nvshmem_world = num_rdma_ranks_;
//     init_with_unique_id(uid_vec, nvshmem_rank, nvshmem_world);

//     nvshmem_barrier_all();
//     rdma_buffer_ptr_ = nvshmem_malloc(num_rdma_bytes_);
//     CUDA_CHECK(cudaMemset(rdma_buffer_ptr_, 0, num_rdma_bytes_));
//     nvshmem_barrier_all();
//   }

//   available_ = true;
// }

py::bytearray Buffer::get_local_ipc_handle() const {
  if (!local_allocated_ || num_nvl_bytes_ == 0) {
    throw std::runtime_error("No local NVLink buffer allocated");
  }
  return py::bytearray(
      reinterpret_cast<const char*>(ipc_handles_[nvl_rank_].reserved),
      sizeof(cudaIpcMemHandle_t));
}

void Buffer::open_ipc_handles(
    const std::vector<std::optional<py::bytearray>>& all_handles) {
  if (num_nvl_bytes_ == 0) return;
  if (static_cast<int>(all_handles.size()) != num_local_pes_) {
    throw std::runtime_error(
        "open_ipc_handles: size mismatch with num_local_pes");
  }
  for (int pe = 0; pe < num_local_pes_; ++pe) {
    if (pe == local_pe_) continue;
    if (!all_handles[pe].has_value()) {
      throw std::runtime_error("Missing IPC handle for peer");
    }
    auto h = std::string(all_handles[pe]->cast<std::string>());
    if (h.size() != sizeof(cudaIpcMemHandle_t)) {
      throw std::runtime_error("IPC handle size mismatch");
    }
    std::memcpy(ipc_handles_[pe].reserved, h.data(),
                sizeof(cudaIpcMemHandle_t));
    CUDA_CHECK(cudaIpcOpenMemHandle(&buffer_ptrs_[pe], ipc_handles_[pe],
                                    cudaIpcMemLazyEnablePeerAccess));
  }
}

void Buffer::intranode_memcpy_to(int dst_local_pe, int64_t dst_offset_bytes,
                                 torch::Tensor src) {
  if (dst_local_pe < 0 || dst_local_pe >= num_local_pes_) {
    throw std::runtime_error("dst_local_pe out of range");
  }
  if (buffer_ptrs_[dst_local_pe] == nullptr) {
    throw std::runtime_error("Destination peer buffer not mapped");
  }
  if (!src.is_cuda()) {
    throw std::runtime_error("src must be CUDA tensor");
  }
  int64_t nbytes = src.nbytes();
  void* dst =
      static_cast<uint8_t*>(buffer_ptrs_[dst_local_pe]) + dst_offset_bytes;
  CUDA_CHECK(cudaMemcpy(dst, src.data_ptr(), nbytes, cudaMemcpyDeviceToDevice));
}

torch::Tensor Buffer::get_local_buffer_u8() const {
  if (!local_allocated_ || num_nvl_bytes_ == 0) {
    throw std::runtime_error("No local NVLink buffer allocated");
  }
  return torch::from_blob(buffer_ptrs_[nvl_rank_], {num_nvl_bytes_},
                          torch::dtype(torch::kUInt8).device(torch::kCUDA));
}

void Buffer::internode_put(torch::Tensor dst_symmetric, torch::Tensor src,
                           int dst_pe) {
  if (!src.is_cuda() || !dst_symmetric.is_cuda()) {
    throw std::runtime_error("internode_put expects CUDA tensors");
  }
  nvshmem_putmem(dst_symmetric.data_ptr(), src.data_ptr(), src.nbytes(),
                 dst_pe);
  nvshmem_quiet();
}

void Buffer::internode_get(torch::Tensor dst, torch::Tensor src_symmetric,
                           int src_pe) {
  if (!dst.is_cuda() || !src_symmetric.is_cuda()) {
    throw std::runtime_error("internode_get expects CUDA tensors");
  }
  nvshmem_getmem(dst.data_ptr(), src_symmetric.data_ptr(), dst.nbytes(),
                 src_pe);
  nvshmem_quiet();
}

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

void Buffer::destroy() {
  CUDA_CHECK(cudaDeviceSynchronize());
  // Close CUDA IPC
  for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i) {
    if (i != nvl_rank_ && buffer_ptrs_[i] != nullptr) {
      cudaIpcCloseMemHandle(buffer_ptrs_[i]);
      buffer_ptrs_[i] = nullptr;
    }
  }
  if (local_allocated_ && buffer_ptrs_[nvl_rank_] != nullptr) {
    CUDA_CHECK(cudaFree(buffer_ptrs_[nvl_rank_]));
    buffer_ptrs_[nvl_rank_] = nullptr;
  }
  // Free NVSHMEM
  if (rdma_buffer_ptr_ != nullptr) {
    nvshmem_barrier_all();
    nvshmem_free(rdma_buffer_ptr_);
    rdma_buffer_ptr_ = nullptr;
  }
  available_ = false;
}

int Buffer::get_local_pe() const { return local_pe_; }
int Buffer::get_num_local_pes() const { return num_local_pes_; }
int Buffer::get_device_id() const { return device_id_; }
int64_t Buffer::get_num_nvl_bytes() const { return num_nvl_bytes_; }