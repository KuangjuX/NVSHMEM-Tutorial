#pragma once

#include "utils.hpp"

#include <torch/extension.h>

#include <chrono>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace nvshmem_tutorial::nvshmem {

/**
 * Get the unique id of the current process.
 * @return The unique id of the current process.
 */
inline std::vector<uint8_t> get_unique_id() {
  nvshmemx_uniqueid_t unique_id;
  nvshmemx_get_uniqueid(&unique_id);

  std::vector<uint8_t> result(sizeof(unique_id));
  std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
  return result;
}

/**
 * Initialize NVSHMEM with the unique id of the current process.
 * @param unique_id_vec The unique id of the current process.
 * @param rank The rank of the current process.
 * @param num_ranks The number of ranks in the current process group.
 */
inline int init_with_unique_id(const std::vector<uint8_t>& unique_id_vec,
                               int rank, int num_ranks,
                               bool low_latency_mode = false) {
  if (unique_id_vec.size() != sizeof(nvshmemx_uniqueid_t)) {
    throw std::runtime_error("unique_id_vec size mismatch");
  }

  nvshmemx_uniqueid_t root_unique_id;
  nvshmemx_init_attr_t attr;
  std::memcpy(&root_unique_id, unique_id_vec.data(),
              sizeof(nvshmemx_uniqueid_t));
  nvshmemx_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

  // Create sub-RDMA teams
  if (low_latency_mode and num_ranks > NUM_MAX_NVL_PEERS) {
    nvshmem_team_t cpu_rdma_team = NVSHMEM_TEAM_INVALID;
    nvshmem_team_config_t cpu_rdma_team_config;
    HOST_ASSERT(nvshmem_team_split_strided(
                    NVSHMEM_TEAM_WORLD, rank % NUM_MAX_NVL_PEERS,
                    NUM_MAX_NVL_PEERS, num_ranks / NUM_MAX_NVL_PEERS,
                    &cpu_rdma_team_config, 0, &cpu_rdma_team) == 0);
    HOST_ASSERT(cpu_rdma_team != NVSHMEM_TEAM_INVALID);
  }

  nvshmem_barrier_all();
  return nvshmem_my_pe();
}

/**
 * Allocate Symmetric memory with NVSHMEM
 * @param size The size of the memory to allocate.
 * @param alignment The alignment of the memory to allocate.
 * @return A pointer to the allocated memory.
 */
inline void* alloc(size_t size, size_t alignment) {
  return nvshmem_align(alignment, size);
}

inline torch::Tensor alloc_tensor(int64_t size, size_t alignment) {
  void* ptr = alloc(size, alignment);
  return torch::from_blob(ptr, {size},
                          torch::dtype(torch::kUInt8).device(torch::kCUDA));
}

/**
 * Free Symmetric memory with NVSHMEM
 * @param ptr The pointer to the memory to free.
 */
inline void free(void* ptr) { nvshmem_free(ptr); }

inline void free_tensor(torch::Tensor& tensor) {
  void* ptr = tensor.data_ptr();
  free(ptr);
}

/**
 * Barrier with NVSHMEM
 */
inline void barrier() { nvshmem_barrier_all(); }

/**
 * Get memory with NVSHMEM
 * @param local_ptr The pointer to the local memory.
 * @param remote_ptr The pointer to the remote memory.
 * @param nbytes The size of the memory to get.
 * @param rank The rank of the remote memory.
 */
inline void get_mem(void* local_ptr, void* remote_ptr, size_t nbytes,
                    int rank) {
  nvshmem_getmem(local_ptr, remote_ptr, nbytes, rank);
}

/**
 * Get memory with NVSHMEM asynchronously
 * @param local_ptr The pointer to the local memory.
 * @param remote_ptr The pointer to the remote memory.
 * @param nbytes The size of the memory to get.
 * @param rank The rank of the remote memory.
 * @param stream The stream to use for the asynchronous operation.
 */
inline void get_mem_async(void* local_ptr, void* remote_ptr, size_t nbytes,
                          int rank, cudaStream_t stream) {
  nvshmemx_getmem_nbi_on_stream(local_ptr, remote_ptr, nbytes, rank, stream);
}

/**
 * Put memory with NVSHMEM
 * @param dst The pointer to the destination memory.
 * @param src The pointer to the source memory.
 * @param nbytes The size of the memory to put.
 * @param rank The rank of the remote memory.
 */
inline void put_mem(void* remote_ptr, void* local_ptr, size_t nbytes,
                    int rank) {
  nvshmem_putmem(remote_ptr, local_ptr, nbytes, rank);
}

/**
 * Put memory with NVSHMEM asynchronously
 * @param remote_ptr The pointer to the remote memory.
 * @param local_ptr The pointer to the local memory.
 * @param nbytes The size of the memory to put.
 * @param rank The rank of the remote memory.
 * @param stream The stream to use for the asynchronous operation.
 */
inline void put_mem_async(void* remote_ptr, void* local_ptr, size_t nbytes,
                          int rank, cudaStream_t stream) {
  nvshmemx_putmem_nbi_on_stream(remote_ptr, local_ptr, nbytes, rank, stream);
}

/**
 * Get memory with NVSHMEM
 * @param local_tensor The local tensor to store the memory.
 * @param remote_tensor The remote tensor to get the memory from.
 * @param nbytes The size of the memory to get.
 * @param rank The rank of the remote tensor.
 */
inline void get_tensor(torch::Tensor& local_tensor,
                       torch::Tensor& remote_tensor, size_t nbytes, int rank) {
  void* local_ptr = local_tensor.data_ptr();
  void* remote_ptr = remote_tensor.data_ptr();
  get_mem(local_ptr, remote_ptr, nbytes, rank);
}

/**
 * Put memory with NVSHMEM
 * @param local_tensor The local tensor to put the memory from.
 * @param remote_tensor The remote tensor to put the memory to.
 * @param nbytes The size of the memory to put.
 * @param rank The rank of the remote tensor.
 */
inline void put_tensor(torch::Tensor& remote_tensor,
                       torch::Tensor& local_tensor, size_t nbytes, int rank) {
  void* local_ptr = local_tensor.data_ptr();
  void* remote_ptr = remote_tensor.data_ptr();
  put_mem(remote_ptr, local_ptr, nbytes, rank);
}

}  // namespace nvshmem_tutorial::nvshmem
