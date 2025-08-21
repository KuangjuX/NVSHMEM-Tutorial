#pragma once

#include "utils.hpp"

#include <torch/extension.h>

#include <chrono>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

/**
 * Get the unique id of the current process.
 * @return The unique id of the current process.
 */
inline std::vector<int8_t> get_unique_id() {
  nvshmemx_uniqueid_t unique_id;
  nvshmemx_get_uniqueid(&unique_id);

  // Convert the unique_id to a vector of int8_t
  const int8_t* id_ptr = reinterpret_cast<const int8_t*>(&unique_id);
  return std::vector<int8_t>(id_ptr, id_ptr + sizeof(unique_id));
}

/**
 * Initialize NVSHMEM with the unique id of the current process.
 * @param unique_id_vec The unique id of the current process.
 * @param rank The rank of the current process.
 * @param num_ranks The number of ranks in the current process group.
 */
inline void init_with_unique_id(const std::vector<int8_t>& unique_id_vec,
                                int rank, int num_ranks) {
  if (unique_id_vec.size() != sizeof(nvshmemx_uniqueid_t)) {
    throw std::runtime_error("unique_id_vec size mismatch");
  }
  // Convert vector<int8_t> back to nvshmemx_uniqueid_t
  nvshmemx_uniqueid_t unique_id = NVSHMEMX_UNIQUEID_INITIALIZER;
  memcpy(&unique_id, unique_id_vec.data(), sizeof(unique_id));

  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  nvshmemx_set_attr_uniqueid_args(rank, num_ranks, &unique_id, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
  int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  assert(mype_node == rank);
  nvshmem_barrier_all();
}

namespace nvshmem_tutorial::nvshmem {

/**
 * Allocate Symmetric memory with NVSHMEM
 * @param size The size of the memory to allocate.
 * @param alignment The alignment of the memory to allocate.
 * @return A pointer to the allocated memory.
 */
void* alloc(size_t size, size_t alignment) {
  return nvshmem_align(alignment, size);
}

/**
 * Free Symmetric memory with NVSHMEM
 * @param ptr The pointer to the memory to free.
 */
void free(void* ptr) { nvshmem_free(ptr); }

/**
 * Barrier with NVSHMEM
 */
void barrier() { nvshmem_barrier_all(); }

/**
 * Get memory with NVSHMEM
 * @param local_ptr The pointer to the local memory.
 * @param remote_ptr The pointer to the remote memory.
 * @param nbytes The size of the memory to get.
 * @param rank The rank of the remote memory.
 */
void get_mem(void* local_ptr, void* remote_ptr, size_t nbytes, int rank) {
  nvshmem_getmem(local_ptr, remote_ptr, nbytes, rank);
}

/**
 * Put memory with NVSHMEM
 * @param dst The pointer to the destination memory.
 * @param src The pointer to the source memory.
 * @param nbytes The size of the memory to put.
 * @param rank The rank of the remote memory.
 */
void put_mem(void* remote_ptr, void* local_ptr, size_t nbytes, int rank) {
  nvshmem_putmem(remote_ptr, local_ptr, nbytes, rank);
}

/**
 * Get memory with NVSHMEM
 * @param local_tensor The local tensor to store the memory.
 * @param remote_tensor The remote tensor to get the memory from.
 * @param nbytes The size of the memory to get.
 * @param rank The rank of the remote tensor.
 */
void get_mem_tensor(torch::Tensor& local_tensor, torch::Tensor& remote_tensor,
                    size_t nbytes, int rank) {
  void* local_ptr = local_tensor.data_ptr();
  void* remote_ptr = remote_tensor.data_ptr();
  get_mem(local_ptr, remote_ptr, nbytes, rank);
}

void put_mem_tensor(torch::Tensor& local_tensor, torch::Tensor& remote_tensor,
                    size_t nbytes, int rank) {
  void* local_ptr = local_tensor.data_ptr();
  void* remote_ptr = remote_tensor.data_ptr();
  put_mem(remote_ptr, local_ptr, nbytes, rank);
}

/**
 * Put memory with NVSHMEM
 * @param local_tensor The local tensor to put the memory from.
}  // namespace nvshmem_tutorial::nvshmem