#pragma once

#include "utils.hpp"

#include <torch/extension.h>

#include <chrono>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// ===================================================================
// Part 1: Manual Bootstrap for NVSHMEM with torchrun
// ===================================================================

// 获取 unique id，返回 vector<int8_t>
inline std::vector<int8_t> get_unique_id() {
  nvshmemx_uniqueid_t unique_id;
  nvshmemx_get_uniqueid(&unique_id);

  // Convert the unique_id to a vector of int8_t
  const int8_t* id_ptr = reinterpret_cast<const int8_t*>(&unique_id);
  return std::vector<int8_t>(id_ptr, id_ptr + sizeof(unique_id));
}

// 用 unique id 初始化
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
