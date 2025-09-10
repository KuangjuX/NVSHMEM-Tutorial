#pragma once

#include "utils.hpp"

namespace nvshmem_tutorial {

__device__ __forceinline__ void trap() { asm("trap;"); }

__device__ __forceinline__ int ld_volatile_global(const int* ptr) {
  int ret;
  asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ float ld_volatile_global(const float* ptr) {
  float ret;
  asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(const int64_t* ptr) {
  int64_t ret;
  asm volatile("ld.volatile.global.s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(const uint64_t* ptr) {
  int64_t ret;
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ void memory_fence() {
  asm volatile("fence.acq_rel.sys;" ::: "memory");
}

constexpr uint64_t kEvictFirst = 0x12f0000000000000;
constexpr uint64_t kEvictNormal = 0x1000000000000000;

__device__ __forceinline__ void tma_load_1d(const void* gmem_ptr,
                                            const void* smem_ptr,
                                            uint64_t* mbar_ptr, int num_bytes,
                                            bool evict_first = false) {
  auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes."
      "L2::"
      "cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(smem_int_ptr),
      "l"(gmem_ptr), "r"(num_bytes), "r"(mbar_int_ptr), "l"(cache_hint)
      : "memory");
}

__device__ __forceinline__ void tma_store_1d(const void* smem_ptr,
                                             const void* gmem_ptr,
                                             int num_bytes,
                                             bool evict_first = false) {
  auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;
  asm volatile(
      "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], "
      "%2, %3;\n" ::"l"(gmem_ptr),
      "r"(smem_int_ptr), "r"(num_bytes), "l"(cache_hint)
      : "memory");
  asm volatile("cp.async.bulk.commit_group;");
}

}  // namespace nvshmem_tutorial
