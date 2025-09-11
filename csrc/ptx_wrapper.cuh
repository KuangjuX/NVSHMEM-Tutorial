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

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar_ptr,
                                              uint32_t arrive_count) {
  auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" ::"r"(arrive_count),
               "r"(mbar_int_ptr));
}

template <bool kWithMultiStages = false>
__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar_ptr,
                                              uint32_t& phase,
                                              int stage_idx = 0) {
  auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  const auto& wait = kWithMultiStages ? (phase >> stage_idx) & 1 : phase;
  asm volatile(
      "{\n\t"
      ".reg .pred       P1; \n\t"
      "LAB_WAIT: \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
      "@P1 bra DONE; \n\t"
      "bra     LAB_WAIT; \n\t"
      "DONE: \n\t"
      "}" ::"r"(mbar_int_ptr),
      "r"(wait), "r"(0x989680));
  phase ^= kWithMultiStages ? (1 << stage_idx) : 1;
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx(
    uint64_t* mbar_ptr, int num_bytes) {
  auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::"r"(
          num_bytes),
      "r"(mbar_int_ptr));
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar_ptr) {
  auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  asm volatile(
      "mbarrier.arrive.shared::cta.b64 _, [%0]; \n\t" ::"r"(mbar_int_ptr));
}

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

template <int N = 0>
__device__ __forceinline__ void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ void tma_store_fence() {
  asm volatile("fence.proxy.async.shared::cta;");
}

}  // namespace nvshmem_tutorial
