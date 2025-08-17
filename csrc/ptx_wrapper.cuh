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

}  // namespace nvshmem_tutorial
