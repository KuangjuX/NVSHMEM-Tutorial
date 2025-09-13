#pragma once

#include "launch.cuh"
#include "ptx_wrapper.cuh"
#include "utils.hpp"

#include <torch/extension.h>

namespace nvshmem_tutorial::kernels {

template <typename DType, const int kChunkSize>
__global__ void tma_copy_kernel(const DType* input, DType* output,
                                int total_bytes);

template <typename DType, const int kChunkSize>
void tma_copy_host(const DType* input, DType* output, int total_bytes,
                   cudaStream_t stream);

// Experiment APIs for tests.
void tma_copy(const torch::Tensor& input, torch::Tensor& output);
}  // namespace nvshmem_tutorial::kernels
