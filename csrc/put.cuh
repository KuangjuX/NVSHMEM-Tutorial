#pragma once

#include "utils.hpp"

#include <torch/extension.h>

void put_blocking(torch::Tensor dst, torch::Tensor src, int dst_pe) {
  nvshmem_putmem(dst.data_ptr(), src.data_ptr(), src.nbytes(), dst_pe);
  nvshmem_quiet();
}

float launch_ring_put_block(torch::Tensor send_tensor,
                            torch::Tensor recv_tensor, int num_blocks,
                            int threads_per_block);