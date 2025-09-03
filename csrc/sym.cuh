#include "utils.hpp"

namespace nvshmem_tutorial {

template <typename DType>
struct SymLayout {
 public:
  SymLayout(void*& ptr, imt num_elems, int num_ranks) {
    num_bytes = num_elems * sizeof(DType);

    int per_channel_bytes = num_bytes * num_ranks;
    total_bytes = per_channel_bytes * 2;
    send_ptr = reinterpret_cast<uint8_t*>(ptr);
    recv_ptr = send_ptr + per_channel_bytes;
    ptr = reinterpret_cast<uint8_t*>(ptr) + total_bytes;
  }

  DEVICE DType* send_buffer(int idx = 0) {
    return reinterpret_cast<DType*>(send_ptr + idx * num_bytes);
  }

  DEVICE DType* recv_buffer(int idx = 0) {
    return reinterpret_cast<DType*>(recv_ptr + idx * num_bytes);
  }

 private:
  uint8_t* send_ptr;
  uint8_t* recv_ptr;

  int num_bytes;

  int total_bytes;
};

}  // namespace nvshmem_tutorial