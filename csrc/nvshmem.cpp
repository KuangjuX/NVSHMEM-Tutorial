#include <torch/extension.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <vector>
#include <stdexcept>
#include <chrono>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>

namespace py = pybind11;

// ===================================================================
// Part 1: Manual Bootstrap for NVSHMEM with torchrun
// ===================================================================

// 获取 unique id，返回 vector<int8_t>
inline std::vector<int8_t> get_unique_id()
{
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);

    // Convert the unique_id to a vector of int8_t
    const int8_t *id_ptr = reinterpret_cast<const int8_t *>(&unique_id);
    return std::vector<int8_t>(id_ptr, id_ptr + sizeof(unique_id));
}

// 用 unique id 初始化
inline void init_with_unique_id(
    const std::vector<int8_t> &unique_id_vec, int rank, int num_ranks)
{
    if (unique_id_vec.size() != sizeof(nvshmemx_uniqueid_t))
    {
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

// ===================================================================
// Part 2: NVSHMEM Operations for Benchmarking (No changes needed here)
// ===================================================================

torch::Tensor alloc_symmetric(int64_t size_bytes)
{
    void *ptr = nvshmem_malloc(size_bytes);
    if (!ptr)
    {
        throw std::runtime_error("nvshmem_malloc failed!");
    }
    auto options = torch::TensorOptions()
                       .device(torch::kCUDA)
                       .dtype(torch::kUInt8);
    return torch::from_blob(ptr, {size_bytes}, [ptr](void *)
                            { nvshmem_free(ptr); }, options);
}

void put_blocking(torch::Tensor dst, torch::Tensor src, int dst_pe)
{
    nvshmem_putmem(dst.data_ptr(), src.data_ptr(), src.nbytes(), dst_pe);
    nvshmem_quiet();
}

void finalize()
{
    nvshmem_finalize();
}

// ===================================================================
// Part 3: Pybind11 Module Definition
// ===================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "NVSHMEM bindings for benchmarking with torchrun";

    // *** MODIFICATION FOR CONSTANT ***
    // Instead of a macro, use sizeof() which is more robust
    m.attr("UNIQUE_ID_LEN") = py::int_(sizeof(nvshmemx_uniqueid_t));

    // Bootstrap functions
    m.def("get_unique_id", &get_unique_id, "Get a unique ID for NVSHMEM initialization (call on rank 0)");
    m.def("init_with_unique_id", &init_with_unique_id, "Initialize NVSHMEM using a unique ID",
          py::arg("unique_id_vec"), py::arg("rank"), py::arg("world_size"));

    // NVSHMEM operations
    m.def("alloc_symmetric", &alloc_symmetric, "Allocate a symmetric tensor", py::arg("size_bytes"));
    m.def("put_blocking", &put_blocking, "Perform a blocking put operation", py::arg("dst"), py::arg("src"), py::arg("dst_pe"));
    m.def("finalize", &finalize, "Finalize NVSHMEM");

    // Utility functions
    m.def("my_pe", &nvshmem_my_pe, "Get my processing element (PE) ID");
    m.def("n_pes", &nvshmem_n_pes, "Get the number of PEs");
    m.def("barrier_all", &nvshmem_barrier_all, "Barrier across all PEs");
}