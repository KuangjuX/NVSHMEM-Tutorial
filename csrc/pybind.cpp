
#include "nvshmem.hpp"
#include "put.cuh"

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// ===================================================================
// Part 3: Pybind11 Module Definition
// ===================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "NVSHMEM bindings for benchmarking with torchrun";

  // *** MODIFICATION FOR CONSTANT ***
  // Instead of a macro, use sizeof() which is more robust
  m.attr("UNIQUE_ID_LEN") = py::int_(sizeof(nvshmemx_uniqueid_t));

  // Bootstrap functions
  m.def("get_unique_id", &get_unique_id,
        "Get a unique ID for NVSHMEM initialization (call on rank 0)");
  m.def("init_with_unique_id", &init_with_unique_id,
        "Initialize NVSHMEM using a unique ID", py::arg("unique_id_vec"),
        py::arg("rank"), py::arg("world_size"));

  // NVSHMEM operations
  m.def("alloc_symmetric", &alloc_symmetric, "Allocate a symmetric tensor",
        py::arg("size_bytes"));
  m.def("finalize", &finalize, "Finalize NVSHMEM");

  // Utility functions
  m.def("my_pe", &nvshmem_my_pe, "Get my processing element (PE) ID");
  m.def("n_pes", &nvshmem_n_pes, "Get the number of PEs");
  m.def("barrier_all", &nvshmem_barrier_all, "Barrier across all PEs");

  m.def("put_blocking", &put_blocking, "Perform a blocking put operation",
        py::arg("dst"), py::arg("src"), py::arg("dst_pe"));
  m.def("launch_ring_put_block", &launch_ring_put_block,
        "Launch a ring put block operation", py::arg("send_tensor"),
        py::arg("recv_tensor"), py::arg("num_blocks"),
        py::arg("threads_per_block"));
}