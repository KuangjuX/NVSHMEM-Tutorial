
#include "buffer.cuh"
#include "nvshmem.hpp"
#include "put.cuh"

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace nvshmem_tutorial;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "NVSHMEM bindings for benchmarking with torchrun";

  m.attr("UNIQUE_ID_LEN") = py::int_(sizeof(nvshmemx_uniqueid_t));

  py::class_<Buffer>(m, "Buffer")
      .def(py::init<int, int, int64_t, int64_t>())
      .def("alloc_symmetric", &Buffer::alloc_symmetric)
      .def("free_symmetric", &Buffer::free_symmetric)
      .def("get_local_ipc_handle", &Buffer::get_local_ipc_handle)
      .def("get_local_buffer_u8", &Buffer::get_local_buffer_u8)
      .def("sync", &Buffer::sync)
      // Intranode communication
      .def("intranode_all_gather", &Buffer::intranode_all_gather)
      .def("intranode_all_to_all", &Buffer::intranode_all_to_all)
      .def("destroy", &Buffer::destroy)
      // Introspection
      .def("is_available", &Buffer::is_available)
      .def("get_local_buffer_tensor", &Buffer::get_local_buffer_tensor)
      .def("get_local_pe", &Buffer::get_local_pe)
      .def("get_local_device_id", &Buffer::get_local_device_id)
      .def("get_num_device_sms", &Buffer::get_num_device_sms)
      .def("get_num_local_pes", &Buffer::get_num_local_pes)
      .def("get_num_nvl_bytes", &Buffer::get_num_nvl_bytes);

  // Native API
  m.def("get_unique_id", &get_unique_id,
        "Get a unique ID for NVSHMEM initialization (call on rank 0)");
  m.def("init_with_unique_id", &init_with_unique_id,
        "Initialize NVSHMEM using a unique ID", py::arg("unique_id_vec"),
        py::arg("rank"), py::arg("world_size"));

  m.def("nvshmem_alloc", &nvshmem::alloc,
        "Allocate Symmetric memory with NVSHMEM", py::arg("size"),
        py::arg("alignment"));
  m.def("nvshmem_free", &nvshmem::free, "Free Symmetric memory with NVSHMEM",
        py::arg("ptr"));
  m.def("nvshmem_barrier", &nvshmem::barrier, "Barrier with NVSHMEM");

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
