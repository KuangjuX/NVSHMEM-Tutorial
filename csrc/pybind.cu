
#include "buffer.cuh"
#include "nvshmem.hpp"

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace nvshmem_tutorial;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "NVSHMEM bindings for benchmarking with torchrun";

  m.attr("UNIQUE_ID_LEN") = py::int_(sizeof(nvshmemx_uniqueid_t));

  py::class_<Buffer>(m, "Buffer")
      .def(py::init<int, int, int64_t, int64_t>())
      .def("get_local_ipc_handle", &Buffer::get_local_ipc_handle)
      .def("get_local_buffer_u8", &Buffer::get_local_buffer_u8)
      .def("sync", &Buffer::sync)
      // Intranode communication
      .def("intranode_all_gather", &Buffer::intranode_all_gather)
      .def("intranode_all_to_all", &Buffer::intranode_all_to_all)
      // Internode communication
      .def("internode_all_gather", &Buffer::internode_all_gather)
      .def("destroy", &Buffer::destroy)
      // Introspection
      .def("is_available", &Buffer::is_available)
      .def("get_local_buffer_tensor", &Buffer::get_local_buffer_tensor)
      .def("get_local_nvshmem_unique_id", &Buffer::get_local_nvshmem_unique_id)
      .def("get_local_pe", &Buffer::get_local_pe)
      .def("get_local_device_id", &Buffer::get_local_device_id)
      .def("get_num_device_sms", &Buffer::get_num_device_sms)
      .def("get_num_local_pes", &Buffer::get_num_local_pes)
      .def("get_num_nvl_bytes", &Buffer::get_num_nvl_bytes)
      // RDMA rank
      .def("get_rdma_rank", &Buffer::get_rdma_rank)
      .def("get_num_rdma_ranks", &Buffer::get_num_rdma_ranks)
      .def("get_root_rdma_rank", &Buffer::get_root_rdma_rank);

  // Native API
  m.def("get_unique_id", &nvshmem::get_unique_id,
        "Get a unique ID for NVSHMEM initialization (call on rank 0)");
  m.def("init_with_unique_id", &nvshmem::init_with_unique_id,
        "Initialize NVSHMEM using a unique ID", py::arg("unique_id_vec"),
        py::arg("rank"), py::arg("world_size"), py::arg("low_latency_mode"));

  m.def("nvshmem_alloc_tensor", &nvshmem::alloc_tensor,
        "Allocate Symmetric memory with NVSHMEM", py::arg("size"),
        py::arg("alignment"));
  m.def("nvshmem_free_tensor", &nvshmem::free_tensor,
        "Free Symmetric memory with NVSHMEM", py::arg("tensor"));
  m.def("nvshmem_barrier", &nvshmem::barrier, "Barrier with NVSHMEM");

  m.def("nvshmem_get_tensor", &nvshmem::get_tensor, "Get memory with NVSHMEM",
        py::arg("local_tensor"), py::arg("remote_tensor"), py::arg("nbytes"),
        py::arg("rank"));
  m.def("nvshmem_put_tensor", &nvshmem::put_tensor, "Put memory with NVSHMEM",
        py::arg("remote_tensor"), py::arg("local_tensor"), py::arg("nbytes"),
        py::arg("rank"));

  m.def("nvshmem_get_tensor_async", &nvshmem::get_tensor_async,
        "Get memory with NVSHMEM asynchronously", py::arg("local_tensor"),
        py::arg("remote_tensor"), py::arg("nbytes"), py::arg("rank"));
  m.def("nvshmem_put_tensor_async", &nvshmem::put_tensor_async,
        "Put memory with NVSHMEM asynchronously", py::arg("remote_tensor"),
        py::arg("local_tensor"), py::arg("nbytes"), py::arg("rank"));
}
