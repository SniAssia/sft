// bindings.cpp
// pybind11 module `uds_loader`. Built as a torch extension so torch::Tensor
// members auto-convert to torch.Tensor on the Python side.
"""bindings.cpp serves as a bridge between Python and your C++ implementation. It:

Registers the PipelineConfig, CollatedPool, and DataPipeline C++ classes as Python classes.
Exposes configuration fields (such as B, pad_id, and shards) so they can be read and modified directly from Python.
Automatically converts torch::Tensor objects in C++ into torch.Tensor objects in Python.
Exposes pipeline methods (start, stop, next_pool) so the training loop can control the C++ pipeline.
Makes internal performance metrics (formation time, stalls, queue size, streamed samples, etc.) accessible for benchmarking.
Releases Python's Global Interpreter Lock while next_pool() waits for a batch, allowing other Python threads to continue running."""
#include <torch/extension.h>
#include <pybind11/stl.h>

#include "data_pipeline.hpp"

namespace py = pybind11;
using namespace uds;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "UDS SFT C++ data pipeline (reader/queues/scheduler/collator/prefetch/DDP)";
    py::class_<PipelineConfig>(m, "PipelineConfig")
        .def(py::init<>())
        .def_readwrite("shards", &PipelineConfig::shards)
        .def_readwrite("rank", &PipelineConfig::rank)
        .def_readwrite("world_size", &PipelineConfig::world_size)
        .def_readwrite("seed", &PipelineConfig::seed)
        .def_readwrite("num_epochs", &PipelineConfig::num_epochs)
        .def_readwrite("max_queue_occupancy", &PipelineConfig::max_queue_occupancy)
        .def_readwrite("B", &PipelineConfig::B)
        .def_readwrite("profile_bands", &PipelineConfig::profile_bands)
        .def_readwrite("profile_mix", &PipelineConfig::profile_mix)
        .def_readwrite("profile_is_chunked", &PipelineConfig::profile_is_chunked)
        .def_readwrite("baseline", &PipelineConfig::baseline)
        .def_readwrite("resident_window", &PipelineConfig::resident_window)
        .def_readwrite("band_cutoffs", &PipelineConfig::band_cutoffs)
        .def_readwrite("band_max_seq_len", &PipelineConfig::band_max_seq_len)
        .def_readwrite("pad_id", &PipelineConfig::pad_id)
        .def_readwrite("ignore_index", &PipelineConfig::ignore_index)
        .def_readwrite("option_b_window", &PipelineConfig::option_b_window)
        .def_readwrite("pad_to_multiple", &PipelineConfig::pad_to_multiple)
        .def_readwrite("prefetch_workers", &PipelineConfig::prefetch_workers)
        .def_readwrite("ring_capacity", &PipelineConfig::ring_capacity);
    py::class_<CollatedPool, std::shared_ptr<CollatedPool>>(m, "CollatedPool")
        .def_readonly("input_ids", &CollatedPool::input_ids)
        .def_readonly("attention_mask", &CollatedPool::attention_mask)
        .def_readonly("labels", &CollatedPool::labels)
        .def_readonly("is_chunked", &CollatedPool::is_chunked)
        .def_readonly("band", &CollatedPool::band)
        .def_readonly("profile_index", &CollatedPool::profile_index)
        .def_readonly("mixed", &CollatedPool::mixed)
        .def_readonly("fell_back", &CollatedPool::fell_back)
        .def_readonly("baseline", &CollatedPool::baseline)
        .def_readonly("batch_size", &CollatedPool::batch_size)
        .def_readonly("padded_width", &CollatedPool::padded_width)
        .def_readonly("padded_tokens", &CollatedPool::padded_tokens)
        .def_readonly("real_tokens", &CollatedPool::real_tokens)
        .def_readonly("pad_tokens", &CollatedPool::pad_tokens)
        .def_readonly("prompt_len", &CollatedPool::prompt_len)
        .def_readonly("context_len", &CollatedPool::context_len)
        .def_readonly("response_len", &CollatedPool::response_len)
        .def_readonly("is_chunked_flags", &CollatedPool::is_chunked_flags)
        .def_readonly("case_codes", &CollatedPool::case_codes)
        .def_readonly("formation_seconds", &CollatedPool::formation_seconds)
        .def("__len__", [](const CollatedPool& p) { return p.input_ids.size(0); });
    py::class_<DataPipeline>(m, "DataPipeline")
        .def(py::init<PipelineConfig>())
        .def("start", &DataPipeline::start)
        .def("stop", &DataPipeline::stop)
        // Release the GIL while blocking on the prefetch ring so Python threads run.
        .def("next_pool", &DataPipeline::next_pool,
             py::call_guard<py::gil_scoped_release>())
        .def("formation_total_s", &DataPipeline::formation_total_s)
        .def("formation_mean_ms", &DataPipeline::formation_mean_ms)
        .def("formation_count", &DataPipeline::formation_count)
        .def("stall_total_s", &DataPipeline::stall_total_s)
        .def("stall_mean_ms", &DataPipeline::stall_mean_ms)
        .def("samples_streamed", &DataPipeline::samples_streamed)
        .def("queue_size", &DataPipeline::queue_size)
        .def("empty_alerts", &DataPipeline::empty_alerts)
        .def("fallback_pools", &DataPipeline::fallback_pools)
        .def("skipped_categories", &DataPipeline::skipped_categories)
        .def("streamer_done", &DataPipeline::streamer_done);
}