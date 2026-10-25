/**
 * tyr_distributed.cpp - Distributed training primitives for Tyr
 *
 * Provides Lean FFI bindings for PyTorch distributed communication:
 * - Process group initialization and management
 * - Collective operations (reduce_scatter, all_gather, all_reduce, broadcast)
 * - Async operation handling with futures
 */

#include <lean/lean.h>
#include <torch/torch.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#if defined(USE_C10D_NCCL)
#include <c10/cuda/CUDAFunctions.h>
#endif

#if defined(USE_C10D_NCCL)
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#define TYR_HAS_NCCL 1
#endif

#if defined(USE_C10D_GLOO)
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#define TYR_HAS_GLOO 1
#endif

#ifndef TYR_HAS_NCCL
#define TYR_HAS_NCCL 0
#endif

#ifndef TYR_HAS_GLOO
#define TYR_HAS_GLOO 0
#endif

#include <memory>
#include <string>
#include <mutex>
#include <limits>
#include <unordered_map>

// Forward declarations from tyr.cpp
extern lean_object* fromTorchTensor(torch::Tensor t);
extern torch::Tensor borrowTensor(b_lean_obj_arg o);

namespace {

// Global process group state
c10::intrusive_ptr<c10d::Backend> g_process_group;
c10::intrusive_ptr<c10d::Store> g_store;
int g_rank = 0;
int g_world_size = 1;
bool g_initialized = false;
// Never reused; graphs retain only this number, never a backend/store owner.
int64_t g_process_group_generation = 0;
std::mutex g_dist_mutex;

// Future storage for async operations
std::unordered_map<uint64_t, c10::intrusive_ptr<c10d::Work>> g_pending_work;
std::atomic<uint64_t> g_next_work_id{1};

// Reduce operations mapping
c10d::ReduceOp::RedOpType reduceOpFromInt(uint8_t op) {
    switch (op) {
        case 0: return c10d::ReduceOp::SUM;
        case 1: return c10d::ReduceOp::AVG;
        case 2: return c10d::ReduceOp::PRODUCT;
        case 3: return c10d::ReduceOp::MIN;
        case 4: return c10d::ReduceOp::MAX;
        default: return c10d::ReduceOp::SUM;
    }
}

void maybeSetCurrentCudaDeviceFromTensor(const torch::Tensor& t) {
#if TYR_HAS_NCCL
    if (t.is_cuda()) {
        c10::cuda::set_device(static_cast<c10::DeviceIndex>(t.get_device()));
    }
#else
    (void)t;
#endif
}

// All helpers below run with g_dist_mutex held. The mutex covers validation,
// submission, and wait, so teardown cannot replace a group during a mapping.
// It does not coordinate ordering across ranks: callers must serialize the
// entire distributed forward/backward schedule identically on every rank.
void requireTpGeneration(int64_t generation) {
    TORCH_CHECK(g_initialized && g_process_group &&
                generation == g_process_group_generation,
                "Tensor-parallel autograd process-group generation is no longer active");
}

void sumTpTensor(torch::Tensor& tensor) {
    maybeSetCurrentCudaDeviceFromTensor(tensor);
    std::vector<torch::Tensor> tensors{tensor};
    c10d::AllreduceOptions options;
    options.reduceOp = c10d::ReduceOp::SUM;
    auto work = g_process_group->allreduce(tensors, options);
    TORCH_CHECK(work->wait(), "Tensor-parallel autograd SUM did not complete");
}

void requireFirstOrderTpBackward(const torch::autograd::variable_list& gradients) {
    // The autograd engine enables GradMode in backward only for create_graph.
    // Raw c10d collectives do not build a higher-order autograd graph.
    TORCH_CHECK(!torch::autograd::GradMode::is_enabled(),
                "Tensor-parallel autograd mappings support first-order gradients only");
    TORCH_CHECK(gradients.size() == 1 && gradients[0].defined(),
                "Tensor-parallel autograd mapping requires one defined output gradient");
}

class CopyToTensorParallel : public torch::autograd::Function<CopyToTensorParallel> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* context,
                                 const torch::Tensor& input) {
        std::lock_guard<std::mutex> lock(g_dist_mutex);
        requireTpGeneration(g_process_group_generation);
        context->saved_data["generation"] = g_process_group_generation;
        return input.clone(at::MemoryFormat::Contiguous);
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* context,
                                                  torch::autograd::variable_list gradients) {
        std::lock_guard<std::mutex> lock(g_dist_mutex);
        requireTpGeneration(context->saved_data.at("generation").toInt());
        requireFirstOrderTpBackward(gradients);
        // Never reduce the engine-owned grad_output: it may have other users.
        auto result = gradients[0].clone(at::MemoryFormat::Contiguous);
        sumTpTensor(result);
        return {result};
    }
};

class ReduceFromTensorParallel : public torch::autograd::Function<ReduceFromTensorParallel> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* context,
                                 const torch::Tensor& input) {
        std::lock_guard<std::mutex> lock(g_dist_mutex);
        requireTpGeneration(g_process_group_generation);
        context->saved_data["generation"] = g_process_group_generation;
        auto result = input.clone(at::MemoryFormat::Contiguous);
        sumTpTensor(result);
        return result;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* context,
                                                  torch::autograd::variable_list gradients) {
        std::lock_guard<std::mutex> lock(g_dist_mutex);
        // Identity backward still belongs to the original group lifetime.
        requireTpGeneration(context->saved_data.at("generation").toInt());
        requireFirstOrderTpBackward(gradients);
        return {gradients[0]};
    }
};

} // anonymous namespace

extern "C" {

/**
 * Set the current CUDA device for this process.
 *
 * This should be called with LOCAL_RANK before NCCL collectives to avoid
 * duplicate-device communicator initialization.
 */
lean_object* lean_torch_dist_set_cuda_device(uint64_t device, lean_object* /*w*/) {
    try {
#if TYR_HAS_NCCL
        if (!torch::cuda::is_available()) {
            return lean_io_result_mk_ok(lean_box(0));
        }
        c10::cuda::set_device(static_cast<c10::DeviceIndex>(device));
#else
        (void)device;
#endif
        return lean_io_result_mk_ok(lean_box(0));
    } catch (const std::exception& e) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string(("Failed to set CUDA device: " + std::string(e.what())).c_str())));
    }
}

/**
 * Initialize the distributed process group.
 *
 * @param backend_obj - String: "nccl", "gloo", or "mpi"
 * @param master_addr_obj - String: Master address (e.g., "localhost")
 * @param master_port - Port number for rendezvous
 * @param rank - This process's rank
 * @param world_size - Total number of processes
 * @param w - Lean world token
 */
lean_object* lean_torch_dist_init_process_group(
    b_lean_obj_arg backend_obj,
    b_lean_obj_arg master_addr_obj,
    uint64_t master_port,
    uint64_t rank,
    uint64_t world_size,
    lean_object* w
) {
    std::lock_guard<std::mutex> lock(g_dist_mutex);

    if (g_initialized) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("Process group already initialized")));
    }

    try {
        std::string backend(lean_string_cstr(backend_obj));
        std::string master_addr(lean_string_cstr(master_addr_obj));

        if (master_port > static_cast<uint64_t>(std::numeric_limits<uint16_t>::max())) {
            return lean_io_result_mk_error(lean_mk_io_user_error(
                lean_mk_string(("Invalid master_port: " + std::to_string(master_port) +
                    " is out of range [0, 65535]").c_str())));
        }

        TORCH_CHECK(world_size > 0 && world_size <= static_cast<uint64_t>(std::numeric_limits<int>::max()) &&
                    rank < world_size, "Invalid process-group rank/world size");
        TORCH_CHECK(g_process_group_generation < std::numeric_limits<int64_t>::max(),
                    "Process-group generation exhausted");
        TORCH_CHECK(backend == "nccl" || backend == "gloo", "Unsupported backend: ", backend);
        TORCH_CHECK(backend != "nccl" || TYR_HAS_NCCL,
                    "NCCL backend requested but ProcessGroupNCCL is unavailable in this build");
        TORCH_CHECK(backend != "gloo" || TYR_HAS_GLOO,
                    "Gloo backend requested but ProcessGroupGloo is unavailable in this build");
        const int next_rank = static_cast<int>(rank);
        const int next_world_size = static_cast<int>(world_size);

        // Build candidate ownership locally. Any failed initialization releases
        // its store/backend without exposing an incomplete global lifetime.
        c10::intrusive_ptr<c10d::Backend> next_group;
        // Create TCP store for rendezvous
        bool is_master = (rank == 0);
        auto store_opts = c10d::TCPStoreOptions();
        store_opts.port = static_cast<uint16_t>(master_port);
        store_opts.isServer = is_master;
        store_opts.numWorkers = static_cast<int>(world_size);
        store_opts.waitWorkers = true;

        auto next_store = c10::make_intrusive<c10d::TCPStore>(master_addr, store_opts);

        if (backend == "nccl") {
#if TYR_HAS_NCCL
            auto options = c10d::ProcessGroupNCCL::Options::create();
            options->is_high_priority_stream = true;
            next_group = c10::make_intrusive<c10d::ProcessGroupNCCL>(
                next_store, next_rank, next_world_size, options);
#else
            return lean_io_result_mk_error(lean_mk_io_user_error(
                lean_mk_string("NCCL backend requested but ProcessGroupNCCL is unavailable in this build")));
#endif
        } else if (backend == "gloo") {
#if TYR_HAS_GLOO
            auto options = c10d::ProcessGroupGloo::Options::create();
            next_group = c10d::ProcessGroupGloo::createProcessGroupGloo(
                next_store, next_rank, next_world_size, options);
#else
            return lean_io_result_mk_error(lean_mk_io_user_error(
                lean_mk_string("Gloo backend requested but ProcessGroupGloo is unavailable in this build")));
#endif
        } else {
            return lean_io_result_mk_error(lean_mk_io_user_error(
                lean_mk_string(("Unsupported backend: " + backend).c_str())));
        }

        g_process_group = std::move(next_group);
        g_store = std::move(next_store);
        g_rank = next_rank;
        g_world_size = next_world_size;
        ++g_process_group_generation;
        g_initialized = true;
        return lean_io_result_mk_ok(lean_box(0));

    } catch (const std::exception& e) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string(("Failed to init process group: " + std::string(e.what())).c_str())));
    }
}

/**
 * Get the rank of this process.
 */
lean_object* lean_torch_dist_get_rank(lean_object* w) {
    std::lock_guard<std::mutex> lock(g_dist_mutex);
    if (!g_initialized) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("Process group not initialized")));
    }
    return lean_io_result_mk_ok(lean_box_uint64(static_cast<uint64_t>(g_rank)));
}

/**
 * Get the world size (total number of processes).
 */
lean_object* lean_torch_dist_get_world_size(lean_object* w) {
    std::lock_guard<std::mutex> lock(g_dist_mutex);
    if (!g_initialized) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("Process group not initialized")));
    }
    return lean_io_result_mk_ok(lean_box_uint64(static_cast<uint64_t>(g_world_size)));
}

/**
 * Check if distributed is initialized.
 */
lean_object* lean_torch_dist_is_initialized(lean_object* w) {
    std::lock_guard<std::mutex> lock(g_dist_mutex);
    return lean_io_result_mk_ok(lean_box(g_initialized ? 1 : 0));
}

lean_object* lean_torch_dist_get_process_group_generation(lean_object* /*w*/) {
    std::lock_guard<std::mutex> lock(g_dist_mutex);
    if (!g_initialized) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("Process group not initialized")));
    }
    return lean_io_result_mk_ok(lean_box_uint64(static_cast<uint64_t>(g_process_group_generation)));
}

/**
 * Destroy the process group and clean up resources.
 */
lean_object* lean_torch_dist_destroy_process_group(lean_object* w) {
    std::lock_guard<std::mutex> lock(g_dist_mutex);

    if (!g_initialized) {
        return lean_io_result_mk_ok(lean_box(0)); // Already destroyed
    }

    try {
        // Wait for all pending work
        for (auto& [id, work] : g_pending_work) {
            if (work) {
                work->wait();
            }
        }
        g_pending_work.clear();

        // Reset process group
        g_process_group.reset();
        g_store.reset();
        g_initialized = false;
        g_rank = 0;
        g_world_size = 1;

        return lean_io_result_mk_ok(lean_box(0));
    } catch (const std::exception& e) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string(("Failed to destroy process group: " + std::string(e.what())).c_str())));
    }
}

/**
 * Barrier - synchronize all processes.
 */
lean_object* lean_torch_dist_barrier(lean_object* w) {
    std::lock_guard<std::mutex> lock(g_dist_mutex);

    if (!g_initialized) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("Process group not initialized")));
    }

    try {
        c10d::BarrierOptions opts;
        auto work = g_process_group->barrier(opts);
        work->wait();
        return lean_io_result_mk_ok(lean_box(0));
    } catch (const std::exception& e) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string(("Barrier failed: " + std::string(e.what())).c_str())));
    }
}

/**
 * All-reduce operation: sum tensors across all processes.
 *
 * @param tensor - Input/output tensor (in-place reduction)
 * @param op - Reduce operation (0=SUM, 1=AVG, 2=PRODUCT, 3=MIN, 4=MAX)
 * @param async_op - If true, returns work ID for async wait
 * @param w - Lean world token
 */
lean_object* lean_torch_dist_all_reduce(
    lean_obj_arg /*s*/,
    b_lean_obj_arg tensor,
    uint8_t op,
    uint8_t async_op,
    lean_object* w
) {
    std::lock_guard<std::mutex> lock(g_dist_mutex);

    if (!g_initialized) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("Process group not initialized")));
    }

    try {
        auto t = borrowTensor(tensor);
        maybeSetCurrentCudaDeviceFromTensor(t);
        std::vector<torch::Tensor> tensors = {t};

        c10d::AllreduceOptions opts;
        opts.reduceOp = c10d::ReduceOp(reduceOpFromInt(op));

        auto work = g_process_group->allreduce(tensors, opts);

        if (async_op) {
            uint64_t work_id = g_next_work_id++;
            g_pending_work[work_id] = work;
            return lean_io_result_mk_ok(lean_box_uint64(work_id));
        } else {
            work->wait();
            return lean_io_result_mk_ok(lean_box(0));
        }
    } catch (const std::exception& e) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string(("All-reduce failed: " + std::string(e.what())).c_str())));
    }
}

// Custom autograd owns its lock; these wrappers must not acquire it again.
lean_object* lean_torch_dist_copy_to_tensor_parallel(
    lean_obj_arg /*s*/, b_lean_obj_arg tensor, lean_object* /*w*/) {
    try {
        return lean_io_result_mk_ok(fromTorchTensor(CopyToTensorParallel::apply(borrowTensor(tensor))));
    } catch (const std::exception& error) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(error.what())));
    }
}

lean_object* lean_torch_dist_reduce_from_tensor_parallel(
    lean_obj_arg /*s*/, b_lean_obj_arg tensor, lean_object* /*w*/) {
    try {
        return lean_io_result_mk_ok(fromTorchTensor(ReduceFromTensorParallel::apply(borrowTensor(tensor))));
    } catch (const std::exception& error) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(error.what())));
    }
}

/**
 * Broadcast tensor from source rank to all other processes.
 *
 * @param tensor - Input/output tensor
 * @param src_rank - Source rank to broadcast from
 * @param w - Lean world token
 */
lean_object* lean_torch_dist_broadcast(
    lean_obj_arg /*s*/,
    b_lean_obj_arg tensor,
    uint64_t src_rank,
    lean_object* w
) {
    std::lock_guard<std::mutex> lock(g_dist_mutex);

    if (!g_initialized) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("Process group not initialized")));
    }

    try {
        auto t = borrowTensor(tensor);
        maybeSetCurrentCudaDeviceFromTensor(t);
        std::vector<torch::Tensor> tensors = {t};

        c10d::BroadcastOptions opts;
        opts.rootRank = static_cast<int>(src_rank);

        auto work = g_process_group->broadcast(tensors, opts);
        work->wait();

        return lean_io_result_mk_ok(lean_box(0));
    } catch (const std::exception& e) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string(("Broadcast failed: " + std::string(e.what())).c_str())));
    }
}

/**
 * Reduce-scatter: reduce and distribute results to all processes.
 * Each process gets a different portion of the reduced tensor.
 *
 * @param output - Output tensor (portion for this rank)
 * @param input - Input tensor to reduce
 * @param op - Reduce operation
 * @param async_op - If true, returns work ID
 * @param w - Lean world token
 */
lean_object* lean_torch_dist_reduce_scatter_tensor(
    lean_obj_arg /*s_out*/,
    lean_obj_arg /*s_in*/,
    b_lean_obj_arg output,
    b_lean_obj_arg input,
    uint8_t op,
    uint8_t async_op,
    lean_object* w
) {
    std::lock_guard<std::mutex> lock(g_dist_mutex);

    if (!g_initialized) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("Process group not initialized")));
    }

    try {
        auto out_tensor = borrowTensor(output);
        auto in_tensor = borrowTensor(input);
        maybeSetCurrentCudaDeviceFromTensor(out_tensor);

        c10d::ReduceScatterOptions opts;
        opts.reduceOp = c10d::ReduceOp(reduceOpFromInt(op));

        // reduce_scatter_tensor expects output and a list of inputs (one per rank)
        std::vector<torch::Tensor> input_tensors;
        // Split input tensor into world_size chunks
        auto chunks = in_tensor.chunk(g_world_size);
        for (auto& chunk : chunks) {
            input_tensors.push_back(chunk);
        }

        std::vector<torch::Tensor> output_tensors = {out_tensor};
        std::vector<std::vector<torch::Tensor>> input_tensor_lists = {input_tensors};
        auto work = g_process_group->reduce_scatter(
            output_tensors,
            input_tensor_lists,
            opts);

        if (async_op) {
            uint64_t work_id = g_next_work_id++;
            g_pending_work[work_id] = work;
            return lean_io_result_mk_ok(lean_box_uint64(work_id));
        } else {
            work->wait();
            return lean_io_result_mk_ok(lean_box(0));
        }
    } catch (const std::exception& e) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string(("Reduce-scatter failed: " + std::string(e.what())).c_str())));
    }
}

/**
 * Reduce-scatter from an explicit list of input tensors.
 *
 * This matches PyTorch DistMuon semantics where each rank provides a list
 * of per-owner gradients (one tensor per rank in the process group).
 *
 * @param output - Output tensor for this rank
 * @param inputs - Array of input tensors (must have length world_size)
 * @param op - Reduce operation
 * @param w - Lean world token
 */
lean_object* lean_torch_dist_reduce_scatter_list(
    lean_obj_arg /*s*/,
    b_lean_obj_arg output,
    b_lean_obj_arg inputs,
    uint8_t op,
    lean_object* w
) {
    std::lock_guard<std::mutex> lock(g_dist_mutex);

    if (!g_initialized) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("Process group not initialized")));
    }

    try {
        auto out_tensor = borrowTensor(output);
        maybeSetCurrentCudaDeviceFromTensor(out_tensor);

        size_t n = lean_array_size(inputs);
        if (n != static_cast<size_t>(g_world_size)) {
            return lean_io_result_mk_error(lean_mk_io_user_error(
                lean_mk_string("reduce_scatter_list requires inputs length == world size")));
        }

        std::vector<torch::Tensor> input_tensors;
        input_tensors.reserve(n);
        for (size_t i = 0; i < n; i++) {
            auto tensor_obj = lean_array_get_core(inputs, i);
            input_tensors.push_back(borrowTensor(tensor_obj));
        }

        c10d::ReduceScatterOptions opts;
        opts.reduceOp = c10d::ReduceOp(reduceOpFromInt(op));

        std::vector<torch::Tensor> output_tensors = {out_tensor};
        std::vector<std::vector<torch::Tensor>> input_tensor_lists = {input_tensors};
        auto work = g_process_group->reduce_scatter(
            output_tensors,
            input_tensor_lists,
            opts);
        work->wait();

        return lean_io_result_mk_ok(lean_box(0));
    } catch (const std::exception& e) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string(("Reduce-scatter(list) failed: " + std::string(e.what())).c_str())));
    }
}

/**
 * All-gather: gather tensors from all processes to all processes.
 *
 * @param output - Output tensor (concatenated from all ranks)
 * @param input - Input tensor from this rank
 * @param async_op - If true, returns work ID
 * @param w - Lean world token
 */
lean_object* lean_torch_dist_all_gather_into_tensor(
    lean_obj_arg /*s_out*/,
    lean_obj_arg /*s_in*/,
    b_lean_obj_arg output,
    b_lean_obj_arg input,
    uint8_t async_op,
    lean_object* w
) {
    std::lock_guard<std::mutex> lock(g_dist_mutex);

    if (!g_initialized) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("Process group not initialized")));
    }

    try {
        auto out_tensor = borrowTensor(output);
        auto in_tensor = borrowTensor(input);
        maybeSetCurrentCudaDeviceFromTensor(in_tensor);

        // Split output into world_size chunks to receive each rank's data
        auto output_chunks = out_tensor.chunk(g_world_size);
        std::vector<torch::Tensor> output_list;
        for (auto& chunk : output_chunks) {
            output_list.push_back(chunk);
        }

        std::vector<std::vector<torch::Tensor>> output_tensor_lists = {output_list};
        std::vector<torch::Tensor> input_tensors = {in_tensor};
        auto work = g_process_group->allgather(
            output_tensor_lists,
            input_tensors);

        if (async_op) {
            uint64_t work_id = g_next_work_id++;
            g_pending_work[work_id] = work;
            return lean_io_result_mk_ok(lean_box_uint64(work_id));
        } else {
            work->wait();
            return lean_io_result_mk_ok(lean_box(0));
        }
    } catch (const std::exception& e) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string(("All-gather failed: " + std::string(e.what())).c_str())));
    }
}

/**
 * All-gather into an explicit list of output tensors.
 *
 * This matches PyTorch DistMuon semantics where each rank gathers one updated
 * owner tensor into a list of per-owner destination tensors.
 *
 * @param outputs - Array of output tensors (must have length world_size)
 * @param input - Input tensor from this rank
 * @param w - Lean world token
 */
lean_object* lean_torch_dist_all_gather_list(
    lean_obj_arg /*s*/,
    b_lean_obj_arg outputs,
    b_lean_obj_arg input,
    lean_object* w
) {
    std::lock_guard<std::mutex> lock(g_dist_mutex);

    if (!g_initialized) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("Process group not initialized")));
    }

    try {
        auto in_tensor = borrowTensor(input);
        maybeSetCurrentCudaDeviceFromTensor(in_tensor);

        size_t n = lean_array_size(outputs);
        if (n != static_cast<size_t>(g_world_size)) {
            return lean_io_result_mk_error(lean_mk_io_user_error(
                lean_mk_string("all_gather_list requires outputs length == world size")));
        }

        std::vector<torch::Tensor> output_list;
        output_list.reserve(n);
        for (size_t i = 0; i < n; i++) {
            auto tensor_obj = lean_array_get_core(outputs, i);
            output_list.push_back(borrowTensor(tensor_obj));
        }

        std::vector<std::vector<torch::Tensor>> output_tensor_lists = {output_list};
        std::vector<torch::Tensor> input_tensors = {in_tensor};
        auto work = g_process_group->allgather(
            output_tensor_lists,
            input_tensors);
        work->wait();

        return lean_io_result_mk_ok(lean_box(0));
    } catch (const std::exception& e) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string(("All-gather(list) failed: " + std::string(e.what())).c_str())));
    }
}

/**
 * Wait for an async operation to complete.
 *
 * @param work_id - Work ID returned from async operation
 * @param w - Lean world token
 */
lean_object* lean_torch_dist_wait(uint64_t work_id, lean_object* w) {
    std::lock_guard<std::mutex> lock(g_dist_mutex);

    auto it = g_pending_work.find(work_id);
    if (it == g_pending_work.end()) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("Invalid work ID")));
    }

    try {
        it->second->wait();
        g_pending_work.erase(it);
        return lean_io_result_mk_ok(lean_box(0));
    } catch (const std::exception& e) {
        g_pending_work.erase(it);
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string(("Wait failed: " + std::string(e.what())).c_str())));
    }
}

/**
 * Check if an async operation is complete (non-blocking).
 *
 * @param work_id - Work ID
 * @param w - Lean world token
 * @return Bool - true if complete
 */
lean_object* lean_torch_dist_is_completed(uint64_t work_id, lean_object* w) {
    std::lock_guard<std::mutex> lock(g_dist_mutex);

    auto it = g_pending_work.find(work_id);
    if (it == g_pending_work.end()) {
        // Not found means already completed and cleaned up
        return lean_io_result_mk_ok(lean_box(1));
    }

    bool completed = it->second->isCompleted();
    if (completed) {
        g_pending_work.erase(it);
    }
    return lean_io_result_mk_ok(lean_box(completed ? 1 : 0));
}

} // extern "C"
