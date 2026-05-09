#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fe = cudnn_frontend;

namespace {

struct Conv3dFp8Context {
    int64_t n = 0;
    int64_t c = 0;
    int64_t d = 0;
    int64_t h = 0;
    int64_t w = 0;
    int64_t k = 0;
    int64_t t = 0;
    int64_t r = 0;
    int64_t s = 0;

    int64_t out_d = 0;
    int64_t out_h = 0;
    int64_t out_w = 0;
    int64_t device_index = 0;
    bool with_bias = false;

    std::shared_ptr<fe::graph::Graph> graph;
    std::shared_ptr<fe::graph::Tensor_attributes> x_attr;
    std::shared_ptr<fe::graph::Tensor_attributes> w_attr;
    std::shared_ptr<fe::graph::Tensor_attributes> y_attr;
    std::shared_ptr<fe::graph::Tensor_attributes> descale_x_attr;
    std::shared_ptr<fe::graph::Tensor_attributes> descale_w_attr;
    std::shared_ptr<fe::graph::Tensor_attributes> bias_attr;

    cudnnHandle_t handle = nullptr;
    int64_t plan_index = 0;
    int64_t workspace_bytes = 0;
    torch::Tensor workspace;

    ~Conv3dFp8Context() {
        if (handle != nullptr) {
            cudnnDestroy(handle);
            handle = nullptr;
        }
    }
};

std::atomic<int64_t> g_next_handle_id{1};
std::mutex g_ctx_mutex;
std::unordered_map<int64_t, std::shared_ptr<Conv3dFp8Context>> g_contexts;

void check_cudnn(cudnnStatus_t status, const char* where) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(where) + ": " + cudnnGetErrorString(status));
    }
}

void check_graph(fe::error_t result, const char* where) {
    if (!result.is_good()) {
        throw std::runtime_error(std::string(where) + ": " + result.get_message());
    }
}

void check_cuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
    }
}

void validate_3d_conv_tuple(const std::vector<int64_t>& v, const char* name) {
    if (v.size() != 3) {
        throw std::invalid_argument(std::string(name) + " must contain exactly 3 elements.");
    }
}

void validate_5d_shape(const std::vector<int64_t>& shape, const char* name) {
    if (shape.size() != 5) {
        throw std::invalid_argument(std::string(name) + " must contain exactly 5 elements.");
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            throw std::invalid_argument(std::string(name) + " elements must be > 0.");
        }
    }
}

int64_t conv_out_size(int64_t in, int64_t pad, int64_t stride, int64_t dilation, int64_t kernel) {
    return (in + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

std::shared_ptr<Conv3dFp8Context> get_context_or_throw(int64_t handle_id) {
    std::lock_guard<std::mutex> lock(g_ctx_mutex);
    auto it = g_contexts.find(handle_id);
    if (it == g_contexts.end()) {
        throw std::invalid_argument("Invalid conv3d_fp8 handle_id.");
    }
    return it->second;
}

std::pair<int64_t, int64_t> autotune_best_plan(const std::shared_ptr<Conv3dFp8Context>& ctx) {
    constexpr int kWarmupIterations = 3;
    constexpr int kProfileIterations = 20;

    auto y_options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, ctx->device_index);
    auto scale_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA, ctx->device_index);
    auto ws_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, ctx->device_index);

    auto x = torch::randn({ctx->n, ctx->c, ctx->d, ctx->h, ctx->w}, y_options).to(torch::kFloat8_e4m3fn);
    x = x.to(torch::MemoryFormat::ChannelsLast3d);
    auto w = torch::randn({ctx->k, ctx->c, ctx->t, ctx->r, ctx->s}, y_options).to(torch::kFloat8_e4m3fn);
    w = w.to(torch::MemoryFormat::ChannelsLast3d);
    auto y = torch::empty({ctx->n, ctx->k, ctx->out_d, ctx->out_h, ctx->out_w}, y_options.memory_format(torch::MemoryFormat::ChannelsLast3d));

    auto descale_x = torch::ones({1, 1, 1, 1, 1}, scale_options);
    auto descale_w = torch::ones({1, 1, 1, 1, 1}, scale_options);
    auto bias = torch::zeros({1, ctx->k, 1, 1, 1}, y_options);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {ctx->x_attr, x.data_ptr()},
        {ctx->w_attr, w.data_ptr()},
        {ctx->y_attr, y.data_ptr()},
        {ctx->descale_x_attr, descale_x.data_ptr()},
        {ctx->descale_w_attr, descale_w.data_ptr()},
    };
    if (ctx->with_bias) {
        variant_pack.emplace(ctx->bias_attr, bias.data_ptr());
    }

    const auto stream = at::cuda::getCurrentCUDAStream(static_cast<int>(ctx->device_index)).stream();
    check_cudnn(cudnnSetStream(ctx->handle, stream), "cudnnSetStream(autotune)");

    const int64_t plan_count = static_cast<int64_t>(ctx->graph->get_execution_plan_count());
    int64_t best_plan_index = -1;
    int64_t best_workspace_bytes = 0;
    float best_elapsed_ms = std::numeric_limits<float>::max();

    for (int64_t plan_index = 0; plan_index < plan_count; ++plan_index) {
        const int64_t workspace_bytes = static_cast<int64_t>(ctx->graph->get_workspace_size_plan_at_index(plan_index));
        torch::Tensor workspace;
        void* workspace_ptr = nullptr;
        if (workspace_bytes > 0) {
            workspace = torch::empty({workspace_bytes}, ws_options);
            workspace_ptr = workspace.data_ptr();
        }

        try {
            for (int i = 0; i < kWarmupIterations; ++i) {
                check_graph(
                    ctx->graph->execute_plan_at_index(ctx->handle, variant_pack, workspace_ptr, plan_index),
                    "autotune/warmup execute_plan_at_index");
            }

            cudaEvent_t start_event = nullptr;
            cudaEvent_t stop_event = nullptr;
            check_cuda(cudaEventCreate(&start_event), "cudaEventCreate(start)");
            check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate(stop)");

            check_cuda(cudaEventRecord(start_event, stream), "cudaEventRecord(start)");
            for (int i = 0; i < kProfileIterations; ++i) {
                check_graph(
                    ctx->graph->execute_plan_at_index(ctx->handle, variant_pack, workspace_ptr, plan_index),
                    "autotune/profile execute_plan_at_index");
            }
            check_cuda(cudaEventRecord(stop_event, stream), "cudaEventRecord(stop)");
            check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)");

            float elapsed_ms = 0.0f;
            check_cuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime");
            check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy(start)");
            check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy(stop)");

            if (elapsed_ms < best_elapsed_ms) {
                best_elapsed_ms = elapsed_ms;
                best_plan_index = plan_index;
                best_workspace_bytes = workspace_bytes;
            }
        } catch (...) {
            // Skip failing plans during autotuning and keep trying remaining candidates.
            continue;
        }
    }

    if (best_plan_index < 0) {
        throw std::runtime_error("conv3d_fp8 autotune failed to find a runnable execution plan.");
    }

    return {best_plan_index, best_workspace_bytes};
}

int64_t conv3d_fp8_init(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> w_shape,
    int64_t device_index,
    std::vector<int64_t> padding,
    std::vector<int64_t> stride,
    std::vector<int64_t> dilation,
    bool with_bias) {
    int device_count = 0;
    cudaError_t device_count_error = cudaGetDeviceCount(&device_count);
    if (device_count_error != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(device_count_error));
    }
    if (device_index < 0 || device_index >= device_count) {
        throw std::invalid_argument("device_index is out of CUDA device range.");
    }

    validate_5d_shape(x_shape, "x_shape");
    validate_5d_shape(w_shape, "w_shape");
    validate_3d_conv_tuple(padding, "padding");
    validate_3d_conv_tuple(stride, "stride");
    validate_3d_conv_tuple(dilation, "dilation");

    auto ctx = std::make_shared<Conv3dFp8Context>();
    ctx->n = x_shape[0];
    ctx->c = x_shape[1];
    ctx->d = x_shape[2];
    ctx->h = x_shape[3];
    ctx->w = x_shape[4];
    ctx->device_index = device_index;
    ctx->with_bias = with_bias;

    ctx->k = w_shape[0];
    if (w_shape[1] != ctx->c) {
        throw std::invalid_argument("w_shape[1] must match x_shape[1].");
    }
    ctx->t = w_shape[2];
    ctx->r = w_shape[3];
    ctx->s = w_shape[4];

    ctx->out_d = conv_out_size(ctx->d, padding[0], stride[0], dilation[0], ctx->t);
    ctx->out_h = conv_out_size(ctx->h, padding[1], stride[1], dilation[1], ctx->r);
    ctx->out_w = conv_out_size(ctx->w, padding[2], stride[2], dilation[2], ctx->s);
    if (ctx->out_d <= 0 || ctx->out_h <= 0 || ctx->out_w <= 0) {
        throw std::invalid_argument("Invalid output shape for the given convolution parameters.");
    }

    c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(ctx->device_index));

    check_cudnn(cudnnCreate(&ctx->handle), "cudnnCreate");
    const auto current_stream = at::cuda::getCurrentCUDAStream(static_cast<int>(ctx->device_index)).stream();
    check_cudnn(cudnnSetStream(ctx->handle, current_stream), "cudnnSetStream(init)");

    ctx->graph = std::make_shared<fe::graph::Graph>();
    ctx->graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    ctx->x_attr = ctx->graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("image")
            .set_dim({ctx->n, ctx->c, ctx->d, ctx->h, ctx->w})
            .set_stride({ctx->c * ctx->d * ctx->h * ctx->w, 1, ctx->c * ctx->h * ctx->w, ctx->c * ctx->w, ctx->c})
            .set_data_type(fe::DataType_t::FP8_E4M3));

    ctx->w_attr = ctx->graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("filter")
            .set_dim({ctx->k, ctx->c, ctx->t, ctx->r, ctx->s})
            .set_stride({ctx->c * ctx->t * ctx->r * ctx->s, 1, ctx->c * ctx->r * ctx->s, ctx->c * ctx->s, ctx->c})
            .set_data_type(fe::DataType_t::FP8_E4M3));

    auto conv_options = fe::graph::Conv_fprop_attributes()
                            .set_padding(padding)
                            .set_stride(stride)
                            .set_dilation(dilation)
                            .set_name("conv3d_fp8")
                            .set_compute_data_type(fe::DataType_t::FAST_FLOAT_FOR_FP8);

    auto conv_output_fp8 = ctx->graph->conv_fprop(ctx->x_attr, ctx->w_attr, conv_options);

    ctx->descale_x_attr = ctx->graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("descale_x")
            .set_dim({1, 1, 1, 1, 1})
            .set_stride({1, 1, 1, 1, 1})
            .set_data_type(fe::DataType_t::FLOAT));

    ctx->descale_w_attr = ctx->graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("descale_w")
            .set_dim({1, 1, 1, 1, 1})
            .set_stride({1, 1, 1, 1, 1})
            .set_data_type(fe::DataType_t::FLOAT));

    auto scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
    auto after_descale_x = ctx->graph->pointwise(conv_output_fp8, ctx->descale_x_attr, scale_options);
    auto after_descale_w = ctx->graph->pointwise(after_descale_x, ctx->descale_w_attr, scale_options);

    if (ctx->with_bias) {
        ctx->bias_attr = ctx->graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("bias")
                .set_dim({1, ctx->k, 1, 1, 1})
                .set_stride({ctx->k, 1, 1, 1, 1})
                .set_data_type(fe::DataType_t::BFLOAT16));
        auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
        ctx->y_attr = ctx->graph->pointwise(after_descale_w, ctx->bias_attr, bias_options);
    } else {
        ctx->y_attr = after_descale_w;
    }
    ctx->y_attr->set_output(true).set_data_type(fe::DataType_t::BFLOAT16);

    // ctx->amax_attr = ctx->graph->reduction(
    //     after_descale_w,
    //     fe::graph::Reduction_attributes()
    //         .set_mode(fe::ReductionMode_t::AMAX)
    //         .set_compute_data_type(fe::DataType_t::FLOAT));
    // ctx->amax_attr->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_dim({1, 1, 1, 1, 1});

    check_graph(ctx->graph->validate(), "graph->validate");
    check_graph(ctx->graph->build_operation_graph(ctx->handle), "graph->build_operation_graph");
    check_graph(ctx->graph->create_execution_plans({fe::HeurMode_t::A}), "graph->create_execution_plans");
    check_graph(ctx->graph->check_support(ctx->handle), "graph->check_support");
    check_graph(ctx->graph->build_plans(ctx->handle, fe::BuildPlanPolicy_t::ALL), "graph->build_plans");

    if (ctx->graph->get_execution_plan_count() <= 0) {
        throw std::runtime_error("No execution plan is available for conv3d_fp8.");
    }
    std::tie(ctx->plan_index, ctx->workspace_bytes) = autotune_best_plan(ctx);
    if (ctx->workspace_bytes > 0) {
        auto ws_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, ctx->device_index);
        ctx->workspace = torch::empty({ctx->workspace_bytes}, ws_options);
    }

    const int64_t handle_id = g_next_handle_id.fetch_add(1);
    {
        std::lock_guard<std::mutex> lock(g_ctx_mutex);
        g_contexts.emplace(handle_id, std::move(ctx));
    }
    return handle_id;
}

torch::Tensor conv3d_fp8_forward(
    int64_t handle_id,
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& descale_x,
    const torch::Tensor& descale_w,
    const c10::optional<torch::Tensor>& bias) {
    auto ctx = get_context_or_throw(handle_id);
    if (!x.is_cuda() || !w.is_cuda() || !descale_x.is_cuda() || !descale_w.is_cuda()) {
        throw std::invalid_argument("All tensors passed to conv3d_fp8.forward must be CUDA tensors.");
    }
    if (x.get_device() != ctx->device_index || w.get_device() != ctx->device_index || descale_x.get_device() != ctx->device_index ||
        descale_w.get_device() != ctx->device_index) {
        throw std::invalid_argument("All tensors passed to conv3d_fp8.forward must be on the init device.");
    }
    if (x.dim() != 5 || w.dim() != 5) {
        throw std::invalid_argument("x and w must be 5D tensors.");
    }
    if (x.size(0) != ctx->n || x.size(1) != ctx->c || x.size(2) != ctx->d || x.size(3) != ctx->h || x.size(4) != ctx->w) {
        throw std::invalid_argument("x shape does not match the shape used during init.");
    }
    if (w.size(0) != ctx->k || w.size(1) != ctx->c || w.size(2) != ctx->t || w.size(3) != ctx->r || w.size(4) != ctx->s) {
        throw std::invalid_argument("w shape does not match the shape used during init.");
    }
    c10::optional<torch::Tensor> bias_contiguous;
    if (ctx->with_bias) {
        if (!bias.has_value()) {
            throw std::invalid_argument("bias must be provided when op is initialized with with_bias=True.");
        }
        if (!bias->is_cuda()) {
            throw std::invalid_argument("bias must be a CUDA tensor.");
        }
        if (bias->get_device() != ctx->device_index) {
            throw std::invalid_argument("bias must be on the init device.");
        }
        if (bias->dim() != 1 || bias->size(0) != ctx->k) {
            throw std::invalid_argument("bias shape must be [out_channels].");
        }
        if (bias->scalar_type() != torch::kBFloat16) {
            throw std::invalid_argument("bias dtype must be torch.bfloat16.");
        }
        bias_contiguous = bias->contiguous();
    } else if (bias.has_value()) {
        throw std::invalid_argument("bias must be None when op is initialized with with_bias=False.");
    }

    auto y = torch::empty(
        {ctx->n, ctx->k, ctx->out_d, ctx->out_h, ctx->out_w},
        x.options().dtype(torch::kBFloat16).memory_format(torch::MemoryFormat::ChannelsLast3d));

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {ctx->x_attr, x.data_ptr()},
        {ctx->w_attr, w.data_ptr()},
        {ctx->y_attr, y.data_ptr()},
        {ctx->descale_x_attr, descale_x.data_ptr()},
        {ctx->descale_w_attr, descale_w.data_ptr()},
    };
    if (ctx->with_bias) {
        variant_pack.emplace(ctx->bias_attr, bias_contiguous->data_ptr());
    }

    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
    check_cudnn(cudnnSetStream(ctx->handle, stream), "cudnnSetStream(forward)");

    void* workspace_ptr = nullptr;
    if (ctx->workspace.defined() && ctx->workspace_bytes > 0) {
        workspace_ptr = ctx->workspace.data_ptr();
    }

    check_graph(
        ctx->graph->execute_plan_at_index(ctx->handle, variant_pack, workspace_ptr, ctx->plan_index),
        "graph->execute_plan_at_index");

    return y;
}

void conv3d_fp8_destroy(int64_t handle_id) {
    std::lock_guard<std::mutex> lock(g_ctx_mutex);
    auto it = g_contexts.find(handle_id);
    if (it != g_contexts.end()) {
        g_contexts.erase(it);
    }
}

}  // namespace

TORCH_LIBRARY(conv3d_fp8, m) {
    m.def("init(int[] x_shape, int[] w_shape, int device_index, int[] padding, int[] stride, int[] dilation, bool with_bias=False) -> int");
    m.def("forward(int handle_id, Tensor x, Tensor w, Tensor descale_x, Tensor descale_w, Tensor? bias=None) -> Tensor");
    m.def("destroy(int handle_id) -> ()");
}

TORCH_LIBRARY_IMPL(conv3d_fp8, CUDA, m) {
    m.impl("init", &conv3d_fp8_init);
    m.impl("forward", &conv3d_fp8_forward);
    m.impl("destroy", &conv3d_fp8_destroy);
}

PYBIND11_MODULE(conv3d_fp8_ext, m) {
    m.doc() = "Prebuilt conv3d_fp8 Torch extension module.";
    m.def(
        "init",
        &conv3d_fp8_init,
        pybind11::arg("x_shape"),
        pybind11::arg("w_shape"),
        pybind11::arg("device_index"),
        pybind11::arg("padding"),
        pybind11::arg("stride"),
        pybind11::arg("dilation"),
        pybind11::arg("with_bias") = false);
}
