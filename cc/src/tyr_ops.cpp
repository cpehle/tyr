#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <lean/lean.h>
#include <torch/torch.h>
#include <ATen/ATen.h>

#if defined(__has_include)
#if __has_include(<c10/cuda/CUDAStream.h>) && __has_include(<c10/cuda/CUDAFunctions.h>)
#define TYR_OPS_HAS_CUDA_STREAM 1
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>
#else
#define TYR_OPS_HAS_CUDA_STREAM 0
#endif
#else
#define TYR_OPS_HAS_CUDA_STREAM 0
#endif

// Shared tensor/Lean interop helpers defined in tyr.cpp.
torch::Tensor borrowTensor(b_lean_obj_arg o);
lean_object *giveTensor(torch::Tensor t);
lean_object *fromTorchTensor(torch::Tensor t);

extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100Fwd2Block(
    b_lean_obj_arg q_ptr, b_lean_obj_arg k_ptr, b_lean_obj_arg v_ptr,
    b_lean_obj_arg o_ptr, b_lean_obj_arg l_ptr, uint64_t seq_len, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100Fwd12Block(
    b_lean_obj_arg q_ptr, b_lean_obj_arg k_ptr, b_lean_obj_arg v_ptr,
    b_lean_obj_arg o_ptr, b_lean_obj_arg l_ptr, uint64_t seq_len, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100BwdPrep2Block(
    b_lean_obj_arg dO_ptr, b_lean_obj_arg o_ptr, b_lean_obj_arg d_ptr,
    uint64_t seq_len, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100Bwd2BlockPartials(
    b_lean_obj_arg q_ptr, b_lean_obj_arg k_ptr, b_lean_obj_arg v_ptr,
    b_lean_obj_arg dO_ptr, b_lean_obj_arg l_ptr, b_lean_obj_arg d_ptr,
    b_lean_obj_arg dQ_ptr, b_lean_obj_arg dK_part_ptr, b_lean_obj_arg dV_part_ptr,
    uint64_t seq_len, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100Bwd12BlockPartials(
    b_lean_obj_arg q_ptr, b_lean_obj_arg k_ptr, b_lean_obj_arg v_ptr,
    b_lean_obj_arg dO_ptr, b_lean_obj_arg l_ptr, b_lean_obj_arg d_ptr,
    b_lean_obj_arg dQ_ptr, b_lean_obj_arg dK_part_ptr, b_lean_obj_arg dV_part_ptr,
    uint64_t seq_len, uint64_t head_dim,
    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
    uint64_t block_x, uint64_t block_y, uint64_t block_z,
    uint64_t shared_mem, uint64_t stream);

namespace tyr_ops {

enum class FlashAttnRoute {
  Portable,
  TkMhaH1002Block,
  TkMhaH10012Block,
};

struct LaunchConfig {
  uint64_t grid_x{1};
  uint64_t grid_y{1};
  uint64_t grid_z{1};
  uint64_t block_x{128};
  uint64_t block_y{1};
  uint64_t block_z{1};
  uint64_t shared_mem{0};
  uint64_t stream{0};
};

struct LeanTensorRef {
  lean_object* obj;
  explicit LeanTensorRef(const torch::Tensor& t) : obj(giveTensor(t)) {}
  ~LeanTensorRef() {
    if (obj != nullptr) {
      lean_dec(obj);
    }
  }
  LeanTensorRef(const LeanTensorRef&) = delete;
  LeanTensorRef& operator=(const LeanTensorRef&) = delete;
};

static inline double default_scale_for_dim(int64_t head_dim) {
  return 1.0 / std::sqrt(static_cast<double>(head_dim));
}

static inline torch::Tensor scale_query_if_needed(
    const torch::Tensor& query,
    const c10::optional<double>& scale) {
  if (!scale.has_value()) {
    return query;
  }
  const double default_scale = default_scale_for_dim(query.size(-1));
  if (std::abs(scale.value() - default_scale) <= 1.0e-6) {
    return query;
  }
  const double factor = scale.value() / default_scale;
  return query * factor;
}

static inline bool scale_matches_default(
    int64_t head_dim,
    const c10::optional<double>& scale) {
  if (!scale.has_value()) {
    return true;
  }
  return std::abs(scale.value() - default_scale_for_dim(head_dim)) <= 1.0e-6;
}

static void check_flash_attn_args(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const c10::optional<torch::Tensor>& attn_mask) {
  TORCH_CHECK(query.dim() == 4, "tyr::flash_attn: query must be rank-4 [B, Hq, Q, D]");
  TORCH_CHECK(key.dim() == 4, "tyr::flash_attn: key must be rank-4 [B, Hkv, K, D]");
  TORCH_CHECK(value.dim() == 4, "tyr::flash_attn: value must be rank-4 [B, Hkv, K, D]");
  TORCH_CHECK(query.device() == key.device() && key.device() == value.device(),
    "tyr::flash_attn: Q/K/V must be on the same device");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() && key.scalar_type() == value.scalar_type(),
    "tyr::flash_attn: Q/K/V must have the same dtype");
  TORCH_CHECK(query.size(0) == key.size(0) && key.size(0) == value.size(0),
    "tyr::flash_attn: batch mismatch between Q/K/V");
  TORCH_CHECK(key.size(1) == value.size(1),
    "tyr::flash_attn: KV-head mismatch between K and V");
  TORCH_CHECK(key.size(2) == value.size(2),
    "tyr::flash_attn: KV-sequence mismatch between K and V");
  TORCH_CHECK(query.size(3) == key.size(3) && key.size(3) == value.size(3),
    "tyr::flash_attn: head-dimension mismatch between Q/K/V");
  if (attn_mask.has_value() && attn_mask->defined()) {
    TORCH_CHECK(attn_mask->dim() == 2,
      "tyr::flash_attn: attn_mask must be rank-2 [batch, kv_seq]");
    TORCH_CHECK(attn_mask->size(0) == query.size(0) && attn_mask->size(1) == key.size(2),
      "tyr::flash_attn: attn_mask shape must be [batch, kv_seq]");
  }
}

static inline FlashAttnRoute select_route(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const c10::optional<torch::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const c10::optional<double>& scale,
    bool enable_gqa) {
  const bool mask_ok = !(attn_mask.has_value() && attn_mask->defined());
  const bool device_ok = query.is_cuda();
  const bool dtype_ok = query.scalar_type() == torch::kBFloat16;
  const bool shape_ok =
      query.size(0) == 1 &&
      query.size(1) == 1 &&
      key.size(1) == 1 &&
      value.size(1) == 1 &&
      query.size(2) == key.size(2) &&
      query.size(2) == value.size(2) &&
      query.size(3) == 64;
  const bool semantics_ok =
      mask_ok &&
      dropout_p == 0.0 &&
      !is_causal &&
      !enable_gqa &&
      scale_matches_default(query.size(3), scale);
  if (!(device_ok && dtype_ok && shape_ok && semantics_ok)) {
    return FlashAttnRoute::Portable;
  }
  if (query.size(2) == 128) {
    return FlashAttnRoute::TkMhaH1002Block;
  }
  if (query.size(2) == 768) {
    return FlashAttnRoute::TkMhaH10012Block;
  }
  return FlashAttnRoute::Portable;
}

static inline uint64_t current_stream_handle(const torch::Tensor& t) {
#if TYR_OPS_HAS_CUDA_STREAM
  if (t.is_cuda()) {
    return reinterpret_cast<uint64_t>(c10::cuda::getCurrentCUDAStream(t.get_device()).stream());
  }
#endif
  return 0;
}

static void throw_on_launcher_error(lean_object* io_result, const char* launcher) {
  if (lean_io_result_is_error(io_result)) {
    std::cerr << "tyr::flash_attn: native launcher failed: " << launcher << std::endl;
    lean_io_result_show_error(io_result);
    lean_dec(io_result);
    throw std::runtime_error(std::string("tyr::flash_attn: native launcher failed: ") + launcher);
  }
  lean_dec(io_result);
}

static LaunchConfig launch_config_for(const torch::Tensor& query, FlashAttnRoute route) {
  LaunchConfig cfg;
  cfg.grid_y = (route == FlashAttnRoute::TkMhaH10012Block) ? 12 : 2;
  cfg.stream = current_stream_handle(query);
  return cfg;
}

static int64_t kv_blocks_for(FlashAttnRoute route) {
  return route == FlashAttnRoute::TkMhaH10012Block ? 12 : 2;
}

static std::pair<torch::Tensor, torch::Tensor> native_forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    FlashAttnRoute route) {
  const auto q = query.contiguous();
  const auto k = key.contiguous();
  const auto v = value.contiguous();
  auto out = torch::zeros_like(q);
  auto l = torch::zeros(
      {kv_blocks_for(route), 64},
      q.options().dtype(torch::kFloat32));

  LeanTensorRef q_ref(q);
  LeanTensorRef k_ref(k);
  LeanTensorRef v_ref(v);
  LeanTensorRef out_ref(out);
  LeanTensorRef l_ref(l);

  const auto cfg = launch_config_for(q, route);
  lean_object* result = nullptr;
  if (route == FlashAttnRoute::TkMhaH1002Block) {
    result = lean_launch_Tyr_GPU_Kernels_tkMhaH100Fwd2Block(
        q_ref.obj, k_ref.obj, v_ref.obj, out_ref.obj, l_ref.obj,
        static_cast<uint64_t>(q.size(2)), static_cast<uint64_t>(q.size(3)),
        cfg.grid_x, cfg.grid_y, cfg.grid_z,
        cfg.block_x, cfg.block_y, cfg.block_z,
        cfg.shared_mem, cfg.stream);
    throw_on_launcher_error(result, "tkMhaH100Fwd2Block");
  } else {
    result = lean_launch_Tyr_GPU_Kernels_tkMhaH100Fwd12Block(
        q_ref.obj, k_ref.obj, v_ref.obj, out_ref.obj, l_ref.obj,
        static_cast<uint64_t>(q.size(2)), static_cast<uint64_t>(q.size(3)),
        cfg.grid_x, cfg.grid_y, cfg.grid_z,
        cfg.block_x, cfg.block_y, cfg.block_z,
        cfg.shared_mem, cfg.stream);
    throw_on_launcher_error(result, "tkMhaH100Fwd12Block");
  }

  return {out, l};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> native_backward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& out,
    const torch::Tensor& l,
    const torch::Tensor& grad_out,
    FlashAttnRoute route) {
  const auto q = query.contiguous();
  const auto k = key.contiguous();
  const auto v = value.contiguous();
  const auto o = out.contiguous();
  const auto l_saved = l.contiguous();
  const auto dO = grad_out.contiguous();

  const int64_t seq = q.size(2);
  const int64_t kv_blocks = kv_blocks_for(route);
  const auto f32_opts = q.options().dtype(torch::kFloat32);

  auto dVec = torch::zeros({kv_blocks, 64}, f32_opts);
  auto dQ = torch::zeros({1, 1, seq, 64}, f32_opts);
  auto dKStack = torch::zeros({1, 1, kv_blocks * seq, 64}, f32_opts);
  auto dVStack = torch::zeros({1, 1, kv_blocks * seq, 64}, f32_opts);

  LeanTensorRef dO_ref(dO);
  LeanTensorRef o_ref(o);
  LeanTensorRef d_ref(dVec);

  const auto cfg = launch_config_for(q, route);
  auto prep_result = lean_launch_Tyr_GPU_Kernels_tkMhaH100BwdPrep2Block(
      dO_ref.obj, o_ref.obj, d_ref.obj,
      static_cast<uint64_t>(seq), static_cast<uint64_t>(q.size(3)),
      cfg.grid_x, cfg.grid_y, cfg.grid_z,
      cfg.block_x, cfg.block_y, cfg.block_z,
      cfg.shared_mem, cfg.stream);
  throw_on_launcher_error(prep_result, "tkMhaH100BwdPrep2Block");

  LeanTensorRef q_ref(q);
  LeanTensorRef k_ref(k);
  LeanTensorRef v_ref(v);
  LeanTensorRef l_ref(l_saved);
  LeanTensorRef dQ_ref(dQ);
  LeanTensorRef dK_ref(dKStack);
  LeanTensorRef dV_ref(dVStack);

  lean_object* bwd_result = nullptr;
  if (route == FlashAttnRoute::TkMhaH1002Block) {
    bwd_result = lean_launch_Tyr_GPU_Kernels_tkMhaH100Bwd2BlockPartials(
        q_ref.obj, k_ref.obj, v_ref.obj, dO_ref.obj, l_ref.obj, d_ref.obj,
        dQ_ref.obj, dK_ref.obj, dV_ref.obj,
        static_cast<uint64_t>(seq), static_cast<uint64_t>(q.size(3)),
        cfg.grid_x, cfg.grid_y, cfg.grid_z,
        cfg.block_x, cfg.block_y, cfg.block_z,
        cfg.shared_mem, cfg.stream);
    throw_on_launcher_error(bwd_result, "tkMhaH100Bwd2BlockPartials");
  } else {
    bwd_result = lean_launch_Tyr_GPU_Kernels_tkMhaH100Bwd12BlockPartials(
        q_ref.obj, k_ref.obj, v_ref.obj, dO_ref.obj, l_ref.obj, d_ref.obj,
        dQ_ref.obj, dK_ref.obj, dV_ref.obj,
        static_cast<uint64_t>(seq), static_cast<uint64_t>(q.size(3)),
        cfg.grid_x, cfg.grid_y, cfg.grid_z,
        cfg.block_x, cfg.block_y, cfg.block_z,
        cfg.shared_mem, cfg.stream);
    throw_on_launcher_error(bwd_result, "tkMhaH100Bwd12BlockPartials");
  }

  auto dK = dKStack.view({kv_blocks, seq, 64}).sum(0, false).unsqueeze(0).unsqueeze(0);
  const double scale = 1.0 / std::sqrt(static_cast<double>(q.size(3)));
  auto qf = q.to(torch::kFloat32);
  auto kf = k.to(torch::kFloat32);
  auto dOf = dO.to(torch::kFloat32);
  auto scores = torch::matmul(qf, kf.transpose(-2, -1)) * scale;
  auto probs = torch::softmax(scores, -1);
  auto dV = torch::matmul(probs.transpose(-2, -1), dOf);
  return {dQ, dK, dV};
}

static torch::Tensor expand_kv_heads_for_gqa(
    const torch::Tensor& tensor,
    int64_t q_heads,
    int64_t kv_heads,
    bool enable_gqa,
    const char* which) {
  if (!enable_gqa || q_heads == kv_heads) {
    return tensor;
  }
  TORCH_CHECK(kv_heads > 0, "tyr::flash_attn: ", which, " has zero KV heads");
  TORCH_CHECK(q_heads % kv_heads == 0,
    "tyr::flash_attn: enable_gqa=true requires q_heads to be divisible by kv_heads");
  auto repeat_factor = q_heads / kv_heads;
  return tensor.repeat_interleave(repeat_factor, 1);
}

static torch::Tensor portable_flash_attn(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const c10::optional<torch::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const c10::optional<double>& scale,
    bool enable_gqa) {
  auto q = scale_query_if_needed(query, scale);
  auto k = expand_kv_heads_for_gqa(key, q.size(1), key.size(1), enable_gqa, "key");
  auto v = expand_kv_heads_for_gqa(value, q.size(1), value.size(1), enable_gqa, "value");

  if (!attn_mask.has_value() || !attn_mask->defined()) {
    return torch::scaled_dot_product_attention(
        q, k, v, c10::nullopt, dropout_p, is_causal);
  }

  auto padding_mask = attn_mask->to(q.device());
  auto kv_seq = k.size(2);
  auto q_seq = q.size(2);
  TORCH_CHECK(padding_mask.size(1) == kv_seq,
    "tyr::flash_attn: attn_mask kv dimension must match K/V sequence length");

  auto key_mask = padding_mask.unsqueeze(1).unsqueeze(2);
  torch::Tensor expanded_mask;
  if (is_causal) {
    auto row_idx = torch::arange(q_seq, torch::TensorOptions().dtype(torch::kLong).device(q.device())).unsqueeze(1);
    auto col_idx = torch::arange(kv_seq, torch::TensorOptions().dtype(torch::kLong).device(q.device())).unsqueeze(0);
    auto causal_mask = col_idx > row_idx;
    auto combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) | (key_mask == 0);
    expanded_mask = torch::where(
        combined_mask,
        torch::full(combined_mask.sizes(), -std::numeric_limits<float>::infinity(), q.options()),
        torch::zeros(combined_mask.sizes(), q.options()));
  } else {
    expanded_mask = torch::where(
        key_mask == 0,
        torch::full(key_mask.sizes(), -std::numeric_limits<float>::infinity(), q.options()),
        torch::zeros(key_mask.sizes(), q.options()));
  }

  return torch::scaled_dot_product_attention(
      q, k, v, expanded_mask, dropout_p, false);
}

template <FlashAttnRoute Route>
class NativeFlashAttnFunction : public torch::autograd::Function<NativeFlashAttnFunction<Route>> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      torch::Tensor query,
      torch::Tensor key,
      torch::Tensor value) {
    auto [out, l] = native_forward(query, key, value, Route);
    ctx->save_for_backward({query, key, value, out, l});
    return out;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    if (grad_outputs.empty() || !grad_outputs[0].defined()) {
      return {torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
    auto [dQ, dK, dV] = native_backward(
        saved[0], saved[1], saved[2], saved[3], saved[4], grad_outputs[0], Route);
    return {dQ, dK, dV};
  }
};

torch::Tensor flash_attn_dispatch(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const c10::optional<torch::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const c10::optional<double>& scale,
    bool enable_gqa) {
  check_flash_attn_args(query, key, value, attn_mask);

  auto route = select_route(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
  switch (route) {
    case FlashAttnRoute::TkMhaH1002Block:
      return NativeFlashAttnFunction<FlashAttnRoute::TkMhaH1002Block>::apply(query, key, value);
    case FlashAttnRoute::TkMhaH10012Block:
      return NativeFlashAttnFunction<FlashAttnRoute::TkMhaH10012Block>::apply(query, key, value);
    case FlashAttnRoute::Portable:
    default:
      return portable_flash_attn(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
  }
}

static lean_object* mk_io_error(const std::string& msg) {
  return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(msg.c_str())));
}

} // namespace tyr_ops

TORCH_LIBRARY(tyr, m) {
  m.def("flash_attn(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, float? scale=None, bool enable_gqa=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(tyr, CompositeImplicitAutograd, m) {
  m.impl("flash_attn", TORCH_FN(tyr_ops::flash_attn_dispatch));
}

extern "C" LEAN_EXPORT lean_object* lean_torch_tyr_flash_attn_4d(
    lean_obj_arg /*batch*/,
    lean_obj_arg /*n_head*/,
    lean_obj_arg /*n_kv_head*/,
    lean_obj_arg /*q_seq*/,
    lean_obj_arg /*kv_seq*/,
    lean_obj_arg /*head_dim*/,
    b_lean_obj_arg query,
    b_lean_obj_arg key,
    b_lean_obj_arg value,
    lean_obj_arg attn_mask,
    double dropout_p,
    uint8_t is_causal,
    lean_obj_arg scale,
    uint8_t enable_gqa) {
  try {
    auto q = borrowTensor(query);
    auto k = borrowTensor(key);
    auto v = borrowTensor(value);

    c10::optional<torch::Tensor> mask_opt = c10::nullopt;
    if (!lean_is_scalar(attn_mask)) {
      mask_opt = borrowTensor(lean_ctor_get(attn_mask, 0));
    }
    lean_dec(attn_mask);

    c10::optional<double> scale_opt = c10::nullopt;
    if (!lean_is_scalar(scale)) {
      scale_opt = static_cast<double>(lean_unbox_float(lean_ctor_get(scale, 0)));
    }
    lean_dec(scale);

    auto result = tyr_ops::flash_attn_dispatch(
        q, k, v, mask_opt, dropout_p, is_causal != 0, scale_opt, enable_gqa != 0);
    return fromTorchTensor(result);
  } catch (const c10::Error& e) {
    return tyr_ops::mk_io_error(std::string("lean_torch_tyr_flash_attn_4d: ") + e.what());
  } catch (const std::exception& e) {
    return tyr_ops::mk_io_error(std::string("lean_torch_tyr_flash_attn_4d: ") + e.what());
  }
}
