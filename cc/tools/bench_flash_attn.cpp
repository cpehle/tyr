#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cuda_runtime_api.h>
#include <lean/lean.h>
#include <torch/torch.h>

extern "C" void lean_initialize();

namespace tyr_ops {
torch::Tensor flash_attn_dispatch(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const c10::optional<torch::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const c10::optional<double>& scale,
    bool enable_gqa);
torch::Tensor flash_attn_dispatch_generated(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const c10::optional<torch::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const c10::optional<double>& scale,
    bool enable_gqa);
torch::Tensor flash_attn_dispatch_vendored_tk(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const c10::optional<torch::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const c10::optional<double>& scale,
    bool enable_gqa);
} // namespace tyr_ops

namespace {

struct CaseDef {
  std::string id;
  int64_t batch;
  int64_t q_heads;
  int64_t kv_heads;
  int64_t seq;
  int64_t head_dim;
  bool fwd_bwd;
};

struct Options {
  std::vector<std::string> case_ids{"native_dense_128x64", "native_dense_768x64"};
  std::vector<std::string> backends{"torch_sdpa", "tyr_runtime", "tyr_generated"};
  int warmup{5};
  int iters{20};
  int repeats{3};
  uint64_t seed{20260422};
  bool jsonl_stdout{false};
  std::string jsonl_out;
};

struct RunResult {
  torch::Tensor out;
  torch::Tensor dQ;
  torch::Tensor dK;
  torch::Tensor dV;
};

struct Metrics {
  bool ok{false};
  double out_mae{0.0};
  double out_max{0.0};
  double dq_mae{0.0};
  double dq_max{0.0};
  double dk_mae{0.0};
  double dk_max{0.0};
  double dv_mae{0.0};
  double dv_max{0.0};
};

static const std::vector<CaseDef> kCases{
    {"native_dense_128x64", 1, 1, 1, 128, 64, true},
    {"native_dense_768x64", 1, 1, 1, 768, 64, true},
};

std::vector<std::string> split_csv(const std::string& text) {
  std::vector<std::string> out;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

const CaseDef& find_case(const std::string& id) {
  for (const auto& c : kCases) {
    if (c.id == id) {
      return c;
    }
  }
  throw std::invalid_argument("unknown case: " + id);
}

void print_cases() {
  for (const auto& c : kCases) {
    std::cout << c.id << " seq=" << c.seq << " head_dim=" << c.head_dim
              << " fwd_bwd=" << (c.fwd_bwd ? "true" : "false") << "\n";
  }
}

void print_backends() {
  std::cout << "torch_sdpa\n";
  std::cout << "tyr_runtime\n";
  std::cout << "tyr_generated\n";
  std::cout << "tk_vendored\n";
}

Options parse_args(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    if (arg == "--case") {
      const auto value = next("--case");
      opts.case_ids = (value == "native_now") ? std::vector<std::string>{"native_dense_128x64", "native_dense_768x64"} : split_csv(value);
    } else if (arg == "--backend") {
      const auto value = next("--backend");
      opts.backends = (value == "all")
          ? std::vector<std::string>{"torch_sdpa", "tyr_runtime", "tyr_generated", "tk_vendored"}
          : split_csv(value);
    } else if (arg == "--warmup") {
      opts.warmup = std::stoi(next("--warmup"));
    } else if (arg == "--iters") {
      opts.iters = std::stoi(next("--iters"));
    } else if (arg == "--repeats") {
      opts.repeats = std::stoi(next("--repeats"));
    } else if (arg == "--seed") {
      opts.seed = static_cast<uint64_t>(std::stoull(next("--seed")));
    } else if (arg == "--jsonl-out") {
      opts.jsonl_out = next("--jsonl-out");
    } else if (arg == "--jsonl-stdout") {
      opts.jsonl_stdout = true;
    } else if (arg == "--list-cases") {
      print_cases();
      std::exit(0);
    } else if (arg == "--list-backends") {
      print_backends();
      std::exit(0);
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: bench_flash_attn [--case native_now|id[,id...]] [--backend all|name[,name...]]\n"
          << "                         [--warmup N] [--iters N] [--repeats N]\n"
          << "                         [--jsonl-out PATH] [--jsonl-stdout]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  return opts;
}

void sync_cuda() {
  if (torch::cuda::is_available()) {
    const auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << "bench_flash_attn: cudaDeviceSynchronize failed: "
                << cudaGetErrorString(err) << "\n";
      std::exit(3);
    }
  }
}

torch::Tensor make_leaf(const torch::Tensor& x) {
  auto leaf = x.detach().clone();
  leaf.set_requires_grad(true);
  return leaf;
}

void zero_grad(torch::Tensor& x) {
  auto g = x.grad();
  if (g.defined()) {
    g.zero_();
  }
}

torch::Tensor run_backend(
    const std::string& backend,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v) {
  if (backend == "torch_sdpa") {
    return torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, false);
  }
  if (backend == "tyr_runtime") {
    return tyr_ops::flash_attn_dispatch(q, k, v, c10::nullopt, 0.0, false, c10::nullopt, false);
  }
  if (backend == "tyr_generated") {
    return tyr_ops::flash_attn_dispatch_generated(q, k, v, c10::nullopt, 0.0, false, c10::nullopt, false);
  }
  if (backend == "tk_vendored") {
    return tyr_ops::flash_attn_dispatch_vendored_tk(q, k, v, c10::nullopt, 0.0, false, c10::nullopt, false);
  }
  throw std::invalid_argument("unknown backend: " + backend);
}

RunResult run_once(
    const std::string& backend,
    const torch::Tensor& q_base,
    const torch::Tensor& k_base,
    const torch::Tensor& v_base,
    const torch::Tensor& dO_base) {
  auto q = make_leaf(q_base);
  auto k = make_leaf(k_base);
  auto v = make_leaf(v_base);
  auto out = run_backend(backend, q, k, v);
  auto loss = (out.to(torch::kFloat32) * dO_base.to(torch::kFloat32)).sum();
  loss.backward();
  sync_cuda();
  return {
      out.detach(),
      q.grad().detach().to(torch::kFloat32),
      k.grad().detach().to(torch::kFloat32),
      v.grad().detach().to(torch::kFloat32),
  };
}

double mean_abs(const torch::Tensor& a, const torch::Tensor& b) {
  return (a.to(torch::kFloat32) - b.to(torch::kFloat32)).abs().mean().item<double>();
}

double max_abs(const torch::Tensor& a, const torch::Tensor& b) {
  return std::get<0>((a.to(torch::kFloat32) - b.to(torch::kFloat32)).abs().max(0)).max().item<double>();
}

bool tensors_allclose(const torch::Tensor& a, const torch::Tensor& b, double rtol, double atol) {
  return torch::allclose(a.to(torch::kFloat32), b.to(torch::kFloat32), rtol, atol);
}

Metrics compare_to_ref(const RunResult& got, const RunResult& ref) {
  Metrics m;
  m.out_mae = mean_abs(got.out, ref.out);
  m.out_max = max_abs(got.out, ref.out);
  m.dq_mae = mean_abs(got.dQ, ref.dQ);
  m.dq_max = max_abs(got.dQ, ref.dQ);
  m.dk_mae = mean_abs(got.dK, ref.dK);
  m.dk_max = max_abs(got.dK, ref.dK);
  m.dv_mae = mean_abs(got.dV, ref.dV);
  m.dv_max = max_abs(got.dV, ref.dV);
  m.ok =
      tensors_allclose(got.out, ref.out, 5e-2, 5e-2) &&
      tensors_allclose(got.dQ, ref.dQ, 8e-2, 8e-2) &&
      tensors_allclose(got.dK, ref.dK, 8e-2, 8e-2) &&
      tensors_allclose(got.dV, ref.dV, 8e-2, 8e-2);
  return m;
}

double time_backend_ms(
    const std::string& backend,
    const torch::Tensor& q_base,
    const torch::Tensor& k_base,
    const torch::Tensor& v_base,
    const torch::Tensor& dO_base,
    int warmup,
    int iters) {
  auto q = make_leaf(q_base);
  auto k = make_leaf(k_base);
  auto v = make_leaf(v_base);
  auto step = [&]() {
    zero_grad(q);
    zero_grad(k);
    zero_grad(v);
    auto out = run_backend(backend, q, k, v);
    auto loss = (out.to(torch::kFloat32) * dO_base.to(torch::kFloat32)).sum();
    loss.backward();
  };
  for (int i = 0; i < warmup; ++i) {
    step();
  }
  sync_cuda();

  cudaEvent_t start_event{};
  cudaEvent_t stop_event{};
  auto err = cudaEventCreate(&start_event);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaEventCreate(start) failed: ") + cudaGetErrorString(err));
  }
  err = cudaEventCreate(&stop_event);
  if (err != cudaSuccess) {
    cudaEventDestroy(start_event);
    throw std::runtime_error(std::string("cudaEventCreate(stop) failed: ") + cudaGetErrorString(err));
  }

  err = cudaEventRecord(start_event, 0);
  if (err != cudaSuccess) {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    throw std::runtime_error(std::string("cudaEventRecord(start) failed: ") + cudaGetErrorString(err));
  }
  for (int i = 0; i < iters; ++i) {
    step();
  }
  err = cudaEventRecord(stop_event, 0);
  if (err != cudaSuccess) {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    throw std::runtime_error(std::string("cudaEventRecord(stop) failed: ") + cudaGetErrorString(err));
  }
  err = cudaEventSynchronize(stop_event);
  if (err != cudaSuccess) {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    throw std::runtime_error(std::string("cudaEventSynchronize(stop) failed: ") + cudaGetErrorString(err));
  }
  float total_ms_float = 0.0f;
  err = cudaEventElapsedTime(&total_ms_float, start_event, stop_event);
  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaEventElapsedTime failed: ") + cudaGetErrorString(err));
  }
  const double total_ms = static_cast<double>(total_ms_float);
  return total_ms / static_cast<double>(std::max(1, iters));
}

std::string json_bool(bool value) {
  return value ? "true" : "false";
}

std::string route_for(const std::string& backend, const CaseDef& c) {
  if (backend == "torch_sdpa") {
    return "torch_sdpa";
  }
  if (backend == "tyr_runtime" && c.batch == 1 && c.q_heads == 1 && c.kv_heads == 1 &&
      c.head_dim == 64 && (c.seq == 128 || c.seq == 768)) {
    return c.seq == 128 ? "tk_mha_h100_2block" : "vendored_tk_mha_h100_12block";
  }
  if (backend == "tk_vendored" && c.batch == 1 && c.q_heads == 1 && c.kv_heads == 1 &&
      c.head_dim == 64 && (c.seq == 128 || c.seq == 768)) {
    return c.seq == 128 ? "generated_fallback_2block" : "vendored_tk_mha_h100_12block";
  }
  if (backend == "tyr_generated" && c.batch == 1 && c.q_heads == 1 && c.kv_heads == 1 &&
      c.head_dim == 64 && (c.seq == 128 || c.seq == 768)) {
    return c.seq == 128 ? "tk_mha_h100_2block" : "tk_mha_h100_12block";
  }
  return "portable";
}

std::string summary_json(
    const CaseDef& c,
    const std::string& backend,
    const std::string& status,
    const Metrics& metrics,
    double p50_ms,
    double speedup_vs_sdpa) {
  std::ostringstream os;
  os << "{"
     << "\"event\":\"summary\","
     << "\"caseId\":\"" << c.id << "\","
     << "\"backendExecuted\":\"" << backend << "\","
     << "\"status\":\"" << status << "\","
     << "\"route\":\"" << route_for(backend, c) << "\","
     << "\"seq\":" << c.seq << ","
     << "\"headDim\":" << c.head_dim << ","
     << "\"correctnessOk\":" << json_bool(metrics.ok) << ","
     << "\"latencyMsP50\":" << p50_ms << ","
     << "\"speedupVsSdpaP50\":" << speedup_vs_sdpa << ","
     << "\"outMae\":" << metrics.out_mae << ","
     << "\"outMax\":" << metrics.out_max << ","
     << "\"dqMae\":" << metrics.dq_mae << ","
     << "\"dqMax\":" << metrics.dq_max << ","
     << "\"dkMae\":" << metrics.dk_mae << ","
     << "\"dkMax\":" << metrics.dk_max << ","
     << "\"dvMae\":" << metrics.dv_mae << ","
     << "\"dvMax\":" << metrics.dv_max
     << "}";
  return os.str();
}

double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void emit_line(std::ostream* file, bool to_stdout, const std::string& line) {
  if (to_stdout) {
    std::cout << line << "\n";
  }
  if (file != nullptr) {
    (*file) << line << "\n";
  }
}

void initialize_lean_runtime() {
  lean_initialize();
  lean_io_mark_end_initialization();
}

} // namespace

int main(int argc, char** argv) {
  try {
    const auto opts = parse_args(argc, argv);
    if (!torch::cuda::is_available()) {
      throw std::runtime_error("CUDA is not available");
    }
    initialize_lean_runtime();
    torch::manual_seed(static_cast<int64_t>(opts.seed));

    std::ofstream jsonl_file;
    if (!opts.jsonl_out.empty()) {
      const std::filesystem::path path(opts.jsonl_out);
      if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
      }
      jsonl_file.open(path);
      if (!jsonl_file) {
        throw std::runtime_error("failed to open --jsonl-out: " + opts.jsonl_out);
      }
    }
    std::ostream* file_out = jsonl_file ? &jsonl_file : nullptr;

    emit_line(file_out, opts.jsonl_stdout,
        "{\"event\":\"meta\",\"tool\":\"cc/tools/bench_flash_attn\",\"device\":\"cuda:0\",\"timer\":\"cuda_event\"}");

    bool all_ok = true;
    for (const auto& case_id : opts.case_ids) {
      const auto& c = find_case(case_id);
      const auto device = torch::Device(torch::kCUDA, 0);
      const auto opts_bf16 = torch::TensorOptions().device(device).dtype(torch::kBFloat16);
      const auto shape = std::vector<int64_t>{c.batch, c.q_heads, c.seq, c.head_dim};
      const auto kv_shape = std::vector<int64_t>{c.batch, c.kv_heads, c.seq, c.head_dim};
      auto q_base = torch::randn(shape, opts_bf16);
      auto k_base = torch::randn(kv_shape, opts_bf16);
      auto v_base = torch::randn(kv_shape, opts_bf16);
      auto dO_base = torch::randn(shape, opts_bf16);

      const auto ref = run_once("torch_sdpa", q_base, k_base, v_base, dO_base);
      std::vector<std::pair<std::string, double>> p50_by_backend;
      std::vector<std::tuple<std::string, Metrics, double>> summaries;

      for (const auto& backend : opts.backends) {
        if (backend != "torch_sdpa" && backend != "tyr_runtime" &&
            backend != "tyr_generated" && backend != "tk_vendored") {
          throw std::invalid_argument("unknown backend: " + backend);
        }
        const auto got = (backend == "torch_sdpa") ? ref : run_once(backend, q_base, k_base, v_base, dO_base);
        const auto metrics = compare_to_ref(got, ref);
        std::vector<double> samples;
        samples.reserve(static_cast<size_t>(std::max(0, opts.repeats)));
        for (int r = 0; r < opts.repeats; ++r) {
          samples.push_back(time_backend_ms(
              backend, q_base, k_base, v_base, dO_base, opts.warmup, opts.iters));
        }
        const auto p50_ms = median(samples);
        p50_by_backend.push_back({backend, p50_ms});
        summaries.push_back({backend, metrics, p50_ms});
        all_ok = all_ok && metrics.ok;
      }

      double sdpa_p50 = 0.0;
      for (const auto& [backend, p50] : p50_by_backend) {
        if (backend == "torch_sdpa") {
          sdpa_p50 = p50;
        }
      }
      for (const auto& [backend, metrics, p50_ms] : summaries) {
        const double speedup = (sdpa_p50 > 0.0 && p50_ms > 0.0) ? sdpa_p50 / p50_ms : 0.0;
        emit_line(file_out, opts.jsonl_stdout,
            summary_json(c, backend, "ok", metrics, p50_ms, speedup));
        std::cerr << "bench case=" << c.id
                  << " backend=" << backend
                  << " route=" << route_for(backend, c)
                  << " correctness_ok=" << (metrics.ok ? "true" : "false")
                  << " p50_ms=" << p50_ms
                  << " speedup_vs_sdpa=" << speedup << "\n";
      }
    }
    return all_ok ? 0 : 1;
  } catch (const c10::Error& e) {
    std::cerr << "bench_flash_attn: c10 error: " << e.what() << "\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "bench_flash_attn: error: " << e.what() << "\n";
    return 2;
  }
}
