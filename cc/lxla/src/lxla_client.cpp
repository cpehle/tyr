#include "lxla_client.h"

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/pjrt/gpu/gpu_helpers.h"
#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/tpu_client.h"

#define LXLA_STATUS_MACROS_CONCAT_NAME(x, y)                                 \
  LXLA_STATUS_MACROS_CONCAT_NAME_IMPL(x, y)

#define LXLA_STATUS_MACROS_CONCAT_NAME_IMPL(x, y) x##y

// Macro to be used to consume StatusOr. Will bind lhs
// to value if the status is OK, otherwise will return
// the status.
#define LXLA_ASSIGN_OR_RETURN(lhs, rexpr)                                    \
  LXLA_ASSIGN_OR_RETURN_IMPL(                                                \
    LXLA_STATUS_MACROS_CONCAT_NAME(                                          \
      _status_or_value, __COUNTER__),                                        \
  lhs, rexpr)

#define LXLA_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr)                     \
  auto statusor = (rexpr);                                                   \
  if (!statusor.ok()) {                                                      \
    return statusor.status();                                                \
  }                                                                          \
  lhs = std::move(statusor.value());


namespace lxla {

LxlaBuffer::LxlaBuffer(std::unique_ptr<xla::PjRtBuffer> buffer): buffer_(std::move(buffer)) {}

xla::Status LxlaBuffer::Deallocate() {
  if (buffer_->IsDeleted()) {
    return xla::FailedPrecondition("Attempt to deallocate already deallocated buffer.");
  }
  else {
    buffer_->Delete();
    return tsl::OkStatus();
  }
}

xla::StatusOr<LxlaBuffer *> LxlaBuffer::CopyToDevice(xla::PjRtDevice * dst_device) {
  LXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtBuffer> buf,
      buffer_->CopyToDevice(dst_device));
  return new LxlaBuffer(std::move(buf));
}

xla::StatusOr<LxlaExecutable*> LxlaClient::Compile(const xla::XlaComputation& computation,
                                                   std::vector<xla::Shape*> argument_layouts,
                                                   xla::ExecutableBuildOptions& options,
                                                   bool compile_portable_executable) {
  std::vector<xla::Shape> layouts;
  layouts.reserve(argument_layouts.size());
  for (auto shape : argument_layouts) {
    xla::Shape cpy_shape = xla::ShapeUtil::MakeShape(shape->element_type(), shape->dimensions());
    xla::LayoutUtil::ClearLayout(&cpy_shape);
    layouts.push_back(cpy_shape);
  }

  xla::CompileOptions compile_opts;
  compile_opts.argument_layouts = layouts;
  compile_opts.parameter_is_tupled_arguments = false;
  compile_opts.executable_build_options = options;
  compile_opts.compile_portable_executable = compile_portable_executable;

  LXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtLoadedExecutable> executable,
    client_->Compile(computation, std::move(compile_opts)));
  LXLA_ASSIGN_OR_RETURN(absl::optional<std::string> fingerprint,
    client_->ExecutableFingerprint(*executable));

  return new LxlaExecutable(std::move(executable), std::move(fingerprint), this);
}


xla::StatusOr<LxlaClient*> GetHostClient() {
  LXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
    xla::GetTfrtCpuClient(false));

  return new LxlaClient(std::move(client));
}

xla::StatusOr<LxlaClient*> GetGpuClient(double memory_fraction,
                                        bool preallocate,
                                        xla::GpuAllocatorConfig::Kind kind) {
  xla::GpuAllocatorConfig allocator_config = {
    .kind = kind,
    .memory_fraction = memory_fraction,
    .preallocate = preallocate
  };

  LXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
    xla::GetStreamExecutorGpuClient(false, allocator_config, nullptr, 0));

  return new LxlaClient(std::move(client));
}

xla::StatusOr<LxlaClient*> GetTpuClient() {
  LXLA_ASSIGN_OR_RETURN(std::shared_ptr<xla::PjRtClient> client,
    xla::GetTpuClient(32));

  return new LxlaClient(std::move(client));
}

}