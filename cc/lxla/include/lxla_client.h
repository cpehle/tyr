#pragma once

#include <lean/lean.h>

#include <memory>
#include <vector>
#include <utility>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/compiler/xla/pjrt/gpu/gpu_helpers.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"


namespace lxla {

class LxlaClient;

class LxlaBuffer {
 public:
  LxlaBuffer(std::unique_ptr<xla::PjRtBuffer> buffer);

  int device_id() { return buffer_->device()->id(); }
  xla::PjRtBuffer* buffer() { return buffer_.get(); }
  xla::StatusOr<LxlaBuffer*> CopyToDevice(xla::PjRtDevice * dst_device);
  xla::Status Deallocate();

  ~LxlaBuffer() {
  }

 private:
  std::unique_ptr<xla::PjRtBuffer> buffer_;
};

class LxlaExecutable {
 public:
  LxlaExecutable(std::unique_ptr<xla::PjRtLoadedExecutable> executable,
                 absl::optional<std::string> fingerprint,
                 LxlaClient* client);

  xla::PjRtLoadedExecutable* executable() { return executable_.get(); }

  // xla::StatusOr<ERL_NIF_TERM> Run(ErlNifEnv* env,
  //                                 ERL_NIF_TERM arguments,
  //                                 int device_id);

 private:
  std::unique_ptr<xla::PjRtLoadedExecutable> executable_;
  absl::optional<std::string> fingerprint_;
  LxlaClient* client_;
};

class LxlaClient {
 public:
  explicit LxlaClient(std::shared_ptr<xla::PjRtClient> client);

  virtual ~LxlaClient() = default;

  xla::PjRtClient* client() { return client_.get(); }

  // Compiles the given computation with the given compile options
  xla::StatusOr<LxlaExecutable*> Compile(const xla::XlaComputation&,
                                         std::vector<xla::Shape*> argument_layouts,
                                         xla::ExecutableBuildOptions& options,
                                         bool compile_portable_executable);

  // xla::StatusOr<LxlaBuffer*> BufferFromBinary(ErlNifEnv* env,
  //                                             ERL_NIF_TERM binary_term,
  //                                             xla::Shape& shape,
  //                                             int device_id);

 private:
  std::shared_ptr<xla::PjRtClient> client_;
};

}
