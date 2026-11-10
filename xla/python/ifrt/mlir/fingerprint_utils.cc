/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/python/ifrt/mlir/fingerprint_utils.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "highwayhash/arch_specific.h"
#include "highwayhash/hh_types.h"
#include "highwayhash/highwayhash.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Bytecode/Encoding.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/hlo.pb.h"
#include "xla/status_macros.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"

namespace xla {
namespace ifrt {

namespace {

// Calculates a HighwayHash fingerprint in a streaming manner.
class HighwayHashStream final : public llvm::raw_ostream {
 public:
  HighwayHashStream() : hash_(kHighwayHashKey), pos_(0) { SetUnbuffered(); }

  ~HighwayHashStream() override { CHECK_EQ(buffer_pos_, 0); }

  // Destructively calculates the fingerprint of the data consumed so far.
  uint64_t fingerprint() && {
    flush_buffer();
    highwayhash::HHResult64 result;
    hash_.Finalize(&result);
    return result;
  }

 private:
  // `HighwayHashCatT::Append` updates the hash directly from the source buffer
  // if the provided buffer size is larger than `sizeof(HHPacket)`.
  static constexpr size_t kBufferSize = sizeof(highwayhash::HHPacket);

  // Arbitrarily chosen, forever-unchanging hash key required by HighwayHash.
  static constexpr highwayhash::HHKey kHighwayHashKey = {
      0x4451e30f87db9609ULL,
      0xca7358a1fd2737f8ULL,
      0x4b2c991fcee4fdeaULL,
      0x0b2658e18326f6baULL,
  };

  void write_impl(const char* Ptr, size_t Size) final {
    // For tiny writes, it is more efficient to accumulate the data to a buffer
    // first and flush it since `HighwayHashCatT::Append` is optimized for
    // large writes.
    static constexpr size_t kSmallWriteSize = 4;
    static_assert(kSmallWriteSize <= kBufferSize);

    if (Size <= kSmallWriteSize) {
      if (buffer_pos_ + Size > kBufferSize) {
        flush_buffer();
      }
      std::memcpy(buffer_ + buffer_pos_, Ptr, Size);
      buffer_pos_ += Size;
    } else {
      flush_buffer();
      hash_.Append(Ptr, Size);
    }
    pos_ += Size;
  }

  uint64_t current_pos() const final { return pos_; }

  void flush_buffer() {
    if (buffer_pos_ > 0) {
      hash_.Append(buffer_, buffer_pos_);
      buffer_pos_ = 0;
    }
  }

  highwayhash::HighwayHashCatT<HH_TARGET> hash_;
  uint64_t pos_;

  char buffer_[kBufferSize];
  uint64_t buffer_pos_ = 0;
};

// Writes a `tpu_custom_call` `backend_config` without the Mosaic kernel
// locations that JAX stores in `custom_call_config.debug_locations` (see
// `CustomCallBackendConfig.to_json` and `jax_mosaic_unzip_debug_locations` in
// jax/_src/tpu_custom_call.py). Other attributes are written as usual.
mlir::LogicalResult WriteWithoutMosaicDebugLocations(
    mlir::Attribute attr, std::optional<llvm::StringRef>& dialect_group_name,
    mlir::DialectBytecodeWriter& writer) {
  auto config = mlir::dyn_cast<mlir::StringAttr>(attr);
  if (!config || !config.getValue().contains("\"debug_locations\"")) {
    return mlir::failure();
  }
  llvm::Expected<llvm::json::Value> json = llvm::json::parse(config.getValue());
  if (!json) {
    llvm::consumeError(json.takeError());
    return mlir::failure();
  }
  llvm::json::Object* kernel_config =
      json->getAsObject() ? json->getAsObject()->getObject("custom_call_config")
                          : nullptr;
  if (kernel_config == nullptr || !kernel_config->erase("debug_locations")) {
    return mlir::failure();
  }
  std::string stripped;
  llvm::raw_string_ostream os(stripped);
  os << *json;  // Object keys are printed in sorted order.
  // A separate group keeps this encoding distinct from builtin attributes.
  dialect_group_name = llvm::StringLiteral("xla.mosaic_backend_config");
  writer.writeUnownedBlob(stripped);  // Copies `stripped`.
  return mlir::success();
}

}  // namespace

absl::StatusOr<uint64_t> FingerprintModuleOp(mlir::ModuleOp module_op) {
  tsl::StatusScopedDiagnosticHandler diag_handler(module_op->getContext());

  mlir::BytecodeWriterConfig config;
  config.setElideLocations(true);
  config.attachAttributeCallback(WriteWithoutMosaicDebugLocations);

  // Use a version before `kUseListOrdering` due to an MLIR bug where use list
  // ordering is not stable.
  //
  // TODO(b/503120525): Remove this workaround once
  // https://github.com/llvm/llvm-project/pull/191942 lands.
  config.setDesiredBytecodeVersion(
      mlir::bytecode::BytecodeVersion::kLazyLoading);

  HighwayHashStream os;
  mlir::LogicalResult result = mlir::writeBytecodeToFile(module_op, os, config);
  ABSL_RETURN_IF_ERROR(diag_handler.consumeStatus());
  TF_RET_CHECK(mlir::succeeded(result));
  return std::move(os).fingerprint();
}

}  // namespace ifrt
}  // namespace xla
