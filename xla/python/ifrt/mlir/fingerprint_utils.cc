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
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "highwayhash/arch_specific.h"
#include "highwayhash/hh_types.h"
#include "highwayhash/highwayhash.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Bytecode/Encoding.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
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

// Fingerprints `module_op` as is, eliding locations.
absl::StatusOr<uint64_t> FingerprintWithoutLocations(mlir::ModuleOp module_op) {
  tsl::StatusScopedDiagnosticHandler diag_handler(module_op->getContext());

  mlir::BytecodeWriterConfig config;
  config.setElideLocations(true);

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

// Returns the `backend_config` of a `tpu_custom_call` without the locations of
// its Mosaic kernel, which would otherwise be hashed (e.g., client file paths).
// JAX writes this JSON in `CustomCallBackendConfig.to_json`
// (jax/_src/tpu_custom_call.py): `custom_call_config.body` is the base64
// serialized kernel module, whose locations JAX may instead move to
// `custom_call_config.debug_locations` (`jax_mosaic_unzip_debug_locations`).
// Like `_strip_mosaic_debug_info` in jax/_src/cache_key.py, drops
// `debug_locations` and replaces `body` by its location-free fingerprint.
// Configs and kernels that fail to parse are kept as is.
std::string StripMosaicLocations(llvm::StringRef config,
                                 mlir::MLIRContext& context) {
  llvm::Expected<llvm::json::Value> json = llvm::json::parse(config);
  if (!json) {
    llvm::consumeError(json.takeError());
    return config.str();
  }
  llvm::json::Object* kernel_config =
      json->getAsObject() ? json->getAsObject()->getObject("custom_call_config")
                          : nullptr;
  if (kernel_config == nullptr) {
    return config.str();
  }
  kernel_config->erase("debug_locations");
  std::optional<llvm::StringRef> body = kernel_config->getString("body");
  std::string kernel;
  if (body.has_value() &&
      absl::Base64Unescape(absl::string_view(body->data(), body->size()),
                           &kernel)) {
    mlir::OwningOpRef<mlir::ModuleOp> kernel_module =
        mlir::parseSourceString<mlir::ModuleOp>(kernel, &context);
    if (kernel_module) {
      absl::StatusOr<uint64_t> fingerprint =
          FingerprintWithoutLocations(*kernel_module);
      if (fingerprint.ok()) {
        (*kernel_config)["body"] = absl::StrCat(*fingerprint);
      }
    }
  }
  std::string stripped;
  llvm::raw_string_ostream os(stripped);
  os << *json;  // Object keys are printed in sorted order.
  return stripped;
}

}  // namespace

absl::StatusOr<uint64_t> FingerprintModuleOp(mlir::ModuleOp module_op) {
  // Only modules with Mosaic kernels are cloned to strip the kernel locations.
  std::unique_ptr<mlir::MLIRContext> kernel_context;
  mlir::IRMapping mapping;
  mlir::OwningOpRef<mlir::ModuleOp> clone;
  module_op.walk([&](mlir::Operation* op) {
    auto target = op->getAttrOfType<mlir::StringAttr>("call_target_name");
    auto config = op->getAttrOfType<mlir::StringAttr>("backend_config");
    if (!target || target.getValue() != "tpu_custom_call" || !config) {
      return;
    }
    if (!clone) {
      clone = mlir::cast<mlir::ModuleOp>(module_op->clone(mapping));
      kernel_context = std::make_unique<mlir::MLIRContext>(
          mlir::MLIRContext::Threading::DISABLED);
      kernel_context->allowUnregisteredDialects();
      kernel_context->getDiagEngine().registerHandler(
          [](mlir::Diagnostic&) { return mlir::success(); });
    }
    mapping.lookup(op)->setAttr(
        "backend_config",
        mlir::StringAttr::get(
            op->getContext(),
            StripMosaicLocations(config.getValue(), *kernel_context)));
  });
  return FingerprintWithoutLocations(clone ? *clone : module_op);
}

}  // namespace ifrt
}  // namespace xla
