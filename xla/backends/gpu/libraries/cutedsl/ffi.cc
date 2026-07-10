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

#include "xla/backends/gpu/ffi.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"
#include "xla/backends/gpu/libraries/cutedsl/runtime_api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"

namespace xla::gpu::cutedsl {
namespace {

constexpr absl::string_view kCutlassCallTarget = "__xla_gpu_cutedsl_call_v3";
constexpr absl::string_view kCutlassCallNoCudaGraphTarget =
    "__xla_gpu_cutedsl_call_no_cuda_graph_v3";
constexpr absl::string_view kFunctionPrefix = "cutlass_call";
constexpr size_t kCacheKeySize = 32;
// Avoid heap allocation for the common case while retaining support for calls
// with an arbitrary number of buffers.
constexpr size_t kInlineBufferCount = 8;

// A POD descriptor matching cutlass.jax.types.JaxArray. Generated CuTeDSL
// wrappers receive a pointer to one of these descriptors for every XLA buffer.
struct CuteXlaFfiBuffer {
  void* buffer;
  const int64_t* shape;
};

std::string RuntimeError(const RuntimeFunctions& functions,
                         CuteDSLRT_Error_t error) {
  const char* name = functions.get_error_name(error);
  const char* description = functions.get_error_string(error);
  return absl::StrFormat(
      "%s (error %d): %s", name == nullptr ? "Unknown" : name, error,
      description == nullptr ? "Unknown error" : description);
}

absl::Status ValidateCacheKey(absl::string_view module, absl::string_view key) {
  if (key.size() != kCacheKeySize) {
    return absl::InvalidArgumentError(
        absl::StrFormat("CuTeDSL cache key must be %d bytes; got %d",
                        kCacheKeySize, key.size()));
  }

  llvm::SHA256 hasher;
  hasher.update(llvm::StringRef(module.data(), module.size()));
  std::array<uint8_t, kCacheKeySize> digest = hasher.final();
  if (!std::equal(digest.begin(), digest.end(),
                  reinterpret_cast<const uint8_t*>(key.data()))) {
    return absl::InvalidArgumentError(
        "CuTeDSL cache key does not match the module SHA-256 digest");
  }

  return absl::OkStatus();
}

// Owns a runtime module and its cutlass_call entry point. The Bazel-linked
// runtime has the same lifetime as the image containing this registration.
class ModuleAndFunction {
 public:
  ModuleAndFunction(RuntimeFunctions functions, CuteDSLRT_Module_t* module,
                    CuteDSLRT_Function_t* function)
      : functions_(functions), module_(module), function_(function) {}

  ModuleAndFunction(const ModuleAndFunction&) = delete;
  ModuleAndFunction& operator=(const ModuleAndFunction&) = delete;

  ~ModuleAndFunction() {
    if (module_ == nullptr) return;
    CuteDSLRT_Error_t error = functions_.module_destroy(module_);
    if (error != kCuteDslRtSuccess) {
      LOG(ERROR) << "Failed to destroy CuTeDSL runtime module: "
                 << RuntimeError(functions_, error);
    }
  }

  static absl::StatusOr<std::shared_ptr<ModuleAndFunction>> Create(
      const RuntimeFunctions& functions, absl::string_view module_bytes) {
    CuteDSLRT_Module_t* module = nullptr;
    // Standalone static and shared runtimes register their CUDA helper symbols
    // directly with ORC. shared_libs is reserved for actual dependencies of
    // the generated module.
    CuteDSLRT_Error_t error = functions.module_create_from_bytes(
        &module, reinterpret_cast<const unsigned char*>(module_bytes.data()),
        module_bytes.size(), /*shared_libs=*/nullptr,
        /*shared_libs_size=*/0);
    if (error != kCuteDslRtSuccess) {
      if (module != nullptr) {
        CuteDSLRT_Error_t destroy_error = functions.module_destroy(module);
        if (destroy_error != kCuteDslRtSuccess) {
          LOG(ERROR) << "Failed to destroy CuTeDSL runtime module after "
                        "creation failed: "
                     << RuntimeError(functions, destroy_error);
        }
      }
      return absl::InternalError(
          absl::StrFormat("Failed to create CuTeDSL runtime module: %s",
                          RuntimeError(functions, error)));
    }
    if (module == nullptr) {
      return absl::InternalError(
          "CuTeDSL runtime created a null module without returning an error");
    }

    CuteDSLRT_Function_t* function = nullptr;
    error = functions.module_get_function(&function, module,
                                          kFunctionPrefix.data());
    if (error != kCuteDslRtSuccess || function == nullptr) {
      CuteDSLRT_Error_t destroy_error = functions.module_destroy(module);
      if (destroy_error != kCuteDslRtSuccess) {
        LOG(ERROR) << "Failed to destroy CuTeDSL runtime module after function "
                      "lookup failed: "
                   << RuntimeError(functions, destroy_error);
      }
      if (error != kCuteDslRtSuccess) {
        return absl::InternalError(
            absl::StrFormat("Failed to load CuTeDSL cutlass_call function: %s",
                            RuntimeError(functions, error)));
      }
      return absl::InternalError(
          "CuTeDSL runtime returned a null cutlass_call function without "
          "returning an error");
    }

    return std::make_shared<ModuleAndFunction>(functions, module, function);
  }

  const RuntimeFunctions& functions() const { return functions_; }
  CuteDSLRT_Function_t* function() const { return function_; }

 private:
  RuntimeFunctions functions_;
  CuteDSLRT_Module_t* module_;
  CuteDSLRT_Function_t* function_;
};

// Weak ownership ties runtime-module lifetime to live XLA executables while
// still sharing identical modules across calls and executables.
class ModuleCache {
 public:
  static ModuleCache& Global() {
    static auto* cache = new ModuleCache;
    return *cache;
  }

  absl::StatusOr<std::shared_ptr<ModuleAndFunction>> GetOrCreate(
      absl::string_view module, absl::string_view key) {
    absl::Status key_status = ValidateCacheKey(module, key);
    if (!key_status.ok()) return key_status;

    absl::StatusOr<RuntimeFunctions> functions = GetRuntimeFunctions();
    if (!functions.ok()) return functions.status();

    std::string cache_key(key);
    absl::MutexLock lock(&mu_);
    auto it = cache_.find(cache_key);
    if (it != cache_.end()) {
      if (std::shared_ptr<ModuleAndFunction> cached = it->second.lock()) {
        return cached;
      }
      cache_.erase(it);
    }

    absl::StatusOr<std::shared_ptr<ModuleAndFunction>> loaded =
        ModuleAndFunction::Create(*functions, module);
    if (!loaded.ok()) return loaded.status();
    cache_.emplace(std::move(cache_key), *loaded);
    return *loaded;
  }

 private:
  absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::weak_ptr<ModuleAndFunction>> cache_
      ABSL_GUARDED_BY(mu_);
};

class CutlassCallStateV3 {
 public:
  void SetModule(std::shared_ptr<ModuleAndFunction> module) {
    absl::MutexLock lock(&mu_);
    module_ = std::move(module);
  }

  absl::StatusOr<std::shared_ptr<ModuleAndFunction>> GetModule() const {
    absl::MutexLock lock(&mu_);
    if (module_ == nullptr) {
      return absl::FailedPreconditionError(
          "CuTeDSL custom call executed before prepare completed");
    }
    return module_;
  }

 private:
  mutable absl::Mutex mu_;
  std::shared_ptr<ModuleAndFunction> module_ ABSL_GUARDED_BY(mu_);
};

absl::StatusOr<std::unique_ptr<CutlassCallStateV3>> Instantiate(
    absl::string_view module, absl::string_view key) {
  // Runtime access is deferred to prepare so registration and metadata queries
  // also work in binaries built with the unavailable provider.
  if (module.empty()) {
    return absl::InvalidArgumentError("CuTeDSL module attribute is empty");
  }
  if (key.size() != kCacheKeySize) {
    return absl::InvalidArgumentError(
        absl::StrFormat("CuTeDSL cache key must be %d bytes; got %d",
                        kCacheKeySize, key.size()));
  }
  return std::make_unique<CutlassCallStateV3>();
}

absl::Status Prepare(CutlassCallStateV3* state, ffi::RemainingArgs,
                     ffi::RemainingRets, absl::string_view module,
                     absl::string_view key) {
  absl::StatusOr<std::shared_ptr<ModuleAndFunction>> loaded =
      ModuleCache::Global().GetOrCreate(module, key);
  if (!loaded.ok()) return loaded.status();
  state->SetModule(std::move(*loaded));
  return absl::OkStatus();
}

absl::Status Initialize() { return absl::OkStatus(); }

absl::Status ExecuteFunction(const RuntimeFunctions& functions, void* stream,
                             CuteDSLRT_Function_t* function,
                             ffi::RemainingArgs inputs,
                             ffi::RemainingRets outputs) {
  absl::InlinedVector<CuteXlaFfiBuffer, kInlineBufferCount> buffers;
  buffers.reserve(inputs.size() + outputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    absl::StatusOr<ffi::AnyBuffer> input = inputs.get<ffi::AnyBuffer>(i);
    if (!input.ok()) return input.status();
    ffi::AnyBuffer::Dimensions dimensions = input->dimensions();
    buffers.push_back({input->untyped_data(),
                       dimensions.empty() ? nullptr : dimensions.data()});
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    absl::StatusOr<ffi::Result<ffi::AnyBuffer>> output =
        outputs.get<ffi::AnyBuffer>(i);
    if (!output.ok()) return output.status();
    ffi::AnyBuffer::Dimensions dimensions = (*output)->dimensions();
    buffers.push_back({(*output)->untyped_data(),
                       dimensions.empty() ? nullptr : dimensions.data()});
  }

  // CuTeDSL compiled wrappers use MLIR's packed C interface: every entry is a
  // pointer to storage containing the corresponding argument value.
  absl::InlinedVector<void*, kInlineBufferCount + 1> arguments;
  arguments.reserve(buffers.size() + 1);
  arguments.push_back(stream);
  for (CuteXlaFfiBuffer& buffer : buffers) arguments.push_back(&buffer);

  absl::InlinedVector<void*, kInlineBufferCount + 2> packed_arguments;
  packed_arguments.reserve(arguments.size() + 1);
  for (void*& argument : arguments) packed_arguments.push_back(&argument);

  // Generated wrappers write the CUDA launch status to the final argument.
  // CUDA error enums use a 32-bit representation; keeping the ABI type here
  // avoids a link-time dependency on the CUDA runtime.
  int32_t cuda_error = 0;
  packed_arguments.push_back(&cuda_error);

  CuteDSLRT_Error_t error = functions.function_run(
      function, packed_arguments.data(), packed_arguments.size());
  if (error != kCuteDslRtSuccess) {
    return absl::InternalError(
        absl::StrFormat("Failed to execute CuTeDSL kernel: %s; CUDA error %d",
                        RuntimeError(functions, error), cuda_error));
  }
  if (cuda_error != 0) {
    return absl::InternalError(
        absl::StrFormat("CuTeDSL kernel returned CUDA error %d", cuda_error));
  }

  return absl::OkStatus();
}

absl::Status Execute(void* stream, CutlassCallStateV3* state,
                     ffi::RemainingArgs inputs, ffi::RemainingRets outputs,
                     absl::string_view, absl::string_view) {
  absl::StatusOr<std::shared_ptr<ModuleAndFunction>> module =
      state->GetModule();
  if (!module.ok()) return module.status();

  return ExecuteFunction((*module)->functions(), stream, (*module)->function(),
                         inputs, outputs);
}

XLA_FFI_DEFINE_HANDLER(kInstantiate, Instantiate,
                       ffi::Ffi::BindInstantiate()
                           .Attr<absl::string_view>("module")
                           .Attr<absl::string_view>("key"));

XLA_FFI_DEFINE_HANDLER(kPrepare, Prepare,
                       ffi::Ffi::BindPrepare()
                           .Ctx<ffi::State<CutlassCallStateV3>>()
                           .RemainingArgs()
                           .RemainingRets()
                           .Attr<absl::string_view>("module")
                           .Attr<absl::string_view>("key"));

XLA_FFI_DEFINE_HANDLER(kInitialize, Initialize, ffi::Ffi::BindInitialize());

XLA_FFI_DEFINE_HANDLER(kExecute, Execute,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<void*>>()
                           .Ctx<ffi::State<CutlassCallStateV3>>()
                           .RemainingArgs()
                           .RemainingRets()
                           .Attr<absl::string_view>("module")
                           .Attr<absl::string_view>("key"),
                       {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER(kExecuteNoCudaGraph, Execute,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<void*>>()
                           .Ctx<ffi::State<CutlassCallStateV3>>()
                           .RemainingArgs()
                           .RemainingRets()
                           .Attr<absl::string_view>("module")
                           .Attr<absl::string_view>("key"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kCutlassCallTarget.data(), "CUDA",
                         (XLA_FFI_Handler_Bundle{/*instantiate=*/kInstantiate,
                                                 /*prepare=*/kPrepare,
                                                 /*initialize=*/kInitialize,
                                                 /*execute=*/kExecute}));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         kCutlassCallNoCudaGraphTarget.data(), "CUDA",
                         (XLA_FFI_Handler_Bundle{
                             /*instantiate=*/kInstantiate,
                             /*prepare=*/kPrepare,
                             /*initialize=*/kInitialize,
                             /*execute=*/kExecuteNoCudaGraph}));

}  // namespace
}  // namespace xla::gpu::cutedsl
