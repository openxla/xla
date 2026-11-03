/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/record_ffi.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/record_c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_interop.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/mem.h"

namespace xla::gpu {

namespace {

template <int64_t kAlignment>
class AlignedMemory {
 public:
  explicit AlignedMemory(size_t size_bytes)
      : base_(tsl::port::AlignedMalloc(
            size_bytes, static_cast<std::align_val_t>(kAlignment))),
        size_bytes_(size_bytes) {}
  AlignedMemory(const AlignedMemory&) = delete;
  AlignedMemory& operator=(const AlignedMemory&) = delete;
  AlignedMemory& operator=(AlignedMemory&& other) = delete;

  // Move only type.
  AlignedMemory(AlignedMemory&& other) noexcept
      : base_(std::exchange(other.base_, nullptr)),
        size_bytes_(std::exchange(other.size_bytes_, 0)) {}

  ~AlignedMemory() {
    // Check for nullptr because a moved-from object will have base_ == nullptr.
    if (base_ != nullptr) {
      tsl::port::AlignedSizedFree(base_, size_bytes_,
                                  static_cast<std::align_val_t>(kAlignment));
    }
  }
  void* data() const { return base_; }
  size_t size_bytes() const { return size_bytes_; }

 private:
  void* base_;
  size_t size_bytes_;
};

// Custom implementation of se::KernelArgsPackedArrayBase for XLA:FFI.
class FfiKernelArgsPacked : public se::KernelArgsPackedArrayBase {
  static constexpr int64_t kAlignment = 64;

 public:
  explicit FfiKernelArgsPacked(int64_t num_args) {
    argument_addresses_.reserve(num_args);
  }

  // Arguments passed by pointer to the kernel.
  void AddDeviceAddress(void* opaque) {
    device_ptrs_.push_back(opaque);
    argument_addresses_.push_back(&device_ptrs_.back());
  }

  // Arguments passed by value to the kernel.
  void AddHostAddress(const void* data, int64_t size) {
    CHECK_GT(size, 0) << "Host value size must be positive.";
    AlignedMemory<kAlignment> aligned_memory(size);
    std::memcpy(aligned_memory.data(), data, size);
    allocated_args_.push_back(std::move(aligned_memory));
    argument_addresses_.push_back(allocated_args_.back().data());
  }

  void add_shared_bytes(int64_t bytes) { shared_memory_bytes_ += bytes; }

  absl::Span<const void* const> argument_addresses() const override {
    return argument_addresses_;
  }

  size_t number_of_arguments() const override {
    return argument_addresses_.size() + (shared_memory_bytes_ > 0);
  }

  size_t number_of_shared_bytes() const override {
    return shared_memory_bytes_;
  }

 private:
  // Dequeue to keep addresses stable.
  std::deque<void*> device_ptrs_;
  std::vector<AlignedMemory<kAlignment>> allocated_args_;
  std::vector<const void*> argument_addresses_;
  int64_t shared_memory_bytes_ = 0;
};

absl::Status PackArgs(const XLA_FFI_KernelArgs* args,
                      FfiKernelArgsPacked& packed_args) {
  if (args == nullptr || (args->num_args > 0 && args->args == nullptr)) {
    return absl::InvalidArgumentError("Invalid XLA_FFI_KernelArgs struct");
  }
  for (int64_t i = 0; i < args->num_args; ++i) {
    if (args->args[i].type == XLA_FFI_KernelArgType_HostValue) {
      if (args->args[i].size <= 0) {
        return absl::InvalidArgumentError(
            "Host value args should have size > 0.");
      }
      packed_args.AddHostAddress(args->args[i].arg_address, args->args[i].size);
      continue;
    }
    if (args->args[i].type == XLA_FFI_KernelArgType_DevicePtr) {
      if (args->args[i].size != 0) {
        return absl::InvalidArgumentError(
            "Device pointer args should have size 0.");
      }
      packed_args.AddDeviceAddress(
          // FFI API is const we need to cast away constness.
          // NOLINTNEXTLINE
          const_cast<void*>(args->args[i].arg_address));
      continue;
    }
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unsupported kernel argument type: %d", args->args[i].type));
  }
  return absl::OkStatus();
}

se::CommandBuffer::Command* AsStreamExecutorCommand(
    const XLA_FFI_Command* command) {
  // We encode the se::CommandBuffer::Command* into the XLA_FFI_Command* C
  // object. Here we simply decode it back.
  // NOLINTNEXTLINE
  return reinterpret_cast<se::CommandBuffer::Command*>(
      // FFI API is const we need to cast away constness.
      // NOLINTNEXTLINE
      const_cast<XLA_FFI_Command*>(command));
}

const XLA_FFI_Command* AsXlaFfiCommand(
    const se::CommandBuffer::Command* command) {
  // We encode the se::CommandBuffer::Command* into the opaque XLA_FFI_Command*.
  // NOLINTNEXTLINE
  return reinterpret_cast<XLA_FFI_Command*>(
      // reinterpret_cast does not work with const.
      // NOLINTNEXTLINE
      const_cast<se::CommandBuffer::Command*>(command));
}

bool EqualDims(XLA_FFI_Dim3 a, XLA_FFI_Dim3 b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

bool PositiveDims(XLA_FFI_Dim3 dims) {
  return dims.x > 0 && dims.y > 0 && dims.z > 0;
}

bool Divides(XLA_FFI_Dim3 divisor, XLA_FFI_Dim3 dividend) {
  return dividend.x % divisor.x == 0 && dividend.y % divisor.y == 0 &&
         dividend.z % divisor.z == 0;
}

std::optional<se::ClusterDim> ClusterDims(XLA_FFI_Dim3 dims) {
  if (EqualDims(dims, {0, 0, 0})) {
    return std::nullopt;
  }
  return se::ClusterDim(dims.x, dims.y, dims.z);
}

absl::Status ValidateLaunchConfig(const XLA_FFI_LaunchConfig* config) {
  if (config == nullptr ||
      config->struct_size < XLA_FFI_LaunchConfig_STRUCT_SIZE) {
    return absl::InvalidArgumentError(
        "Missing or truncated launch configuration");
  }
  const auto& dims = config->launch_dims;
  if (!PositiveDims(dims.grid) || !PositiveDims(dims.block)) {
    return absl::InvalidArgumentError(
        "Grid and block dimensions must be positive");
  }
  if (!EqualDims(dims.cluster, {0, 0, 0}) &&
      (!PositiveDims(dims.cluster) || !Divides(dims.cluster, dims.grid))) {
    return absl::InvalidArgumentError(
        "Cluster dimensions must be all zero or positive divisors of the grid");
  }
  if (!EqualDims(config->preferred_cluster, {0, 0, 0}) &&
      (!PositiveDims(config->preferred_cluster) ||
       !PositiveDims(dims.cluster) ||
       config->preferred_cluster.z != dims.cluster.z ||
       !Divides(dims.cluster, config->preferred_cluster) ||
       !Divides(config->preferred_cluster, dims.grid))) {
    return absl::InvalidArgumentError(
        "Preferred cluster x/y dimensions must be positive multiples of "
        "cluster "
        "dimensions and divide the grid; z must match the cluster");
  }
  if ((config->cooperative != 0 && config->cooperative != 1) ||
      (config->uses_pdl != 0 && config->uses_pdl != 1)) {
    return absl::InvalidArgumentError(
        "Cooperative and PDL flags must be 0 or 1");
  }
  return absl::OkStatus();
}

struct FfiLaunchParams {
  XLA_FFI_LaunchConfig config;
  se::NativeKernel kernel;
};

class FfiKernelCache : public se::CommandBuffer::Resource {
 public:
  void SaveLaunchParams(const se::CommandBuffer::Command* cmd,
                        FfiLaunchParams params) {
    launch_params_[cmd] = std::move(params);
  }

  const FfiLaunchParams* GetLaunchParams(
      const se::CommandBuffer::Command* cmd) const {
    auto it = launch_params_.find(cmd);
    if (it != launch_params_.end()) {
      return &it->second;
    }
    return nullptr;
  }

  FfiLaunchParams* GetMutableLaunchParams(
      const se::CommandBuffer::Command* cmd) {
    auto it = launch_params_.find(cmd);
    if (it != launch_params_.end()) {
      return &it->second;
    }
    return nullptr;
  }

 private:
  absl::flat_hash_map<const se::CommandBuffer::Command*, FfiLaunchParams>
      launch_params_;
};

XLA_FFI_Error* FfiCreateLaunchWithConfig(
    XLA_FFI_RecordContext* ctx, const char* kernel_name,
    const void* kernel_data, int64_t kernel_size, XLA_FFI_SourceFormat format,
    const XLA_FFI_LaunchConfig* config, const XLA_FFI_KernelArgs* args,
    const XLA_FFI_Command* const* dependencies, uint32_t num_dependencies,
    const XLA_FFI_Command** out_command) {
  if (absl::Status status = ValidateLaunchConfig(config); !status.ok()) {
    return ffi::CreateError(status);
  }
  if (format != XLA_FFI_SourceFormat_FUNCTION_PTR) {
    return ffi::CreateError(absl::InvalidArgumentError(
        absl::StrFormat("Unsupported XLA_FFI_SourceFormat: %d", format)));
  }
  if (kernel_data == nullptr) {
    return ffi::CreateError(absl::InvalidArgumentError(
        "kernel_data cannot be null for XLA_FFI_SourceFormat_FUNCTION_PTR"));
  }

  FfiLaunchParams params{
      *config,
      // Native driver function handles are non-const opaque pointers.
      {const_cast<void*>(kernel_data), std::string(kernel_name),
       ctx->pdl_enabled && config->uses_pdl != 0, config->cooperative != 0,
       ClusterDims(config->preferred_cluster)}};
  if (args == nullptr || args->num_args < 0) {
    return ffi::CreateError(
        absl::InvalidArgumentError("Invalid XLA_FFI_KernelArgs struct"));
  }
  FfiKernelArgsPacked packed_args(args->num_args);
  packed_args.add_shared_bytes(config->shared_mem_bytes);
  if (absl::Status status = PackArgs(args, packed_args); !status.ok()) {
    return ffi::CreateError(status);
  }

  std::vector<const se::CommandBuffer::Command*> deps;
  deps.reserve(num_dependencies + ctx->dependencies.size());
  deps.insert(deps.end(), ctx->dependencies.begin(), ctx->dependencies.end());
  for (uint32_t i = 0; i < num_dependencies; ++i) {
    deps.push_back(AsStreamExecutorCommand(dependencies[i]));
  }

  const auto& dims = config->launch_dims;
  auto command = ctx->command_buffer->CreateLaunch(
      se::ThreadDim(dims.block.x, dims.block.y, dims.block.z),
      se::BlockDim(dims.grid.x, dims.grid.y, dims.grid.z),
      ClusterDims(dims.cluster), params.kernel, packed_args, deps);
  if (!command.ok()) {
    return ffi::CreateError(command.status());
  }
  ctx->command_buffer->GetOrConstructResource<FfiKernelCache>()
      ->SaveLaunchParams(*command, std::move(params));
  *out_command = AsXlaFfiCommand(*command);
  return nullptr;
}

XLA_FFI_Error* FfiCreateLaunch(
    XLA_FFI_RecordContext* ctx, const char* kernel_name,
    const void* kernel_data, int64_t kernel_size, XLA_FFI_SourceFormat format,
    XLA_FFI_LaunchDims launch_dims, uint32_t shared_mem_bytes, int32_t use_pdl,
    const XLA_FFI_KernelArgs* args, const XLA_FFI_Command* const* dependencies,
    uint32_t num_dependencies, const XLA_FFI_Command** out_command) {
  XLA_FFI_LaunchConfig config{XLA_FFI_LaunchConfig_STRUCT_SIZE,
                              launch_dims,
                              shared_mem_bytes,
                              use_pdl != 0,
                              0,
                              {0, 0, 0}};
  return FfiCreateLaunchWithConfig(ctx, kernel_name, kernel_data, kernel_size,
                                   format, &config, args, dependencies,
                                   num_dependencies, out_command);
}

XLA_FFI_Error* FfiUpdateLaunchWithConfig(XLA_FFI_RecordContext* ctx,
                                         const XLA_FFI_Command* command,
                                         const XLA_FFI_LaunchConfig* config,
                                         const XLA_FFI_KernelArgs* args) {
  if (absl::Status status = ValidateLaunchConfig(config); !status.ok()) {
    return ffi::CreateError(status);
  }
  const se::CommandBuffer::Command* cmd = AsStreamExecutorCommand(command);
  auto* cache = ctx->command_buffer->GetOrConstructResource<FfiKernelCache>();
  FfiLaunchParams* params = cache->GetMutableLaunchParams(cmd);
  if (params == nullptr) {
    return ffi::CreateError(absl::InvalidArgumentError(
        "Failed to find cached launch parameters for update"));
  }
  const auto& original = params->config;
  if (!EqualDims(config->launch_dims.cluster, original.launch_dims.cluster) ||
      !EqualDims(config->preferred_cluster, original.preferred_cluster) ||
      config->cooperative != original.cooperative ||
      config->uses_pdl != original.uses_pdl) {
    return ffi::CreateError(absl::InvalidArgumentError(
        "Cluster dimensions, preferred cluster dimensions, cooperative mode, "
        "and PDL cannot change during a launch update"));
  }

  if (args == nullptr || args->num_args < 0) {
    return ffi::CreateError(
        absl::InvalidArgumentError("Invalid XLA_FFI_KernelArgs struct"));
  }
  FfiKernelArgsPacked packed_args(args->num_args);
  packed_args.add_shared_bytes(config->shared_mem_bytes);
  if (absl::Status status = PackArgs(args, packed_args); !status.ok()) {
    return ffi::CreateError(status);
  }
  const auto& dims = config->launch_dims;
  absl::Status status = ctx->command_buffer->UpdateLaunch(
      cmd, se::ThreadDim(dims.block.x, dims.block.y, dims.block.z),
      se::BlockDim(dims.grid.x, dims.grid.y, dims.grid.z),
      ClusterDims(dims.cluster), params->kernel, packed_args);
  if (!status.ok()) {
    return ffi::CreateError(status);
  }
  params->config = *config;
  return nullptr;
}

XLA_FFI_Error* FfiUpdateLaunch(XLA_FFI_RecordContext* ctx,
                               const XLA_FFI_Command* command,
                               const XLA_FFI_KernelArgs* args) {
  auto* cache = ctx->command_buffer->GetOrConstructResource<FfiKernelCache>();
  const FfiLaunchParams* params =
      cache->GetLaunchParams(AsStreamExecutorCommand(command));
  if (params == nullptr) {
    return ffi::CreateError(absl::InternalError(
        "Failed to find cached launch parameters for update"));
  }
  return FfiUpdateLaunchWithConfig(ctx, command, &params->config, args);
}

XLA_FFI_Error* FfiCreateMemcpyD2D(XLA_FFI_RecordContext* ctx, void* dst,
                                  void* src, int64_t size,
                                  const XLA_FFI_Command* const* dependencies,
                                  uint32_t num_dependencies,
                                  const XLA_FFI_Command** out_command) {
  se::CommandBuffer* cmd_buffer = ctx->command_buffer;

  std::vector<const se::CommandBuffer::Command*> deps;
  deps.reserve(num_dependencies);
  for (uint32_t i = 0; i < num_dependencies; ++i) {
    deps.push_back(
        reinterpret_cast<const se::CommandBuffer::Command*>(dependencies[i]));
  }

  se::DeviceAddressBase dst_addr(dst, size);
  se::DeviceAddressBase src_addr(src, size);
  auto status_or_cmd =
      cmd_buffer->CreateMemcpyD2D(&dst_addr, src_addr, size, deps);

  if (!status_or_cmd.ok()) {
    return xla::ffi::CreateError(status_or_cmd.status());
  }

  *out_command = reinterpret_cast<const XLA_FFI_Command*>(*status_or_cmd);
  return nullptr;
}

XLA_FFI_Error* FfiUpdateMemcpyD2D(XLA_FFI_RecordContext* ctx,
                                  const XLA_FFI_Command* command, void* dst,
                                  void* src, int64_t size) {
  se::CommandBuffer* cmd_buffer = ctx->command_buffer;
  auto* cmd = reinterpret_cast<const se::CommandBuffer::Command*>(command);

  se::DeviceAddressBase dst_addr(dst, size);
  se::DeviceAddressBase src_addr(src, size);
  absl::Status status =
      cmd_buffer->UpdateMemcpyD2D(cmd, &dst_addr, src_addr, size);
  if (!status.ok()) {
    return xla::ffi::CreateError(status);
  }
  return nullptr;
}

XLA_FFI_Error* FfiRequestStreamCapture(XLA_FFI_RecordContext* ctx) {
  ctx->stream_capture_requested = true;
  return nullptr;
}

XLA_FFI_Error* FfiCreateEmptyCommand(XLA_FFI_RecordContext* ctx,
                                     const XLA_FFI_Command* const* dependencies,
                                     uint32_t num_dependencies,
                                     const XLA_FFI_Command** out_command) {
  se::CommandBuffer* cmd_buffer = ctx->command_buffer;
  std::vector<const se::CommandBuffer::Command*> deps;
  deps.reserve(num_dependencies);
  for (uint32_t i = 0; i < num_dependencies; ++i) {
    deps.push_back(AsStreamExecutorCommand(dependencies[i]));
  }
  auto status_or_cmd = cmd_buffer->CreateEmptyCmd(deps);
  if (!status_or_cmd.ok()) {
    return xla::ffi::CreateError(status_or_cmd.status());
  }
  *out_command = AsXlaFfiCommand(*status_or_cmd);
  return nullptr;
}

}  // namespace

const XLA_FFI_RecordApi* GetXlaFfiRecordApi() {
  static constexpr XLA_FFI_RecordApi ffi_record_api = {
      /*.create_launch =*/FfiCreateLaunch,
      /*.update_launch =*/FfiUpdateLaunch,
      /*.create_memcpy_d2d =*/FfiCreateMemcpyD2D,
      /*.update_memcpy_d2d =*/FfiUpdateMemcpyD2D,
      /*.request_stream_capture =*/FfiRequestStreamCapture,
      /*.create_empty_command =*/FfiCreateEmptyCommand,
      /*.create_launch_with_config =*/FfiCreateLaunchWithConfig,
      /*.update_launch_with_config =*/FfiUpdateLaunchWithConfig,
  };
  return &ffi_record_api;
}

}  // namespace xla::gpu
