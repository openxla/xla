/* Copyright 2025 The OpenXLA Authors.

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

#if defined(INTEL_MKL)
#include "xla/backends/cpu/runtime/onednn/onednn_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi_api.h"
#include "xla/primitive_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/util.h"

namespace xla::cpu {

using AttributesMap = ffi::CallFrameBuilder::AttributesMap;

// TODO(intel-tf): Replace this parsing function
// with a more oneDNN-specific attribute parser that can
// parse config protos like OneDnnMatMulConfig
static absl::StatusOr<AttributesMap> ParseAttributes(
    absl::string_view backend_config) {
  AttributesMap attributes;
  if (!backend_config.empty() && backend_config != "{}") {
    // Parse backend config into an MLIR dictionary.
    mlir::MLIRContext mlir_context;
    mlir::Attribute attr = mlir::parseAttribute(backend_config, &mlir_context);
    if (auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr)) {
      // Convert the MLIR dictionary to FFI attributes.
      TF_ASSIGN_OR_RETURN(attributes, xla::ffi::BuildAttributesMap(dict));
    } else {
      return Internal(
          "Unsupported backend config. Expected a string parsable into "
          "dictionary attribute");
    }

    VLOG(3) << absl::StreamFormat("  attributes: %s", backend_config);
  }

  return attributes;
}

// Call `instantiate` callback if passed. This function needs its own copy of
// attributes, that's what AttributesBuilder expects, there's no way around it.
static absl::Status InstantiateHandlerState(
    ffi::HandlerRegistration& handler, ffi::ExecutionState* execution_state,
    AttributesMap attributes) {
  // Initialize FFI handler state if it has an instantiate callback.
  if (handler.bundle.instantiate) {
    // At FFI handler instantiation time, we don't have any arguments or results
    ffi::CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);

    ffi::CallFrameBuilder::AttributesBuilder attrs;
    attrs.Append(std::move(attributes));

    builder.AddAttributes(attrs.Build());
    ffi::CallFrame instantiate_call_frame = builder.Build();

    ffi::CallOptions options;
    options.execution_state = execution_state;
    TF_RETURN_IF_ERROR(Call(handler.bundle.instantiate, instantiate_call_frame,
                            options, XLA_FFI_ExecutionStage_INSTANTIATE));
  }

  return absl::OkStatus();
}

// Builds a call frame prototype for typed-FFI oneDNN thunk calls with dummy
// device memory addresses. This is called once when creating the OneDnn thunk,
// then the thunk will need to update the addresses at runtime.
static absl::StatusOr<ffi::CallFrame> BuildCallFrameForTypedFFI(
    const CustomCallApiVersion version, const OpBuffers& op_buffers,
    const absl::string_view backend_config, AttributesMap attributes) {
  ffi::CallFrameBuilder builder(
      /*num_args=*/op_buffers.arguments_buffers.size(),
      /*num_rets=*/op_buffers.results_buffers.size());

  // Add prototype input buffers with actual data types and shapes. Device
  // memory addresses will be updated at runtime.
  for (int i = 0; i < op_buffers.arguments_buffers.size(); ++i) {
    auto& shape = op_buffers.arguments_shapes[i];

    if (shape.IsToken()) {
      builder.AddTokenArg();
      continue;
    }

    auto elements = absl::c_accumulate(shape.dimensions(), 1ULL,
                                       std::multiplies<int64_t>());
    auto dtype_bytes = primitive_util::ByteWidth(shape.element_type());
    se::DeviceMemoryBase placeholder_arg(nullptr, elements * dtype_bytes);
    builder.AddBufferArg(placeholder_arg, shape.element_type(),
                         shape.dimensions());
  }

  // Add prototype output buffers with actual data types and shapes. Device
  // memory addresses will be updated at runtime.
  for (int i = 0; i < op_buffers.results_buffers.size(); ++i) {
    auto& shape = op_buffers.results_shapes[i];

    if (shape.IsToken()) {
      builder.AddTokenRet();
      continue;
    }

    auto elements = absl::c_accumulate(shape.dimensions(), 1ULL,
                                       std::multiplies<int64_t>());
    auto dtype_bytes = primitive_util::ByteWidth(shape.element_type());
    se::DeviceMemoryBase placeholder_ret(nullptr, elements * dtype_bytes);
    builder.AddBufferRet(placeholder_ret, shape.element_type(),
                         shape.dimensions());
  }

  // Add attributes if any.
  if (!attributes.empty()) {
    ffi::CallFrameBuilder::AttributesBuilder attrs;
    attrs.Append(std::move(attributes));
    builder.AddAttributes(attrs.Build());
  }

  return builder.Build();
}

absl::StatusOr<std::unique_ptr<OneDnnThunk>> OneDnnThunk::Create(
    Info info, absl::string_view target_name, OpBuffers op_buffers,
    absl::string_view backend_config, CustomCallApiVersion api_version) {
  std::optional<ffi::CallFrame> call_frame;
  auto execution_state = std::make_unique<ffi::ExecutionState>();

  // Resolve oneDNN custom call thunk target name to the target callable.
  std::variant<CustomCallTarget, ffi::HandlerRegistration> target;

  TF_ASSIGN_OR_RETURN(target, ffi::FindHandler(target_name, "Host"));

  AttributesMap attributes;

  TF_ASSIGN_OR_RETURN(attributes, ParseAttributes(backend_config));

  TF_RETURN_IF_ERROR(InstantiateHandlerState(
      std::get<1>(target), execution_state.get(), attributes));

  TF_ASSIGN_OR_RETURN(call_frame, BuildCallFrameForTypedFFI(
                                      api_version, op_buffers, backend_config,
                                      std::move(attributes)));

  return absl::WrapUnique(new OneDnnThunk(
      std::move(info), target_name, std::move(target), std::move(op_buffers),
      api_version, std::move(backend_config), std::move(call_frame),
      std::move(execution_state)));
}

OneDnnThunk::OneDnnRuntime::OneDnnRuntime(
    Eigen::ThreadPoolInterface* thread_pool)
    : threadpool(
          std::make_unique<OneDnnThreadPool>(thread_pool, /*is_async=*/true)),
      cpu_engine(dnnl::engine::kind::cpu, 0),
      onednn_stream(
          dnnl::threadpool_interop::make_stream(cpu_engine, threadpool.get())),
      resources() {}

OneDnnThunk::OneDnnThunk(
    Info info, absl::string_view target_name,
    std::variant<CustomCallTarget, ffi::HandlerRegistration> target,
    OpBuffers op_buffers, CustomCallApiVersion api_version,
    absl::string_view backend_config, std::optional<ffi::CallFrame> call_frame,
    std::unique_ptr<ffi::ExecutionState> execution_state)
    : Thunk(Kind::kCustomCall, std::move(info)),
      target_name_(target_name),
      target_(std::move(target)),
      op_buffers_(std::move(op_buffers)),
      api_version_(api_version),
      backend_config_(std::move(backend_config)),
      call_frame_(std::move(call_frame)),
      call_frames_([this] { return call_frame_->Copy(); }),
      execution_state_(std::move(execution_state)) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> OneDnnThunk::Execute(
    const ExecuteParams& params) {
  VLOG(1) << absl::StreamFormat(
      "CustomCall: %s, #arguments=%d, #results=%d", target_name_,
      op_buffers_.arguments_buffers.size(), op_buffers_.results_buffers.size());

  if (api_version_ != CustomCallApiVersion::API_VERSION_TYPED_FFI) {
    return Internal(
        "OneDnnThunk only supports typed-FFI custom calls, "
        "got API version: %s",
        CustomCallApiVersion_Name(api_version_));
  }

  if (params.custom_call_params == nullptr) {
    return Internal("CustomCallExecuteParams cannot be nullptr.");
  }

  // Forward ExecutableRunOptions to the FFI handlers via the call options.
  CustomCallExecuteParams* custom_call_params = params.custom_call_params;

  Eigen::ThreadPoolInterface* thread_pool =
      custom_call_params->intra_op_thread_pool->getPool();

  auto runtime = std::make_unique<OneDnnRuntime>(thread_pool);

  // Collect argument buffers.
  absl::InlinedVector<se::DeviceMemoryBase, 8> arguments;
  arguments.reserve(op_buffers_.arguments_buffers.size());
  runtime->resources.arg_memrefs.reserve(op_buffers_.arguments_buffers.size());
  for (int i = 0; i < op_buffers_.arguments_buffers.size(); ++i) {
    BufferAllocation::Slice& slice = op_buffers_.arguments_buffers[i];
    TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase arg,
                        params.buffer_allocations->GetDeviceAddress(slice));
    arguments.emplace_back(arg);
    auto memref =
        CreateMemrefFromShape(op_buffers_.arguments_shapes[i], arg.opaque());
    runtime->resources.arg_memrefs.push_back(std::move(memref));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(arguments[i].opaque(),
                                        arguments[i].size());
    VLOG(3) << absl::StreamFormat(
        "  arg: %s in slice %s (%p)",
        op_buffers_.arguments_shapes[i].ToString(true), slice.ToString(),
        arguments[i].opaque());
  }

  // Collect results buffers.
  absl::InlinedVector<se::DeviceMemoryBase, 4> results;
  results.reserve(op_buffers_.results_buffers.size());
  runtime->resources.result_memrefs.reserve(op_buffers_.results_buffers.size());
  for (int i = 0; i < op_buffers_.results_buffers.size(); ++i) {
    BufferAllocation::Slice& slice = op_buffers_.results_buffers[i];
    TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase res,
                        params.buffer_allocations->GetDeviceAddress(slice));
    results.emplace_back(res);
    auto memref =
        CreateMemrefFromShape(op_buffers_.results_shapes[i], res.opaque());
    runtime->resources.result_memrefs.push_back(std::move(memref));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(results[i].opaque(), results[i].size());
    VLOG(3) << absl::StreamFormat("  res: %s in slice %s (%p)",
                                  op_buffers_.results_shapes[i].ToString(true),
                                  slice.ToString(), results[i].opaque());
  }

  // Borrow the FFI call frame from the object pool and update with the actual
  // device memory addresses.
  TF_ASSIGN_OR_RETURN(auto call_frame, call_frames_.GetOrCreate());
  TF_RETURN_IF_ERROR(call_frame->UpdateWithBuffers(arguments, results));

  // Do a heap allocation of the ExecutionContext, to ensure that
  // the context remains valid until the execution is complete.
  auto owned_context = std::make_unique<ffi::ExecutionContext>();

  // Insert the oneDNN-related objects into the execution context.
  // This allows the FFI handler to safely access the oneDNN engine, stream,
  // and resources until the execution of the custom call is complete.
  auto status = owned_context->Insert(&runtime->cpu_engine);
  if (!status.ok()) {
    return Internal("Failed to add oneDNN engine to the execution context: %s",
                    status.message());
  }
  status = owned_context->Insert(&runtime->onednn_stream);
  if (!status.ok()) {
    return Internal("Failed to add oneDNN stream to the execution context: %s",
                    status.message());
  }
  status = owned_context->Insert(runtime->threadpool.get());
  if (!status.ok()) {
    return Internal("Failed to add oneDNN threadpool to the execution context: %s",
                    status.message());
  }
  status = owned_context->Insert(&runtime->resources);
  if (!status.ok()) {
    return Internal(
        "Failed to add oneDNN resources to the execution context: %s",
        status.message());
  }

  ffi::CallOptions call_options = {
      custom_call_params->run_id,
      custom_call_params->device_ordinal,
      ffi::CallOptions::CpuOptions{custom_call_params->intra_op_thread_pool},
      /*called_computation=*/nullptr,
      owned_context.get(),
      execution_state_.get()};

  ffi::HandlerRegistration& handler = std::get<1>(target_);

  auto executed =
      ffi::CallAsync(handler.bundle.execute, *call_frame, call_options);

  executed.AndThen(
      [runtime = std::move(runtime), owned_context = std::move(owned_context)] {
        // runtime and context_holder will be automatically destroyed here
      });

  return executed;
}

OneDnnThunk::BufferUses OneDnnThunk::buffer_uses() const {
  BufferUses buffer_uses;
  for (const auto& argument : op_buffers_.arguments_buffers) {
    buffer_uses.emplace_back(argument, BufferUse::kRead);
  }
  for (const auto& result : op_buffers_.results_buffers) {
    buffer_uses.emplace_back(result, BufferUse::kWrite);
  }
  return buffer_uses;
}

}  // namespace xla::cpu
#endif  // INTEL_MKL
