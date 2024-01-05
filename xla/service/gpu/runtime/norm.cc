/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/runtime/norm.h"

#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/Sequence.h"
#include "xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "xla/runtime/custom_call.h"
#include "xla/runtime/executable.h"
#include "xla/service/gpu/gpu_asm_opts_util.h"
#include "xla/service/gpu/runtime/support.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/translate/mhlo_to_hlo/attribute_exporter.h"
#include "xla/xla.pb.h"

namespace xla {

using xla::runtime::CustomCall;
using xla::runtime::EnumAttrEncoding;
using xla::runtime::FlatMemrefView;
using xla::runtime::State;
using xla::runtime::StridedMemrefView;
using xla::runtime::Tagged;

namespace lmhlo_gpu = ::mlir::lmhlo_gpu;
namespace gpu {

struct NormAlgorithmConfig {
  int64_t algorithm;
  int64_t workspace_size;
};

static CudnnNormKind EncodeNormKind(lmhlo_gpu::CudnnNormKind signature) {
  switch (signature) {
    case lmhlo_gpu::CudnnNormKind::LayerFwdInfer:
      return CudnnNormKind::kLayerForwardInfer;
    case lmhlo_gpu::CudnnNormKind::LayerFwdTrain:
      return CudnnNormKind::kLayerForwardTrain;
    case lmhlo_gpu::CudnnNormKind::LayerBwd:
      return CudnnNormKind::kLayerBackward;
  }
}

void PopulateNormKindAttrEncoding(
    runtime::CustomCallAttrEncodingSet& encoding) {
  {  // --- Encode `lmhlo_gpu::CudnnNormKindAttr`.
    encoding.Add<
        EnumAttrEncoding<lmhlo_gpu::CudnnNormKindAttr, lmhlo_gpu::CudnnNormKind,
                         xla::gpu::CudnnNormKind>>(EncodeNormKind);
  }
}

void PopulateNormAlgorithmConfigAttrEncoding(
    runtime::CustomCallAttrEncodingSet& encoding) {
  {  // --- Encode `lmhlo_gpu::NormAlgorithmConfigAttr`.
    using Attr = mlir::lmhlo_gpu::NormAlgorithmConfigAttr;
    encoding
        .Add<xla::runtime::AggregateAttrEncoding<Attr, NormAlgorithmConfig>>(
            encoding, xla::runtime::AggregateAttrDef<Attr>()
                          .Add("algorithm", &Attr::getAlgorithm)
                          .Add("workspace_size", &Attr::getWorkspaceSize));
  }
}
}  // namespace gpu

namespace runtime {
XLA_RUNTIME_REGISTER_ENUM_ATTR_DECODING(xla::gpu::CudnnNormKind);

XLA_RUNTIME_REGISTER_AGGREGATE_ATTR_DECODING(
    xla::gpu::NormAlgorithmConfig,  //
    AggregateMember<int64_t>("algorithm"),
    AggregateMember<int64_t>("workspace_size"));
}  // namespace runtime

namespace gpu {

void RegisterNormTypeIdNames(runtime::TypeIDNameRegistry& registry) {
  registry.Register<Tagged<NormAlgorithmConfig>>(
      "__type_id_norm_algorithm_config");
  registry.Register<Tagged<CudnnNormKind>>("__type_id_cudnn_norm_kind");
}

static GpuNormDescriptor GetGpuNormDescriptor(
    StridedMemrefView x, StridedMemrefView scale, StridedMemrefView y_or_dx,
    std::optional<StridedMemrefView> bias, std::optional<StridedMemrefView> dy,
    std::optional<StridedMemrefView> expectation,
    std::optional<StridedMemrefView> norm_factor,
    std::optional<StridedMemrefView> dscale,
    std::optional<StridedMemrefView> dbias, double epsilon,
    NormAlgorithmConfig algorithm_config,
    absl::Span<const int64_t> operand_layouts, CudnnNormKind kind) {
  GpuNormDescriptor descriptor;

  auto* algorithm = descriptor.backend_config.mutable_algorithm();
  algorithm->set_algo_id(algorithm_config.algorithm);
  algorithm->set_is_cudnn_frontend(true);
  if (algorithm_config.workspace_size >= 0) {
    algorithm->mutable_workspace_size()->set_value(
        algorithm_config.workspace_size);
  }
  descriptor.kind = kind;

  // Apply backend config layout to the shape.
  int layout_idx = 0;
  auto apply_shape = [&operand_layouts,
                      &layout_idx](const StridedMemrefView& memref) -> Shape {
    std::vector<int64_t> minor_to_major = {
        operand_layouts.begin() + layout_idx,
        operand_layouts.begin() + layout_idx + memref.sizes.size()};
    layout_idx += memref.sizes.size();
    Shape shape = ToShape(memref);
    return ShapeUtil::MakeShapeWithDenseLayout(
        shape.element_type(), shape.dimensions(), minor_to_major);
  };

  descriptor.x_shape = apply_shape(x);
  descriptor.scale_shape = apply_shape(scale);
  descriptor.y_or_dx_shape = apply_shape(y_or_dx);
  if (bias) {
    descriptor.bias_shape = apply_shape(bias.value());
  }
  if (dy) {
    descriptor.dy_shape = apply_shape(dy.value());
  }
  if (expectation) {
    descriptor.expectation_shape = apply_shape(expectation.value());
  }
  if (norm_factor) {
    descriptor.norm_factor_shape = apply_shape(norm_factor.value());
  }
  if (dscale) {
    descriptor.dscale_shape = apply_shape(dscale.value());
  }
  if (dbias) {
    descriptor.dbias_shape = apply_shape(dbias.value());
  }

  descriptor.backend_config.set_epsilon(epsilon);

  return descriptor;
}

static absl::Status NormImpl(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, State<NormRunnerState> runner_state,
    StridedMemrefView x, StridedMemrefView scale, StridedMemrefView y_or_dx,
    CustomCall::RemainingArgs remaining_args, int64_t uid, double epsilon,
    absl::Span<const int64_t> operand_layouts,
    NormAlgorithmConfig algorithm_config, CudnnNormKind kind) {
  std::optional<StridedMemrefView> bias, expectation, norm_factor, dy, dscale,
      dbias;
  // Final remaining arg is the scratch space.
  if (kind == CudnnNormKind::kLayerForwardInfer ||
      kind == CudnnNormKind::kLayerForwardTrain) {
    auto bias_ = remaining_args.get<StridedMemrefView>(0);
    if (failed(bias_)) {
      return absl::InternalError("Failure while retrieving bias.");
    }
    bias = bias_.value();
  }
  if (kind == CudnnNormKind::kLayerForwardTrain) {
    auto expectation_ = remaining_args.get<StridedMemrefView>(1);
    if (failed(expectation_)) {
      return absl::InternalError("Failure while retrieving expectation.");
    }
    expectation = expectation_.value();

    auto norm_factor_ = remaining_args.get<StridedMemrefView>(2);
    if (failed(norm_factor_)) {
      return absl::InternalError("Failure while retrieving norm factor.");
    }
    norm_factor = norm_factor_.value();
  }
  if (kind == CudnnNormKind::kLayerBackward) {
    auto dy_ = remaining_args.get<StridedMemrefView>(0);
    if (failed(dy_)) {
      return absl::InternalError("Failure while retrieving dy.");
    }
    dy = dy_.value();

    auto expectation_ = remaining_args.get<StridedMemrefView>(1);
    if (failed(expectation_)) {
      return absl::InternalError("Failure while retrieving expectation.");
    }
    expectation = expectation_.value();

    auto norm_factor_ = remaining_args.get<StridedMemrefView>(2);
    if (failed(norm_factor_)) {
      return absl::InternalError("Failure while retrieving norm factor.");
    }
    norm_factor = norm_factor_.value();

    auto dscale_ = remaining_args.get<StridedMemrefView>(3);
    if (failed(dscale_)) {
      return absl::InternalError("Failure while retrieving dscale.");
    }
    dscale = dscale_.value();

    auto dbias_ = remaining_args.get<StridedMemrefView>(4);
    if (failed(dbias_)) {
      return absl::InternalError("Failure while retrieving dbias.");
    }
    dbias = dbias_.value();
  }

  GpuNormDescriptor descriptor = GetGpuNormDescriptor(
      x, scale, y_or_dx, bias, dy, expectation, norm_factor, dscale, dbias,
      epsilon, algorithm_config, operand_layouts, kind);

  auto config = GpuNormConfig::For(descriptor);
  if (!config.ok()) {
    return tsl::ToAbslStatus(config.status());
  }
  auto current_runner =
      runner_state.GetOrCreate([&config]() -> absl::StatusOr<NormRunnerState> {
        return NormRunnerState(std::move(config.value()));
      });
  if (!current_runner.ok()) {
    return tsl::ToAbslStatus(current_runner.status());
  }

  se::DeviceMemoryBase x_buffer = GetDeviceAddress(x);
  se::DeviceMemoryBase scale_buffer = GetDeviceAddress(scale);
  se::DeviceMemoryBase y_or_dx_buffer = GetDeviceAddress(y_or_dx);
  std::optional<se::DeviceMemoryBase> bias_buffer, dy_buffer,
      expectation_buffer, norm_factor_buffer, dscale_buffer, dbias_buffer;
  if (bias) {
    bias_buffer = GetDeviceAddress(bias.value());
  }
  if (dy) {
    dy_buffer = GetDeviceAddress(dy.value());
  }
  if (expectation) {
    expectation_buffer = GetDeviceAddress(expectation.value());
  }
  if (norm_factor) {
    norm_factor_buffer = GetDeviceAddress(norm_factor.value());
  }
  if (dscale) {
    dscale_buffer = GetDeviceAddress(dscale.value());
  }
  if (dbias) {
    dbias_buffer = GetDeviceAddress(dbias.value());
  }

  auto scratch = remaining_args.get<FlatMemrefView>(remaining_args.size() - 1);
  if (failed(scratch)) {
    return absl::InternalError("Failure while retrieving scratch.");
  }
  se::DeviceMemoryBase scratch_buffer = GetDeviceAddress(scratch.value());

  RunNormOptions opts;
  opts.norm_runner = &current_runner.value()->runner;

  // Run the norm.
  return RunGpuNorm(current_runner.value()->config, x_buffer, scale_buffer,
                    y_or_dx_buffer, bias_buffer, dy_buffer, expectation_buffer,
                    norm_factor_buffer, dscale_buffer, dbias_buffer,
                    scratch_buffer, run_options->stream(), opts);
  return OkStatus();
}

template <typename... Ts>
auto BindNormAttributes(runtime::CustomCallBinding<Ts...> binding) {
  return std::move(binding)
      // Unique convolution id for caching state.
      .template Attr<int64_t>("uid")
      .template Attr<double>("epsilon")
      .template Attr<absl::Span<const int64_t>>("operand_layouts")
      .template Attr<NormAlgorithmConfig>("norm_algorithm_config")
      .template Attr<CudnnNormKind>("kind");
}

auto NormCall(const char* name) {
  return CustomCall::Bind(name)
      .UserData<const ServiceExecutableRunOptions*>()
      .UserData<const DebugOptions*>()
      .State<NormRunnerState>("uid")
      .Arg<StridedMemrefView>()   // x
      .Arg<StridedMemrefView>()   // scale
      .Arg<StridedMemrefView>();  // y_or_dx
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    Norm, FunctionWrapper<NormImpl>(), checks,
    BindNormAttributes(NormCall("xla.gpu.norm").RemainingArgs()));

void RegisterNormCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.norm", Norm);
}

StreamExecutorNormRunners* NormRunnerStates::operator()(
    se::StreamExecutor* executor) {
  absl::MutexLock lock(&mutex_);
  return &runners_[executor];
}

}  // namespace gpu
}  // namespace xla
