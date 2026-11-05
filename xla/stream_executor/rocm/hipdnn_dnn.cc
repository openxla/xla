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

// hipDNN-backed implementations of the graph-era dnn::DnnSupport entry points.
//
// MIOpen serves the legacy convolution, batchnorm, pooling and RNN paths out of
// rocm_dnn.cc. The methods here cover the graph API that MIOpen has no analogue
// for, so they live in a separate translation unit even though they are members
// of MIOpenSupport: only this file includes the hipDNN frontend headers.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

// glibc's <sys/sysmacros.h>, pulled in transitively, defines major() and
// minor() as function-like macros. hipDNN's VersionUtils.hpp has members by
// those names, so the macros have to go before its headers are parsed.
#undef major
#undef minor

#include "hipdnn_frontend.hpp"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/engine_options.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/rocm/hipdnn_frontend_helpers.h"
#include "xla/stream_executor/rocm/rocm_dnn.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace gpu {

namespace fe = hipdnn_frontend;

namespace {

// RAII wrapper giving exclusive, context-activated access to the hipDNN handle
// for the duration of a single call sequence. Mirrors MIOpenHandle.
class HipdnnHandle {
 public:
  HipdnnHandle(StreamExecutor* executor, std::unique_ptr<absl::MutexLock> lock,
               hipdnnHandle_t handle)
      : context_(executor->Activate()),
        lock_(std::move(lock)),
        handle_(handle) {}

  hipdnnHandle_t handle() const { return handle_; }

 private:
  std::unique_ptr<ActivateContext> context_;
  std::unique_ptr<absl::MutexLock> lock_;
  hipdnnHandle_t handle_;  // Not owned.
};

// Owns one hipDNN handle per StreamExecutor. hipDNN calls sharing a handle must
// be serialized across threads, which GetHandle() enforces.
class HipdnnAccess {
 public:
  explicit HipdnnAccess(hipdnnHandle_t handle) : handle_(handle) {}

  ~HipdnnAccess() {
    absl::MutexLock lock(&mutex_);
    hipdnnDestroy(handle_);
  }

  HipdnnHandle GetHandle(StreamExecutor* executor, Stream* stream) {
    auto lock = std::make_unique<absl::MutexLock>(&mutex_);
    mutex_.AssertHeld();
    hipStream_t hip_stream =
        stream
            ? static_cast<hipStream_t>(stream->platform_specific_handle().stream)
            : nullptr;
    CHECK_EQ(hipdnnSetStream(handle_, hip_stream), HIPDNN_STATUS_SUCCESS)
        << "Failed to set hipDNN stream.";
    return HipdnnHandle(executor, std::move(lock), handle_);
  }

 private:
  absl::Mutex mutex_;
  hipdnnHandle_t handle_ ABSL_GUARDED_BY(mutex_);  // Owned.
};

// hipDNN resolves its engine plugins relative to libhipdnn_backend.so, which is
// correct under Bazel runfiles and wheel installs alike. The environment
// variable is an escape hatch for out-of-tree plugin builds.
void MaybeSetEnginePluginPaths() {
  const char* plugin_path = std::getenv("XLA_HIPDNN_PLUGIN_PATH");
  if (plugin_path == nullptr) {
    return;
  }
  std::vector<std::string> paths = {plugin_path};
  fe::Error error =
      fe::setEnginePluginPaths(paths, fe::PluginLoadingMode::MODE_ADDITIVE);
  if (error.is_bad()) {
    LOG(WARNING) << "Failed to add hipDNN engine plugin path '" << plugin_path
                 << "': " << error.get_message();
  }
}

absl::Mutex& HipdnnRegistryMutex() {
  static absl::Mutex* mutex = new absl::Mutex;
  return *mutex;
}

// Returns the hipDNN handle for `parent`, creating it on first use. hipDNN is
// optional at runtime: a missing or broken installation surfaces here as an
// error status rather than taking down MIOpen-only execution.
absl::StatusOr<HipdnnAccess*> GetHipdnnAccess(StreamExecutor* parent) {
  static auto* registry =
      new absl::flat_hash_map<StreamExecutor*, std::unique_ptr<HipdnnAccess>>;

  absl::MutexLock lock(&HipdnnRegistryMutex());
  auto it = registry->find(parent);
  if (it != registry->end()) {
    return it->second.get();
  }

  static const bool plugin_paths_set = [] {
    MaybeSetEnginePluginPaths();
    return true;
  }();
  (void)plugin_paths_set;

  std::unique_ptr<ActivateContext> context = parent->Activate();
  hipdnnHandle_t handle = nullptr;
  hipdnnStatus_t status = hipdnnCreate(&handle);
  if (status != HIPDNN_STATUS_SUCCESS) {
    return absl::InternalError(absl::StrCat(
        "Could not create a hipDNN handle: ", hipdnnGetErrorString(status)));
  }
  VLOG(1) << "Initialized hipDNN " << HipdnnVersionString();
  auto access = std::make_unique<HipdnnAccess>(handle);
  HipdnnAccess* result = access.get();
  registry->emplace(parent, std::move(access));
  return result;
}

absl::Status PopulateProfileFromTimer(EventBasedTimer* timer,
                                      const dnn::AlgorithmDesc& algorithm,
                                      dnn::ProfileResult* profile_result,
                                      std::optional<uint64_t> scratch_size) {
  if (profile_result == nullptr) {
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(absl::Duration duration, timer->GetElapsedDuration());
  profile_result->set_algorithm(algorithm);
  profile_result->set_elapsed_time_in_ms(absl::ToDoubleMilliseconds(duration));
  if (scratch_size.has_value()) {
    profile_result->set_scratch_size(*scratch_size);
  }
  return absl::OkStatus();
}

absl::StatusOr<fe::DataType> ToHipdnnDataType(dnn::DataType data_type) {
  switch (data_type) {
    case dnn::DataType::kFloat:
      return fe::DataType::FLOAT;
    case dnn::DataType::kDouble:
      return fe::DataType::DOUBLE;
    case dnn::DataType::kHalf:
      return fe::DataType::HALF;
    case dnn::DataType::kBF16:
      return fe::DataType::BFLOAT16;
    case dnn::DataType::kInt8:
      return fe::DataType::INT8;
    case dnn::DataType::kInt32:
      return fe::DataType::INT32;
    case dnn::DataType::kF8E4M3FN:
    case dnn::DataType::kF8E4M3:
      return fe::DataType::FP8_E4M3;
    case dnn::DataType::kF8E5M2:
      return fe::DataType::FP8_E5M2;
    case dnn::DataType::kF8E4M3FNUZ:
      return fe::DataType::FP8_E4M3_FNUZ;
    case dnn::DataType::kF8E5M2FNUZ:
      return fe::DataType::FP8_E5M2_FNUZ;
    default:
      return absl::UnimplementedError(absl::StrCat(
          "hipDNN does not support data type ", static_cast<int>(data_type)));
  }
}

// Convolutions accumulate in fp32 for every narrow input type, matching
// GetConvAccumulatorType() on the cuDNN side.
fe::DataType HipdnnConvAccumulatorType(dnn::DataType data_type) {
  switch (data_type) {
    case dnn::DataType::kDouble:
      return fe::DataType::DOUBLE;
    case dnn::DataType::kInt8:
    case dnn::DataType::kInt32:
      return fe::DataType::INT32;
    default:
      return fe::DataType::FLOAT;
  }
}

absl::StatusOr<dnn::DataType> PrimitiveTypeStringToDnnType(
    absl::string_view data_type_string) {
  if (data_type_string == "f32") return dnn::DataType::kFloat;
  if (data_type_string == "f64") return dnn::DataType::kDouble;
  if (data_type_string == "f16") return dnn::DataType::kHalf;
  if (data_type_string == "bf16") return dnn::DataType::kBF16;
  if (data_type_string == "s8") return dnn::DataType::kInt8;
  if (data_type_string == "s32") return dnn::DataType::kInt32;
  if (data_type_string == "f8e4m3fn") return dnn::DataType::kF8E4M3FN;
  if (data_type_string == "f8e5m2") return dnn::DataType::kF8E5M2;
  return absl::InternalError(absl::StrCat(
      "Unsupported data type in graph serialization: ", data_type_string));
}

// The kind of the second operand of a fused op, and of its result. A kScalar
// operand is a broadcast value fed through an all-ones tensor; kTensor has the
// shape of the convolution result; kNone means the op is unary.
enum class TensorKind { kNone, kScalar, kTensor };

using OpMode = std::variant<fe::PointwiseMode, fe::ReductionMode>;

// Maps the op names emitted by CudnnFusedConvRewriter's graph serialization
// onto hipDNN modes. The vocabulary is fixed by
// OpNameStringToOperandKindAndMode() in cuda_dnn.cc and must stay in sync.
absl::StatusOr<std::tuple<TensorKind, TensorKind, OpMode>>
OpNameStringToOperandKindAndMode(absl::string_view op_string) {
  if (op_string == "add") {
    return std::make_tuple(TensorKind::kTensor, TensorKind::kTensor,
                           OpMode(fe::PointwiseMode::ADD));
  }
  if (op_string == "min") {
    return std::make_tuple(TensorKind::kTensor, TensorKind::kTensor,
                           OpMode(fe::PointwiseMode::MIN));
  }
  if (op_string == "max") {
    return std::make_tuple(TensorKind::kTensor, TensorKind::kTensor,
                           OpMode(fe::PointwiseMode::MAX));
  }
  if (op_string == "elu") {
    return std::make_tuple(TensorKind::kNone, TensorKind::kTensor,
                           OpMode(fe::PointwiseMode::ELU_FWD));
  }
  if (op_string == "relu") {
    return std::make_tuple(TensorKind::kNone, TensorKind::kTensor,
                           OpMode(fe::PointwiseMode::RELU_FWD));
  }
  if (op_string == "scale") {
    return std::make_tuple(TensorKind::kScalar, TensorKind::kTensor,
                           OpMode(fe::PointwiseMode::MUL));
  }
  if (op_string == "invscale") {
    return std::make_tuple(TensorKind::kScalar, TensorKind::kTensor,
                           OpMode(fe::PointwiseMode::DIV));
  }
  if (op_string == "amax") {
    return std::make_tuple(TensorKind::kNone, TensorKind::kScalar,
                           OpMode(fe::ReductionMode::AMAX));
  }
  return absl::InternalError(
      absl::StrCat("Unknown op in graph serialization: ", op_string));
}

// One node of the fused graph, parsed out of the serialized form.
struct OpDescriptor {
  int64_t uid;
  std::vector<int64_t> operand_uids;
  OpMode mode;
  TensorKind operand_kind;
  TensorKind result_kind;
  dnn::DataType result_type;
  bool is_virtual;
  // The hipDNN tensor this op produces; filled in while building the graph.
  std::shared_ptr<fe::graph::TensorAttributes> result;
};

// Parses the sequence of ops fused into a ForwardGraph convolution custom call.
//
// The format, produced by CudnnFusedConvRewriter and documented alongside
// GetGenericCudnnOperationGraph() in cuda_dnn.cc, is
//   "UID:[output_type]conv();UID:[output_type]op_name(operand UIDs);..."
// with the convolution assumed to be the first op. Operand UIDs referring to
// values produced outside the graph are elided.
absl::StatusOr<std::vector<OpDescriptor>> ParseSerializedGraph(
    absl::string_view serialized_graph) {
  std::vector<OpDescriptor> ops;
  absl::string_view::size_type pos = 0;
  while (pos < serialized_graph.size()) {
    absl::string_view::size_type m = serialized_graph.find('[', pos);
    absl::string_view::size_type n = serialized_graph.find(']', pos);
    if (m == absl::string_view::npos || n == absl::string_view::npos) {
      return absl::InternalError("Malformed graph serialization.");
    }
    const int64_t uid =
        std::stol(std::string(serialized_graph.substr(pos, m - pos - 1)));
    const absl::string_view data_type_string =
        serialized_graph.substr(m + 1, n - m - 1);
    m = serialized_graph.find('(', pos);
    const absl::string_view op_string = serialized_graph.substr(n + 1, m - n - 1);

    std::vector<int64_t> operands;
    absl::string_view::size_type l = serialized_graph.find_first_of(",)", m + 1);
    if (l == absl::string_view::npos) {
      return absl::InternalError("Malformed graph serialization.");
    }
    while (l > m + 1) {
      operands.push_back(
          std::stol(std::string(serialized_graph.substr(m + 1, l - m - 1))));
      if (serialized_graph[l] == ')') {
        break;
      }
      m = l;
      l = serialized_graph.find_first_of(",)", m + 1);
      if (l == absl::string_view::npos) {
        return absl::InternalError("Malformed graph serialization.");
      }
    }
    if (l + 1 >= serialized_graph.size() || serialized_graph[l + 1] != ';') {
      return absl::InternalError("Unexpected character in graph serialization.");
    }
    pos = l + 2;

    TF_ASSIGN_OR_RETURN(const dnn::DataType result_type,
                        PrimitiveTypeStringToDnnType(data_type_string));

    OpDescriptor op;
    op.uid = uid;
    op.operand_uids = operands;
    op.result_type = result_type;
    op.is_virtual = false;
    if (op_string == "conv") {
      if (!ops.empty()) {
        return absl::InternalError(
            "The graph must not contain more than one convolution op.");
      }
      if (!operands.empty()) {
        return absl::InternalError(
            "Convolution op must not have operands in the graph.");
      }
      op.operand_kind = TensorKind::kNone;
      op.result_kind = TensorKind::kTensor;
      op.mode = OpMode(fe::PointwiseMode::NOT_SET);
    } else {
      if (ops.empty()) {
        return absl::InternalError(
            "The first op in the graph must be a convolution.");
      }
      if (operands.empty()) {
        return absl::InternalError(
            "Non-convolution op must have one or more operands in the graph.");
      }
      TF_ASSIGN_OR_RETURN(std::tie(op.operand_kind, op.result_kind, op.mode),
                          OpNameStringToOperandKindAndMode(op_string));
    }

    // Any op consumed inside the graph produces a virtual intermediate.
    for (int64_t operand_uid : operands) {
      auto it = std::find_if(ops.begin(), ops.end(),
                             [operand_uid](const OpDescriptor& o) {
                               return o.uid == operand_uid;
                             });
      if (it == ops.end()) {
        return absl::InternalError(absl::StrCat(
            "Unknown operand UID in graph serialization: ", operand_uid));
      }
      it->is_virtual = true;
    }
    ops.push_back(std::move(op));
  }
  return ops;
}

// A built hipDNN graph plus the non-virtual tensor UIDs in the order XLA passes
// the corresponding buffers: convolution operands first, then outputs.
struct HipdnnConvGraph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::vector<int64_t> non_virtual_uids;
};

// Builds the hipDNN graph for a forward convolution, with the pointwise and
// reduction ops of a ForwardGraph custom call fused into it.
absl::StatusOr<HipdnnConvGraph> BuildHipdnnConvGraph(
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    absl::string_view serialized_graph) {
  // Only the forward kinds reach this path: __cudnn$convForwardGraph is the
  // sole graph-conv custom call target, and the backward targets stay on the
  // legacy MIOpen runners.
  if (kind != dnn::ConvolutionKind::FORWARD &&
      kind != dnn::ConvolutionKind::FORWARD_GRAPH) {
    return absl::UnimplementedError(
        absl::StrCat("hipDNN graph convolutions support forward kinds only, "
                     "got kind ",
                     static_cast<int>(kind)));
  }

  TF_ASSIGN_OR_RETURN(std::vector<OpDescriptor> ops,
                      ParseSerializedGraph(serialized_graph));
  if (ops.empty()) {
    return absl::InternalError("No supported ops in convolution graph.");
  }

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_compute_data_type(HipdnnConvAccumulatorType(input_type));

  // UIDs are handed out from a single counter. Operands and outputs are tracked
  // separately because the variant pack orders operands before outputs, which
  // is the order the convolution thunk supplies buffers in.
  int64_t uid_counter = 0;
  std::vector<int64_t> operand_uids, output_uids;
  auto next_uid = [&](bool is_operand, bool is_virtual) -> int64_t {
    const int64_t uid = ++uid_counter;
    if (is_operand) {
      operand_uids.push_back(uid);
    } else if (!is_virtual) {
      output_uids.push_back(uid);
    }
    return uid;
  };

  TF_ASSIGN_OR_RETURN(const fe::DataType hipdnn_input_type,
                      ToHipdnnDataType(input_type));

  auto tensor_x = graph->tensor(
      fe::graph::TensorAttributes()
          .set_name("x")
          .set_uid(next_uid(/*is_operand=*/true, /*is_virtual=*/false))
          .set_dim(input_descriptor.full_dims(dnn::DataLayout::kBatchDepthYX))
          .set_stride(
              input_descriptor.full_strides(dnn::DataLayout::kBatchDepthYX))
          .set_data_type(hipdnn_input_type));

  auto tensor_w = graph->tensor(
      fe::graph::TensorAttributes()
          .set_name("w")
          .set_uid(next_uid(/*is_operand=*/true, /*is_virtual=*/false))
          .set_dim(
              filter_descriptor.full_dims(dnn::FilterLayout::kOutputInputYX))
          .set_stride(
              filter_descriptor.full_strides(dnn::FilterLayout::kOutputInputYX))
          .set_data_type(hipdnn_input_type));

  const std::vector<int64_t> conv_strides(
      convolution_descriptor.strides().begin(),
      convolution_descriptor.strides().end());
  const std::vector<int64_t> conv_padding(
      convolution_descriptor.padding().begin(),
      convolution_descriptor.padding().end());
  const std::vector<int64_t> conv_dilations(
      convolution_descriptor.dilations().begin(),
      convolution_descriptor.dilations().end());

  const std::vector<int64_t> output_dims =
      output_descriptor.full_dims(dnn::DataLayout::kBatchDepthYX);
  const std::vector<int64_t> output_strides =
      output_descriptor.full_strides(dnn::DataLayout::kBatchDepthYX);

  TF_ASSIGN_OR_RETURN(const fe::DataType conv_result_type,
                      ToHipdnnDataType(ops[0].result_type));

  auto conv_result = graph->conv_fprop(
      tensor_x, tensor_w,
      fe::graph::ConvFpropAttributes()
          .set_pre_padding(conv_padding)
          .set_post_padding(conv_padding)
          .set_stride(conv_strides)
          .set_dilation(conv_dilations)
          .set_convolution_mode(
              convolution_descriptor.convolution_not_crosscorr()
                  ? fe::ConvolutionMode::CONVOLUTION
                  : fe::ConvolutionMode::CROSS_CORRELATION));
  conv_result->set_name("y")
      .set_uid(next_uid(/*is_operand=*/false, ops[0].is_virtual))
      .set_dim(output_dims)
      .set_stride(output_strides)
      .set_data_type(conv_result_type)
      .set_is_virtual(ops[0].is_virtual)
      .set_output(!ops[0].is_virtual);
  ops[0].result = conv_result;

  for (size_t op_index = 1; op_index < ops.size(); ++op_index) {
    OpDescriptor& op = ops[op_index];
    std::vector<const OpDescriptor*> preceding_ops;
    preceding_ops.reserve(op.operand_uids.size());
    for (int64_t operand_uid : op.operand_uids) {
      auto it = std::find_if(ops.begin(), ops.end(),
                             [operand_uid](const OpDescriptor& o) {
                               return o.uid == operand_uid;
                             });
      preceding_ops.push_back(&*it);
    }

    TF_ASSIGN_OR_RETURN(const fe::DataType hipdnn_result_type,
                        ToHipdnnDataType(op.result_type));

    // The second operand of a binary op that is not itself produced inside the
    // graph: either a broadcast scalar or a full-size side input.
    std::shared_ptr<fe::graph::TensorAttributes> external_operand;
    if (preceding_ops.size() == 1 && op.operand_kind != TensorKind::kNone) {
      TF_ASSIGN_OR_RETURN(const fe::DataType operand_type,
                          ToHipdnnDataType(preceding_ops[0]->result_type));
      const bool is_scalar = op.operand_kind == TensorKind::kScalar;
      const std::vector<int64_t> scalar_shape(output_dims.size(), 1);
      external_operand = graph->tensor(
          fe::graph::TensorAttributes()
              .set_name(absl::StrCat(is_scalar ? "scalar_" : "side_input_",
                                     op.uid))
              .set_uid(next_uid(/*is_operand=*/true, /*is_virtual=*/false))
              .set_dim(is_scalar ? scalar_shape : output_dims)
              .set_stride(is_scalar ? scalar_shape : output_strides)
              .set_data_type(operand_type));
    }

    if (std::holds_alternative<fe::PointwiseMode>(op.mode)) {
      auto attributes = fe::graph::PointwiseAttributes()
                            .set_mode(std::get<fe::PointwiseMode>(op.mode))
                            .set_compute_data_type(fe::DataType::FLOAT);
      if (external_operand != nullptr) {
        op.result = graph->pointwise(preceding_ops[0]->result, external_operand,
                                     attributes);
      } else if (preceding_ops.size() == 2) {
        op.result = graph->pointwise(preceding_ops[0]->result,
                                     preceding_ops[1]->result, attributes);
      } else {
        op.result = graph->pointwise(preceding_ops[0]->result, attributes);
      }
    } else {
      op.result = graph->reduction(
          preceding_ops[0]->result,
          fe::graph::ReductionAttributes()
              .set_mode(std::get<fe::ReductionMode>(op.mode))
              .set_compute_data_type(fe::DataType::FLOAT));
    }

    const std::vector<int64_t> scalar_shape(output_dims.size(), 1);
    const bool scalar_result = op.result_kind == TensorKind::kScalar;
    op.result->set_name(absl::StrCat("op_", op.uid))
        .set_uid(next_uid(/*is_operand=*/false, op.is_virtual))
        .set_dim(scalar_result ? scalar_shape : output_dims)
        .set_stride(scalar_result ? scalar_shape : output_strides)
        .set_data_type(hipdnn_result_type)
        .set_is_virtual(op.is_virtual)
        .set_output(!op.is_virtual);
  }

  std::vector<int64_t> non_virtual_uids = operand_uids;
  non_virtual_uids.insert(non_virtual_uids.end(), output_uids.begin(),
                          output_uids.end());

  return HipdnnConvGraph{std::move(graph), std::move(non_virtual_uids)};
}

template <typename Sig>
class HipdnnExecutionPlanRunner;

// An OpRunner backed by one compiled hipDNN execution plan.
//
// hipDNN identifies a compiled plan by its index into the graph's plan list,
// which is what dnn::AlgorithmDesc::algo_id() carries here. Re-deriving a
// runner from an AlgorithmDesc therefore means rebuilding the graph and
// compiling the plan at that index, which GraphConvolveRunnerFromDesc does.
template <typename... Args>
class HipdnnExecutionPlanRunner<void(Args...)>
    : public dnn::OpRunner<void(Args...)> {
 public:
  HipdnnExecutionPlanRunner(StreamExecutor* parent, HipdnnAccess* hipdnn,
                            std::shared_ptr<fe::graph::Graph> graph,
                            int64_t plan_index, size_t workspace_size,
                            std::string plan_name, std::vector<int64_t> uids,
                            std::vector<int64_t> scalar_uids = {},
                            std::vector<float> scalar_values = {})
      : parent_(parent),
        hipdnn_(hipdnn),
        graph_(std::move(graph)),
        plan_index_(plan_index),
        workspace_size_(workspace_size),
        plan_name_(std::move(plan_name)),
        uids_(std::move(uids)),
        scalar_uids_(std::move(scalar_uids)),
        scalar_values_(std::move(scalar_values)) {}

  std::string ToString() const override { return plan_name_; }

  size_t GetWorkspaceSize() const override { return workspace_size_; }

  absl::StatusOr<dnn::AlgorithmDesc> ToAlgorithmDesc() const override {
    return dnn::AlgorithmDesc(plan_index_, /*use_tensor_ops=*/false,
                              workspace_size_);
  }

  absl::Status operator()(Stream* stream, dnn::ProfileResult* profile_result,
                          DeviceAddressBase scratch_memory,
                          Args... inputs) const override {
    if (parent_ != stream->parent()) {
      return absl::InternalError(
          "HipdnnExecutionPlanRunner cached across multiple StreamExecutors.");
    }

    std::vector<void*> data_ptrs;
    // ForwardGraph convolutions gather their operands dynamically, so Args...
    // is a single std::vector<DeviceAddressBase>.
    if constexpr (sizeof...(Args) == 1 &&
                  std::is_same_v<std::tuple_element_t<0, std::tuple<Args...>>,
                                 std::vector<DeviceAddressBase>>) {
      for (DeviceAddressBase input : std::get<0>(std::tie(inputs...))) {
        data_ptrs.push_back(input.opaque());
      }
    } else {
      data_ptrs = {inputs.opaque()...};
    }

    if (data_ptrs.size() != uids_.size()) {
      return absl::InternalError(
          absl::StrCat("hipDNN graph expects ", uids_.size(),
                       " buffers but got ", data_ptrs.size(), "."));
    }

    std::unordered_map<int64_t, void*> variant_pack;
    variant_pack.reserve(uids_.size() + scalar_uids_.size());
    for (size_t i = 0; i < uids_.size(); ++i) {
      variant_pack[uids_[i]] = data_ptrs[i];
    }
    // Pass-by-value operands such as a norm's epsilon are read from host
    // memory, so the runner's own copy is what goes into the pack.
    for (size_t i = 0; i < scalar_uids_.size(); ++i) {
      variant_pack[scalar_uids_[i]] =
          const_cast<float*>(&scalar_values_[i]);
    }

    auto hipdnn = hipdnn_->GetHandle(parent_, stream);

    std::unique_ptr<EventBasedTimer> timer;
    if (profile_result != nullptr) {
      TF_ASSIGN_OR_RETURN(timer, stream->CreateEventBasedTimer(
                                     profile_result->warmup_run_executed()));
    }

    RETURN_IF_HIPDNN_FRONTEND_ERROR(graph_->execute_plan_at_index(
        hipdnn.handle(), variant_pack, scratch_memory.opaque(), plan_index_));

    if (timer != nullptr) {
      TF_ASSIGN_OR_RETURN(const dnn::AlgorithmDesc desc, ToAlgorithmDesc());
      TF_RETURN_IF_ERROR(PopulateProfileFromTimer(
          timer.get(), desc, profile_result, scratch_memory.size()));
    }
    return absl::OkStatus();
  }

 private:
  StreamExecutor* parent_;
  HipdnnAccess* hipdnn_;
  // Shared because every runner derived from one graph shares its plan list.
  std::shared_ptr<fe::graph::Graph> graph_;
  int64_t plan_index_;
  size_t workspace_size_;
  std::string plan_name_;
  std::vector<int64_t> uids_;
  std::vector<int64_t> scalar_uids_;
  std::vector<float> scalar_values_;
};

// Runs the shared prologue: validate the graph, lower it and enumerate plans.
//
// Plan enumeration must stay deterministic for a given graph, because
// AlgorithmDesc::algo_id() is an index into the resulting plan list and
// GraphConvolveRunnerFromDesc() has to resolve it against a freshly built
// graph.
absl::Status PrepareGraph(fe::graph::Graph& graph, hipdnnHandle_t handle,
                          const EngineOptions& engine_options,
                          bool use_fallback) {
  // Unlike cuDNN, hipDNN exposes no per-plan numerical-note metadata, so there
  // is no way to restrict engine selection to deterministic plans. MIOpen
  // ignores the request on this backend too; warn rather than silently
  // implying a guarantee.
  LOG_IF(WARNING, engine_options.require_determinism)
      << "hipDNN cannot restrict convolution plan selection to deterministic "
         "engines; the request is ignored.";

  RETURN_IF_HIPDNN_FRONTEND_ERROR(graph.validate());
  RETURN_IF_HIPDNN_FRONTEND_ERROR(graph.build_operation_graph(handle));
  RETURN_IF_HIPDNN_FRONTEND_ERROR(graph.create_execution_plans(
      {use_fallback ? fe::HeuristicMode::FALLBACK : fe::HeuristicMode::A}));
  RETURN_IF_HIPDNN_FRONTEND_ERROR(graph.check_support());
  return absl::OkStatus();
}

// Compiles the plan at `plan_index` and wraps it in a runner.
absl::StatusOr<std::unique_ptr<const dnn::GraphConvRunner>> MakeRunner(
    StreamExecutor* parent, HipdnnAccess* hipdnn_access,
    hipdnnHandle_t handle, const HipdnnConvGraph& conv_graph,
    int64_t plan_index) {
  RETURN_IF_HIPDNN_FRONTEND_ERROR(
      conv_graph.graph->build_plan_at_index(plan_index));

  // The handle-taking overload reports the name a plugin-supplied engine
  // declares for itself; without it a plugin engine shows up as a raw ID.
  std::string plan_name;
  if (conv_graph.graph->get_plan_name_at_index(handle, plan_index, plan_name)
          .is_bad()) {
    plan_name = absl::StrCat("hipdnn_plan_", plan_index);
  }
  const int64_t workspace_size =
      conv_graph.graph->get_workspace_size_plan_at_index(plan_index);

  return std::make_unique<HipdnnExecutionPlanRunner<dnn::GraphConvSignature>>(
      parent, hipdnn_access, conv_graph.graph, plan_index,
      workspace_size < 0 ? 0 : workspace_size, std::move(plan_name),
      conv_graph.non_virtual_uids);
}

}  // namespace

// A built hipDNN graph behind the platform-independent dnn::DnnGraph handle
// that CuDnnThunk drives. XLA addresses the operands positionally, so the
// builders below must assign UID n to the n-th custom call operand.
class HipdnnGraph : public dnn::DnnGraph {
 public:
  explicit HipdnnGraph(fe::graph::Graph graph) : graph_(std::move(graph)) {}

  absl::Status Prepare(dnn::DnnSupport* dnn_support,
                       const DeviceDescription& gpu_device_info,
                       const EngineOptions& engine_options) override {
    TF_ASSIGN_OR_RETURN(HipdnnAccess * hipdnn_access,
                        GetHipdnnAccess(Parent(dnn_support)));
    auto hipdnn = hipdnn_access->GetHandle(Parent(dnn_support),
                                           /*stream=*/nullptr);
    return PrepareGraph(graph_, hipdnn.handle(), engine_options,
                        /*use_fallback=*/false);
  }

  absl::Status Build(dnn::DnnSupport* dnn_support,
                     const DeviceDescription& gpu_device_info,
                     std::optional<int64_t> plan_id) override {
    if (plan_id.has_value()) {
      RETURN_IF_HIPDNN_FRONTEND_ERROR(graph_.build_plan_at_index(*plan_id));
      return absl::OkStatus();
    }
    RETURN_IF_HIPDNN_FRONTEND_ERROR(graph_.build_plan_at_index(0));
    return absl::OkStatus();
  }

  absl::Status Execute(Stream& stream, absl::Span<DeviceAddressBase> operands,
                       int64_t local_device_ordinal) const override {
    int64_t workspace_size = 0;
    if (graph_.get_workspace_size(workspace_size).is_bad()) {
      workspace_size = 0;
    }

    absl::Span<DeviceAddressBase> tensor_operands = operands;
    DeviceAddressBase workspace;
    // XLA always appends a workspace buffer, degenerate or not.
    if (!operands.empty() &&
        (workspace_size > 0 || operands.back().size() == 0)) {
      if (workspace_size > 0) {
        workspace = operands.back();
        if (workspace.size() < static_cast<uint64_t>(workspace_size)) {
          return absl::InternalError(absl::StrCat(
              "hipDNN graph needs a workspace of ", workspace_size,
              " bytes but XLA reserved only ", workspace.size(), "."));
        }
      }
      tensor_operands = operands.first(operands.size() - 1);
    }

    std::unordered_map<int64_t, void*> variant_pack;
    variant_pack.reserve(tensor_operands.size());
    for (size_t i = 0; i < tensor_operands.size(); ++i) {
      variant_pack[static_cast<int64_t>(i) + 1] = tensor_operands[i].opaque();
    }

    StreamExecutor* parent = stream.parent();
    TF_ASSIGN_OR_RETURN(HipdnnAccess * hipdnn_access, GetHipdnnAccess(parent));
    auto hipdnn = hipdnn_access->GetHandle(parent, &stream);
    RETURN_HIPDNN_FRONTEND_STATUS(
        graph_.execute(hipdnn.handle(), variant_pack, workspace.opaque()));
  }

  void InitDropoutState(int64_t local_device_count, int64_t seed,
                        int64_t increment) override {
    // No hipDNN 0.4 engine accepts a dropout mask, so the builders reject
    // dropout before a graph is ever created and there is no state to seed.
    LOG(WARNING) << "hipDNN graphs do not support dropout; ignoring the "
                    "requested dropout state.";
  }

  absl::StatusOr<bool> SupportsExplicitCommandBufferConstruction()
      const override {
    // hipDNN has no equivalent of cuDNN's native CUDA-graph population API.
    return false;
  }

  absl::Status PopulateOrUpdateRawCommandBuffer(Stream&,
                                                absl::Span<DeviceAddressBase>,
                                                RawCommandBufferHandle,
                                                bool) override {
    return absl::UnimplementedError(
        "hipDNN graphs cannot be captured into a command buffer directly.");
  }

  fe::graph::Graph& Graph() { return graph_; }
  const fe::graph::Graph& Graph() const { return graph_; }

 private:
  static StreamExecutor* Parent(dnn::DnnSupport* dnn_support) {
    return static_cast<MIOpenSupport*>(dnn_support)->parent();
  }

  fe::graph::Graph graph_;
};

namespace {

// hipDNN 0.4 has engines only for plain and bias-added flash attention; the
// rest of the knobs XLA can set on an fmha Custom Call have no implementation,
// and silently dropping any of them would change results.
absl::Status CheckFlashAttentionSupported(bool use_dropout,
                                          dnn::FMHAMaskKind mask_type,
                                          int sliding_window_length,
                                          int max_seg_per_batch,
                                          bool is_paged_attention,
                                          bool has_score_mod) {
  if (use_dropout) {
    return absl::UnimplementedError("hipDNN attention does not do dropout.");
  }
  if (mask_type != dnn::FMHAMaskKind::NO_MASK &&
      mask_type != dnn::FMHAMaskKind::CAUSAL) {
    return absl::UnimplementedError(
        "hipDNN attention supports no mask and causal masks only.");
  }
  if (sliding_window_length > 0) {
    return absl::UnimplementedError(
        "hipDNN attention does not do sliding windows.");
  }
  if (max_seg_per_batch > 1) {
    return absl::UnimplementedError(
        "hipDNN attention does not do ragged sequences.");
  }
  if (is_paged_attention) {
    return absl::UnimplementedError("hipDNN attention is not paged.");
  }
  if (has_score_mod) {
    return absl::UnimplementedError(
        "hipDNN attention does not take a score modifier.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<dnn::DnnGraph>>
GetHipdnnFlashAttentionOperationGraph(
    dnn::DnnSupport* dnn_support, const DeviceDescription& gpu_device_info,
    const dnn::MatmulTensorDescriptor& q_descriptor,
    const dnn::MatmulTensorDescriptor& k_descriptor,
    const dnn::MatmulTensorDescriptor& v_descriptor,
    const dnn::TensorDescriptor& o_descriptor,
    std::optional<dnn::TensorDescriptor> bias_descriptor,
    std::optional<dnn::TensorDescriptor> stats_descriptor, double scale,
    bool use_dropout, dnn::FMHAMaskKind mask_type, int sliding_window_length,
    int max_seg_per_batch, bool is_paged_attention, bool has_score_mod) {
  TF_RETURN_IF_ERROR(CheckFlashAttentionSupported(
      use_dropout, mask_type, sliding_window_length, max_seg_per_batch,
      is_paged_attention, has_score_mod));

  const dnn::DataType io_type = q_descriptor.type();
  if (k_descriptor.type() != io_type || v_descriptor.type() != io_type ||
      o_descriptor.type() != io_type) {
    return absl::InvalidArgumentError(
        "hipDNN attention requires q, k, v and o to share a data type.");
  }
  TF_ASSIGN_OR_RETURN(const fe::DataType hipdnn_io_type,
                      ToHipdnnDataType(io_type));

  fe::graph::Graph graph;
  graph.set_intermediate_data_type(fe::DataType::FLOAT)
      .set_io_data_type(hipdnn_io_type)
      .set_compute_data_type(fe::DataType::FLOAT);

  // UIDs are 1-based positions in the Custom Call's operand list, which is
  // what HipdnnGraph::Execute packs against.
  int64_t uid = 0;
  auto next_uid = [&uid]() { return ++uid; };

  auto q = graph.tensor(
      fe::graph::TensorAttributes()
          .set_name("q")
          .set_dim(q_descriptor.GetCudnnCompatibleDimensions(true))
          .set_stride(q_descriptor.GetCudnnCompatibleStrides(true))
          .set_uid(next_uid()));
  auto k = graph.tensor(
      fe::graph::TensorAttributes()
          .set_name("k")
          .set_dim(k_descriptor.GetCudnnCompatibleDimensions(true))
          .set_stride(k_descriptor.GetCudnnCompatibleStrides(true))
          .set_uid(next_uid()));
  auto v = graph.tensor(
      fe::graph::TensorAttributes()
          .set_name("v")
          .set_dim(v_descriptor.GetCudnnCompatibleDimensions(false))
          .set_stride(v_descriptor.GetCudnnCompatibleStrides(false))
          .set_uid(next_uid()));

  fe::graph::SdpaAttributes sdpa_options;
  sdpa_options.set_name("flash_attention")
      .set_generate_stats(stats_descriptor.has_value())
      .set_causal_mask(mask_type == dnn::FMHAMaskKind::CAUSAL)
      .set_attn_scale(static_cast<float>(scale));

  if (bias_descriptor.has_value()) {
    TF_ASSIGN_OR_RETURN(const fe::DataType bias_type,
                        ToHipdnnDataType(bias_descriptor->type()));
    sdpa_options.set_bias(
        graph.tensor(fe::graph::TensorAttributes()
                         .set_name("bias")
                         .set_dim(bias_descriptor->dimensions())
                         .set_stride(bias_descriptor->GetLogicalStrides())
                         .set_data_type(bias_type)
                         .set_uid(next_uid())));
  }

  auto outputs = graph.sdpa(q, k, v, sdpa_options);
  outputs[0]
      ->set_name("o")
      .set_output(true)
      .set_dim(o_descriptor.dimensions())
      .set_stride(o_descriptor.GetLogicalStrides())
      .set_uid(next_uid());
  if (stats_descriptor.has_value()) {
    TF_ASSIGN_OR_RETURN(const fe::DataType stats_type,
                        ToHipdnnDataType(stats_descriptor->type()));
    // hipDNN, like cuDNN, wants the stats tensor spelled out in 4D.
    std::vector<int64_t> stats_dims = stats_descriptor->dimensions();
    std::vector<int64_t> stats_strides = stats_descriptor->GetLogicalStrides();
    stats_dims.push_back(1);
    stats_strides.push_back(1);
    outputs[1]
        ->set_name("stats")
        .set_output(true)
        .set_data_type(stats_type)
        .set_dim(stats_dims)
        .set_stride(stats_strides)
        .set_uid(next_uid());
  }

  auto result = std::make_unique<HipdnnGraph>(std::move(graph));
  TF_RETURN_IF_ERROR(result->Prepare(
      dnn_support, gpu_device_info,
      EngineOptions{/*require_determinism=*/false, /*allow_tf32=*/true,
                    /*require_command_buffer=*/false}));
  TF_RETURN_IF_ERROR(
      result->Build(dnn_support, gpu_device_info, /*plan_id=*/std::nullopt));
  return result;
}

absl::StatusOr<std::string> SerializeHipdnnGraph(const dnn::DnnGraph& graph) {
  std::vector<uint8_t> blob;
  RETURN_IF_HIPDNN_FRONTEND_ERROR(
      static_cast<const HipdnnGraph&>(graph).Graph().serialize(blob));
  return std::string(reinterpret_cast<const char*>(blob.data()), blob.size());
}

absl::StatusOr<int64_t> HipdnnGraphWorkspaceSize(const dnn::DnnGraph& graph) {
  int64_t workspace_size = 0;
  RETURN_IF_HIPDNN_FRONTEND_ERROR(
      static_cast<const HipdnnGraph&>(graph).Graph().get_workspace_size(
          workspace_size));
  return workspace_size;
}

absl::StatusOr<std::unique_ptr<dnn::DnnGraph>> MIOpenSupport::DeserializeGraph(
    Stream& stream, absl::string_view serialized_data) const {
  TF_ASSIGN_OR_RETURN(HipdnnAccess * hipdnn_access, GetHipdnnAccess(parent_));
  auto hipdnn = hipdnn_access->GetHandle(parent_, &stream);

  fe::graph::Graph graph;
  RETURN_IF_HIPDNN_FRONTEND_ERROR(graph.deserialize(
      hipdnn.handle(),
      std::vector<uint8_t>(serialized_data.begin(), serialized_data.end())));

  // hipDNN's plugin engines cannot serialize a compiled plan, so a blob only
  // ever carries the graph structure and the plan has to be chosen again here.
  // Heuristic ordering is deterministic, so this picks the same engine the
  // compiler did.
  int64_t workspace_size = 0;
  if (graph.get_workspace_size(workspace_size).is_bad()) {
    RETURN_IF_HIPDNN_FRONTEND_ERROR(
        graph.create_execution_plans({fe::HeuristicMode::A}));
    RETURN_IF_HIPDNN_FRONTEND_ERROR(graph.check_support());
    RETURN_IF_HIPDNN_FRONTEND_ERROR(graph.build_plan_at_index(0));
  }
  return std::make_unique<HipdnnGraph>(std::move(graph));
}

std::string HipdnnVersionString() { return hipdnnVersionString_ext(); }

absl::StatusOr<dnn::VersionInfo> HipdnnVersion() {
  const std::string version = HipdnnVersionString();
  int major = 0, minor = 0, patch = 0;
  if (std::sscanf(version.c_str(), "%d.%d.%d", &major, &minor, &patch) < 2) {
    return absl::InternalError(
        absl::StrCat("Could not parse hipDNN version string '", version, "'."));
  }
  return dnn::VersionInfo(major, minor, patch);
}

absl::Status MIOpenSupport::GetGraphConvolveRunners(
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    dnn::DataType output_type, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor, bool use_fallback,
    const EngineOptions& engine_options,
    std::vector<std::unique_ptr<const dnn::GraphConvRunner>>* out_exec_plans,
    std::string serialized_graph) {
  TF_ASSIGN_OR_RETURN(HipdnnAccess * hipdnn_access, GetHipdnnAccess(parent_));
  TF_ASSIGN_OR_RETURN(
      const HipdnnConvGraph conv_graph,
      BuildHipdnnConvGraph(kind, input_type, input_descriptor,
                           filter_descriptor, output_descriptor,
                           convolution_descriptor, serialized_graph));

  auto hipdnn = hipdnn_access->GetHandle(parent_, stream);
  TF_RETURN_IF_ERROR(PrepareGraph(*conv_graph.graph, hipdnn.handle(),
                                  engine_options, use_fallback));

  const int64_t plan_count = conv_graph.graph->get_execution_plan_count();
  for (int64_t plan_index = 0; plan_index < plan_count; ++plan_index) {
    absl::StatusOr<std::unique_ptr<const dnn::GraphConvRunner>> runner =
        MakeRunner(parent_, hipdnn_access, hipdnn.handle(), conv_graph,
                   plan_index);
    // A plan that fails to compile is simply not offered to the autotuner; the
    // remaining plans still give it something to choose from.
    if (!runner.ok()) {
      VLOG(2) << "hipDNN plan " << plan_index
              << " is unusable: " << runner.status();
      continue;
    }
    out_exec_plans->push_back(*std::move(runner));
  }

  if (out_exec_plans->empty()) {
    return absl::InternalError(
        "hipDNN found no executable plan for the convolution graph.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<const dnn::GraphConvRunner>>
MIOpenSupport::GraphConvolveRunnerFromDesc(
    Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    dnn::DataType output_type, const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    std::string serialized_graph) {
  TF_ASSIGN_OR_RETURN(HipdnnAccess * hipdnn_access, GetHipdnnAccess(parent_));
  TF_ASSIGN_OR_RETURN(
      const HipdnnConvGraph conv_graph,
      BuildHipdnnConvGraph(kind, input_type, input_descriptor,
                           filter_descriptor, output_descriptor,
                           convolution_descriptor, serialized_graph));

  // Enumerated with the default heuristic, matching the use_fallback=false
  // call in GetGraphConvolveRunners(): an AlgorithmDesc carries only the plan
  // index, so a plan picked out of the fallback list cannot be recovered here.
  // The autotuner only reaches for fallback when the default list is empty, in
  // which case the graph does not resolve under either mode.
  auto hipdnn = hipdnn_access->GetHandle(parent_, stream);
  TF_RETURN_IF_ERROR(PrepareGraph(*conv_graph.graph, hipdnn.handle(),
                                  EngineOptions(), /*use_fallback=*/false));

  const int64_t plan_index = algorithm_desc.algo_id();
  if (plan_index < 0 ||
      plan_index >= conv_graph.graph->get_execution_plan_count()) {
    return absl::InternalError(absl::StrCat(
        "hipDNN plan index ", plan_index, " is out of range; the graph has ",
        conv_graph.graph->get_execution_plan_count(), " plans."));
  }
  return MakeRunner(parent_, hipdnn_access, hipdnn.handle(), conv_graph,
                    plan_index);
}

absl::StatusOr<std::unique_ptr<const dnn::NormRunner>>
MIOpenSupport::NormRunnerFromDesc(
    Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
    dnn::NormKind kind, double epsilon,
    const dnn::TensorDescriptor& x_descriptor,
    const dnn::TensorDescriptor& scale_descriptor,
    const dnn::TensorDescriptor& y_or_dx_descriptor,
    std::optional<dnn::TensorDescriptor> bias_descriptor,
    std::optional<dnn::TensorDescriptor> dy_descriptor,
    std::optional<dnn::TensorDescriptor> expectation_descriptor,
    std::optional<dnn::TensorDescriptor> norm_factor_descriptor,
    std::optional<dnn::TensorDescriptor> dscale_descriptor,
    std::optional<dnn::TensorDescriptor> dbias_descriptor) {
  TF_ASSIGN_OR_RETURN(HipdnnAccess * hipdnn_access, GetHipdnnAccess(parent_));

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_compute_data_type(fe::DataType::FLOAT);

  // UIDs follow the buffer order RunGpuNorm() builds: x, scale, y_or_dx, then
  // the per-kind operands. Assigning them explicitly keeps the mapping
  // independent of the order the graph nodes happen to be created in.
  int64_t uid_counter = 0;
  std::vector<int64_t> uids;
  auto make_tensor = [&](const dnn::TensorDescriptor& descriptor,
                         absl::string_view name)
      -> absl::StatusOr<std::shared_ptr<fe::graph::TensorAttributes>> {
    TF_ASSIGN_OR_RETURN(const fe::DataType data_type,
                        ToHipdnnDataType(descriptor.type()));
    const int64_t uid = ++uid_counter;
    uids.push_back(uid);
    return graph->tensor(
        fe::graph::TensorAttributes()
            .set_name(std::string(name))
            .set_uid(uid)
            .set_dim(descriptor.dimensions())
            .set_stride(descriptor.GetPhysicalStridesMajorToMinor())
            .set_data_type(data_type));
  };
  // Outputs are produced by the graph nodes, so only their UID is reserved
  // here; the attributes get filled in once the node exists.
  auto reserve_uid = [&]() -> int64_t {
    const int64_t uid = ++uid_counter;
    uids.push_back(uid);
    return uid;
  };
  auto finish_output =
      [](const std::shared_ptr<fe::graph::TensorAttributes>& tensor,
         int64_t uid, const dnn::TensorDescriptor& descriptor)
      -> absl::Status {
    TF_ASSIGN_OR_RETURN(const fe::DataType data_type,
                        ToHipdnnDataType(descriptor.type()));
    tensor->set_uid(uid)
        .set_dim(descriptor.dimensions())
        .set_stride(descriptor.GetPhysicalStridesMajorToMinor())
        .set_data_type(data_type)
        .set_output(true);
    return absl::OkStatus();
  };

  std::vector<int64_t> scalar_uids;
  std::vector<float> scalar_values;

  TF_ASSIGN_OR_RETURN(auto x_tensor, make_tensor(x_descriptor, "x"));
  TF_ASSIGN_OR_RETURN(auto scale_tensor, make_tensor(scale_descriptor, "scale"));
  const int64_t y_or_dx_uid = reserve_uid();

  switch (kind) {
    case dnn::LAYER_FWD_INFER:
    case dnn::LAYER_FWD_TRAIN: {
      const bool training = kind == dnn::LAYER_FWD_TRAIN;
      if (!bias_descriptor.has_value()) {
        return absl::InvalidArgumentError(
            "Forward layer normalization requires a bias descriptor.");
      }
      TF_ASSIGN_OR_RETURN(auto bias_tensor,
                          make_tensor(*bias_descriptor, "bias"));
      const int64_t expectation_uid = training ? reserve_uid() : 0;
      const int64_t norm_factor_uid = training ? reserve_uid() : 0;

      // Epsilon is a host-side value rather than a device buffer, so it gets a
      // UID outside the ones the thunk supplies buffers for.
      const int64_t epsilon_uid = ++uid_counter;
      scalar_uids.push_back(epsilon_uid);
      scalar_values.push_back(static_cast<float>(epsilon));
      auto epsilon_tensor = graph->tensor(
          fe::graph::TensorAttributes()
              .set_name("epsilon")
              .set_uid(epsilon_uid)
              .set_dim({1, 1, 1, 1})
              .set_stride({1, 1, 1, 1})
              .set_is_pass_by_value(true)
              .set_data_type(fe::DataType::FLOAT));

      auto outputs = graph->layernorm(
          x_tensor, scale_tensor, bias_tensor,
          fe::graph::LayernormAttributes()
              .set_forward_phase(training ? fe::NormFwdPhase::TRAINING
                                          : fe::NormFwdPhase::INFERENCE)
              .set_epsilon(epsilon_tensor)
              .set_compute_data_type(fe::DataType::FLOAT));
      TF_RETURN_IF_ERROR(
          finish_output(outputs[0], y_or_dx_uid, y_or_dx_descriptor));
      if (training) {
        if (!expectation_descriptor.has_value() ||
            !norm_factor_descriptor.has_value()) {
          return absl::InvalidArgumentError(
              "Layer normalization training requires expectation and norm "
              "factor descriptors.");
        }
        TF_RETURN_IF_ERROR(finish_output(outputs[1], expectation_uid,
                                         *expectation_descriptor));
        TF_RETURN_IF_ERROR(finish_output(outputs[2], norm_factor_uid,
                                         *norm_factor_descriptor));
      }
      break;
    }
    case dnn::LAYER_BWD: {
      if (!dy_descriptor.has_value() || !expectation_descriptor.has_value() ||
          !norm_factor_descriptor.has_value() ||
          !dscale_descriptor.has_value() || !dbias_descriptor.has_value()) {
        return absl::InvalidArgumentError(
            "Backward layer normalization is missing a descriptor.");
      }
      TF_ASSIGN_OR_RETURN(auto dy_tensor, make_tensor(*dy_descriptor, "dy"));
      TF_ASSIGN_OR_RETURN(auto expectation_tensor,
                          make_tensor(*expectation_descriptor, "expectation"));
      TF_ASSIGN_OR_RETURN(auto norm_factor_tensor,
                          make_tensor(*norm_factor_descriptor, "norm_factor"));
      const int64_t dscale_uid = reserve_uid();
      const int64_t dbias_uid = reserve_uid();

      // CudnnNormRewriter folds every non-normalized dimension into the first
      // one, so all remaining dimensions are normalized over. hipDNN infers
      // this for the forward pass but requires it spelled out here.
      auto outputs = graph->layernorm_backward(
          dy_tensor, x_tensor, scale_tensor,
          fe::graph::LayernormBackwardAttributes()
              .set_normalized_dim_count(
                  static_cast<int64_t>(x_descriptor.dimensions().size()) - 1)
              .set_saved_mean_and_inv_variance(expectation_tensor,
                                               norm_factor_tensor)
              .set_compute_data_type(fe::DataType::FLOAT));
      TF_RETURN_IF_ERROR(
          finish_output(outputs[0], y_or_dx_uid, y_or_dx_descriptor));
      TF_RETURN_IF_ERROR(
          finish_output(outputs[1], dscale_uid, *dscale_descriptor));
      TF_RETURN_IF_ERROR(
          finish_output(outputs[2], dbias_uid, *dbias_descriptor));
      break;
    }
    default:
      return absl::UnimplementedError(absl::StrCat(
          "hipDNN does not implement norm kind ", static_cast<int>(kind)));
  }

  auto hipdnn = hipdnn_access->GetHandle(parent_, stream);
  TF_RETURN_IF_ERROR(PrepareGraph(*graph, hipdnn.handle(), EngineOptions(),
                                  /*use_fallback=*/false));
  if (graph->get_execution_plan_count() == 0) {
    return absl::InternalError(
        "hipDNN found no executable plan for the normalization graph.");
  }
  RETURN_IF_HIPDNN_FRONTEND_ERROR(graph->build_plan_at_index(0));

  std::string plan_name;
  if (graph->get_plan_name_at_index(hipdnn.handle(), 0, plan_name).is_bad()) {
    plan_name = "hipdnn_norm_plan";
  }
  const int64_t workspace_size = graph->get_workspace_size_plan_at_index(0);

  return std::make_unique<HipdnnExecutionPlanRunner<dnn::NormSignature>>(
      parent_, hipdnn_access, std::move(graph), /*plan_index=*/0,
      workspace_size < 0 ? 0 : workspace_size, std::move(plan_name),
      std::move(uids), std::move(scalar_uids), std::move(scalar_values));
}

}  // namespace gpu
}  // namespace stream_executor
