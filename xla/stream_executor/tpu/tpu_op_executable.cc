/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/stream_executor/tpu/tpu_op_executable.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "third_party/tensorflow/c/tf_status_helper.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_execution_profile.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/c_api_defn.h"  // IWYU pragma: keep
#include "xla/stream_executor/tpu/proto_helper.h"
#include "xla/stream_executor/tpu/status_helper.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/stream_executor/tpu/tpu_executable_interface.h"
#include "xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "xla/stream_executor/tpu/tpu_platform.h"
#include "xla/stream_executor/tpu/tpu_platform_interface.h"
#include "xla/xla_data.pb.h"
#include "tsl/framework/cancellation.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

TpuOpExecutable::TpuOpExecutable(
    const XLA_TpuProgram* core_program,
    std::unique_ptr<xla::HloModule> hlo_module,
    SE_OutsideCompilationParams* outside_compilation_params)
    : TpuExecutableInterface(std::move(hlo_module)),
      core_program_(core_program),
      outside_compilation_params_(outside_compilation_params) {}

tsl::Status TpuOpExecutable::LoadProgramAndEnqueueToStream(
    const xla::ServiceExecutableRunOptions& run_options,
    absl::Span<const se::DeviceMemoryBase> arguments,
    se::DeviceMemoryBase result,
    const std::vector<se::DeviceMemoryBase>& cross_program_prefetch_addrs,
    const std::vector<uint32_t>& cross_program_prefetch_offsets) {
  auto DeviceMemoryBaseToC = [](const se::DeviceMemoryBase& addr) {
    return SE_DeviceMemoryBase{const_cast<void*>(addr.opaque()), addr.size(),
                               addr.payload()};
  };

  std::vector<SE_DeviceMemoryBase> arguments_bases;
  arguments_bases.resize(arguments.size());
  absl::c_transform(arguments, arguments_bases.begin(), DeviceMemoryBaseToC);

  SE_DeviceMemoryBase result_base = DeviceMemoryBaseToC(result);

  std::vector<SE_DeviceMemoryBase> prefetch_bases;
  prefetch_bases.resize(cross_program_prefetch_addrs.size());
  absl::c_transform(cross_program_prefetch_addrs, prefetch_bases.begin(),
                    DeviceMemoryBaseToC);
  int32_t rng_seed = run_options.run_options().rng_seed();

  XLA_DeviceAssignment c_dev_assign{/*bytes=*/nullptr, /*size=*/0};
  auto dev_assign = run_options.run_options().device_assignment();
  stream_executor::tpu::SerializedProto dev_assign_serialized;
  if (dev_assign != nullptr) {
    xla::DeviceAssignmentProto dev_assign_proto;
    TF_RETURN_IF_ERROR(dev_assign->Serialize(&dev_assign_proto));
    dev_assign_serialized =
        stream_executor::tpu::SerializeProto(dev_assign_proto);
    c_dev_assign.bytes = dev_assign_serialized.bytes;
    c_dev_assign.size = dev_assign_serialized.size;
  }

  auto platform = down_cast<tpu::TpuPlatform*>(
      tpu::TpuPlatformInterface::GetRegisteredPlatform());
  auto stream = platform->LookupStream(
      run_options.run_options().stream()->implementation());
  StatusHelper status;

  TpuExecutable_LoadProgramAndEnqueueToStream_Params params;
  params.struct_size = TpuExecutable_LoadProgramAndEnqueueToStream_Params_SIZE;
  params.priv = nullptr;
  params.program = core_program_;
  params.arguments = arguments_bases.empty() ? nullptr : arguments_bases.data();
  params.arguments_len = arguments_bases.size();
  params.result = &result_base;
  params.cross_program_prefetch_addrs =
      prefetch_bases.empty() ? nullptr : prefetch_bases.data();
  params.cross_program_prefetch_addrs_len = prefetch_bases.size();
  params.cross_program_prefetch_offsets =
      cross_program_prefetch_offsets.empty()
          ? nullptr
          : cross_program_prefetch_offsets.data();
  params.cross_program_prefetch_offsets_len =
      cross_program_prefetch_offsets.size();
  params.rng_seed = rng_seed;
  params.device_assignment = &c_dev_assign;
  params.stream = stream;
  params.outside_compilation_params = outside_compilation_params_;
  params.status = status.c_status;

  stream_executor::tpu::OpsApiFn()
      ->TpuExecutable_LoadProgramAndEnqueueToStreamFn(&params);

  if (dev_assign != nullptr) {
    stream_executor::tpu::SerializedProto_Free(dev_assign_serialized);
  }
  return status.status();
}

tsl::StatusOr<xla::ExecutionOutput> TpuOpExecutable::ExecuteAsyncOnStream(
    const xla::ServiceExecutableRunOptions* run_options,
    std::vector<xla::ExecutionInput> arguments,
    xla::HloExecutionProfile* hlo_execution_profile) {
  const int device_ordinal = run_options->device_ordinal();
  tsl::CancellationToken token = 0;
  if (outside_compilation_params_ != nullptr) {
    TF_ASSIGN_OR_RETURN(token, RegisterCancellation(device_ordinal));
  }

  tsl::StatusOr<xla::ExecutionOutput> output =
      xla::TpuExecutableInterface::ExecuteAsyncOnStream(
          run_options, std::move(arguments),
          /*hlo_execution_profile=*/nullptr);

  if (outside_compilation_params_ != nullptr) {
    auto c_stream =
        std::make_unique<SE_Stream>(run_options->stream()->parent());

    TF_RETURN_IF_ERROR(UnregisterCancellation(output.status(), token,
                                              device_ordinal, c_stream.get()));
  }

  VLOG(1) << "Cloud TPU: TPUExecute done";
  return output;
}

namespace {
tsl::Status AlreadyCancelledError(const int device_ordinal) {
  return absl::CancelledError(absl::StrCat(
      "RPC cancelled, not running TPU program on device ", device_ordinal));
}
}  // namespace

tsl::StatusOr<tsl::CancellationToken> TpuOpExecutable::RegisterCancellation(
    const int device_ordinal) {
  TpuOp_RegisterCancellation_Params params;
  params.struct_size = TpuOp_RegisterCancellation_Params_SIZE;
  params.priv = nullptr;
  params.cancellation_manager =
      stream_executor::tpu::OpsApiFn()
          ->OutsideCompilation_GetCancellationManagerFn(
              outside_compilation_params_);
  params.device_ordinal = device_ordinal;
  TpuOp_CancellationResult cancellation_result;
  params.result = &cancellation_result;
  params.status = nullptr;

  stream_executor::tpu::OpsApiFn()->TpuOp_RegisterCancellationFn(&params);
  if (params.status != nullptr) {
    return StatusFromTF_Status(params.status);
  }

  // If the RPC was already cancelled before we managed to register the
  // cancellation callback, we shouldn't attempt to run the TPU program, since
  // it might block forever.
  if (cancellation_result.already_cancelled) {
    return AlreadyCancelledError(device_ordinal);
  }
  return static_cast<tsl::CancellationToken>(cancellation_result.token);
}

tsl::Status TpuOpExecutable::UnregisterCancellation(
    const tsl::Status& status, tsl::CancellationToken cancel_token,
    int device_ordinal, SE_Stream* c_stream) {
  const TfTpu_OpsApiFn* ops_api = stream_executor::tpu::OpsApiFn();
  TpuOp_CancellationManager* cancellation_manager =
      ops_api->OutsideCompilation_GetCancellationManagerFn(
          outside_compilation_params_);

  // If !output.ok(), it means we failed to enqueue the program the TPU. This is
  // possibly caused by a failed cancellation callback closing the chips.
  if (!status.ok()) {
    const bool already_cancelled =
        ops_api->TpuOp_CancellationManagerIsCancelledFn(cancellation_manager) ||
        ops_api->TpuOp_CancellationManagerIsCancellingFn(cancellation_manager);

    // If cancellation manager is already cancelled or cancelling, it means
    // another failure has occurred earlier and this TpuExecuteOp is cancelled
    // regardless of whether itself is an error.
    if (already_cancelled) {
      TF_RETURN_IF_ERROR(AlreadyCancelledError(device_ordinal));
    }
  }

  TpuOp_UnregisterCancellation_Params params;
  params.struct_size = TpuOp_UnregisterCancellation_Params_SIZE;
  params.priv = nullptr;
  params.op_kernel_context = ops_api->OutsideCompilation_GetOpKernelContextFn(
      outside_compilation_params_);
  params.cancellation_manager = cancellation_manager;
  params.stream = c_stream;
  params.device_ordinal = device_ordinal;
  params.token = static_cast<tsl::CancellationToken>(cancel_token);
  params.status = nullptr;
  params.host_transfer_manager =
      ops_api->OutsideCompilation_GetTransferManagerFn(
          outside_compilation_params_);

  ops_api->TpuOp_UnregisterCancellationFn(&params);
  if (params.status != nullptr) {
    return StatusFromTF_Status(params.status);
  }
  return tsl::OkStatus();
}

absl::string_view TpuOpExecutable::fingerprint() const {
  // TODO(skye): the fingerprint can be plumbed through via core_program_
  LOG(FATAL) << "TpuOpExecutable::fingerprint() unimplemented";
}

}  // namespace tensorflow
