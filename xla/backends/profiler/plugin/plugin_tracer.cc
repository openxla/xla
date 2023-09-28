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
#include "xla/backends/profiler/plugin/plugin_tracer.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_profiler_extension.h"
#include "xla/status.h"
#include "tsl/platform/logging.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

using tensorflow::profiler::XPlane;
using tensorflow::profiler::XSpace;

namespace {

absl::StatusOr<const PJRT_Profiler_Extension*> FindProfilerExtension(
    const PJRT_Api* pjrt_api) {
  const PJRT_Structure_Base* next =
      reinterpret_cast<const PJRT_Structure_Base*>(pjrt_api->extension_start);
  while (next != nullptr &&
         next->type != PJRT_Structure_Type::PJRT_Structure_Type_Profiler) {
    next = next->next;
  }
  if (next == nullptr) {
    return absl::UnimplementedError(
        "The plugin does not have a profiler extension.");
  }
  return reinterpret_cast<const PJRT_Profiler_Extension*>(next);
}
}  // namespace

PluginTracer::PluginTracer(const PJRT_Api* pjrt_api) : pjrt_api_(pjrt_api) {
  LOG(INFO) << "PluginTracer::PluginTracer started";
  absl::StatusOr<const PJRT_Profiler_Extension*> profiler_api =
      FindProfilerExtension(pjrt_api);
  if (!profiler_api.ok()) {
    LOG(ERROR) << profiler_api.status().message();
    return;
  }
  profiler_api_ = *profiler_api;

  PJRT_Profiler_Create_Args args;
  PJRT_Error* error = profiler_api_->create(&args);
  if (error != nullptr) {
    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> error_ptr(
        error, pjrt::MakeErrorDeleter(pjrt_api_));
    LOG(ERROR) << pjrt::GetPjrtErrorMessage(error_ptr.get(), pjrt_api_);
    return;
  }

  profiler_ = args.profiler;
  LOG(INFO) << "PluginTracer::PluginTracer finished with profilr created";
}

PluginTracer::~PluginTracer() {
  PJRT_Profiler_Destroy_Args args;
  args.profiler = profiler_;
  PJRT_Error* error = profiler_api_->destroy(&args);
  if (error != nullptr) {
    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> error_ptr(
        error, pjrt::MakeErrorDeleter(pjrt_api_));
    LOG(ERROR) << pjrt::GetPjrtErrorMessage(error_ptr.get(), pjrt_api_);
    return;
  }
}

Status PluginTracer::Start() {
  PJRT_Profiler_Start_Args args;
  args.profiler = profiler_;
  RETURN_STATUS_IF_PJRT_ERROR(profiler_api_->start(&args), pjrt_api_);
  return OkStatus();
}

Status PluginTracer::Stop() {
  PJRT_Profiler_Stop_Args args;
  args.profiler = profiler_;
  RETURN_STATUS_IF_PJRT_ERROR(profiler_api_->stop(&args), pjrt_api_);
  return OkStatus();
}

Status PluginTracer::CollectData(XSpace* space) {
  PJRT_Profiler_CollectData_Args args;
  args.profiler = profiler_;
  args.buffer = nullptr;
  RETURN_STATUS_IF_PJRT_ERROR(profiler_api_->collect_data(&args), pjrt_api_);
  LOG(ERROR) << "buffer_size_in_bytes: " << args.buffer_size_in_bytes;
  // Prepare an appropriately sized buffer.
  if (args.buffer_size_in_bytes > 0) {
    std::vector<uint8_t> buffer(args.buffer_size_in_bytes);
    args.buffer = buffer.data();
    RETURN_STATUS_IF_PJRT_ERROR(profiler_api_->collect_data(&args), pjrt_api_);
    // Deserialize XSpace from the buffer and return it.
    XSpace tpu_space;
    tpu_space.ParseFromArray(buffer.data(), buffer.size());
    for (XPlane& tpu_plane : *tpu_space.mutable_planes()) {
      XPlane* plane = space->add_planes();
      plane->Swap(&tpu_plane);
    }
  }
  return OkStatus();
}

}  // namespace profiler
}  // namespace xla
