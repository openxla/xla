/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_GPU_HLO_SCHEDULE_H_
#define XLA_SERVICE_GPU_GPU_HLO_SCHEDULE_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/stream_executor/device_description.h"
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"


#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/model/analytical_latency_estimator.h"
#include "xla/hlo/experimental/auto_reorder/auto_reorder.h"
#include "xla/service/hlo_memory_scheduler.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/service/p2p_schedule_preparation.h"
#include "xla/service/profile_guided_latency_estimator.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace gpu {

int64_t GetSizeOfShape(const Shape& shape, int pointer_size);

// Determines the schedule of HLO instructions for a module run on the GPU.
Status ScheduleGpuModule(HloModule* module, int64_t pointer_size,
                         int64_t memory_limit,
                         const se::DeviceDescription& gpu_device_info);
HloInstructionSequence PostProcessSchedule(const HloInstructionSequence& input);

int64_t GetSchedulerMemoryLimit(const HloModule* module,
                                const se::DeviceDescription& gpu_device_info,
                                int pointer_size);

constexpr absl::string_view kFingerprintBeforeLHS = "fingerprint_before_lhs";

}  // namespace gpu


}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_HLO_SCHEDULE_H_
