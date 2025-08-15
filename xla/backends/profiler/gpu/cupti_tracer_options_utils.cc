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

#include "xla/backends/profiler/gpu/cupti_tracer_options_utils.h"

#include <cstdint>
#include <functional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/profiler/utils/profiler_options_util.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace xla {
namespace profiler {
using tsl::profiler::SetValue;

absl::Status UpdateCuptiTracerOptionsFromProfilerOptions(
    const tensorflow::ProfileOptions& profile_options,
    CuptiTracerOptions& tracer_options,
    CuptiTracerCollectorOptions& collector_options) {
  absl::flat_hash_set<absl::string_view> input_keys;
  for (const auto& [key, _] : profile_options.advanced_configuration()) {
    input_keys.insert(key);
  }

  TF_RETURN_IF_ERROR(
      SetValue<int64_t>(profile_options, "gpu_max_callback_api_events",
                        input_keys, [&](int64_t value) {
                          collector_options.max_callback_api_events = value;
                        }));

  TF_RETURN_IF_ERROR(
      SetValue<int64_t>(profile_options, "gpu_max_activity_api_events",
                        input_keys, [&](int64_t value) {
                          collector_options.max_activity_api_events = value;
                        }));

  TF_RETURN_IF_ERROR(
      SetValue<int64_t>(profile_options, "gpu_max_annotation_strings",
                        input_keys, [&](int64_t value) {
                          collector_options.max_annotation_strings = value;
                        }));

  TF_RETURN_IF_ERROR(SetValue<bool>(
      profile_options, "gpu_dump_graph_node_mapping", input_keys,
      [&](bool value) { collector_options.dump_graph_nope_mapping = value; }));

  TF_RETURN_IF_ERROR(SetValue<bool>(
      profile_options, "gpu_enable_nvtx_tracking", input_keys,
      [&](bool value) { tracer_options.enable_nvtx_tracking = value; }));

  TF_RETURN_IF_ERROR(
      SetValue<bool>(profile_options, "gpu_enable_cupti_activity_graph_trace",
                     input_keys, [&](bool value) {
                       if (value) {
                         tracer_options.activities_selected.push_back(
                             CUPTI_ACTIVITY_KIND_GRAPH_TRACE);
                       }
                     }));

  if (!input_keys.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Parsing advanced_configuration failed for CUPTI tracer. The following "
        "keys were not recognized: ",
        absl::StrJoin(input_keys, ",")));
  }

  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace xla
