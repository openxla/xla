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
#ifndef XLA_BACKENDS_PROFILER_PLUGIN_PLUGIN_TRACER_H_
#define XLA_BACKENDS_PROFILER_PLUGIN_PLUGIN_TRACER_H_

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_profiler_extension.h"
#include "xla/status.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

// Plugin implementation of ProfilerInterface.
//
// Thread-safety: This class is go/thread-compatible.
class PluginTracer : public tsl::profiler::ProfilerInterface {
 public:
  explicit PluginTracer(const PJRT_Api* pjrt_api);
  ~PluginTracer() override;

  Status Start() override;

  Status Stop() override;

  Status CollectData(tensorflow::profiler::XSpace* space) override;

 private:
  const PJRT_Api* pjrt_api_;
  const PJRT_Profiler_Extension* profiler_api_;
  PJRT_Profiler* profiler_;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_PLUGIN_PLUGIN_TRACER_H_
