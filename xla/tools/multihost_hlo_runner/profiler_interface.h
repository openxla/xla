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

#ifndef XLA_TOOLS_MULTIHOST_HLO_RUNNER_PROFILER_INTERFACE_H_
#define XLA_TOOLS_MULTIHOST_HLO_RUNNER_PROFILER_INTERFACE_H_

namespace xla {

// Interface for profiler plugins. If being set in RunningOptions, profiling
// session will be created for the last run of the HLO module.
class ProfilerInterface {
 public:
  virtual ~ProfilerInterface() = default;
  // Creates profiling session while running HLO module.
  virtual void CreateSession() = 0;
  // Uploads profiling session data after finishing running HLO module.
  virtual void UploadSession() = 0;
  // Returns the number of replay passes required by the profiler (e.g. for
  // hardware counter collection that needs multiple passes over the same
  // workload). Only valid after CreateSession(). Returns 0 by default.
  virtual int GetNumRequiredPasses() const { return 0; }
};

}  // namespace xla

#endif  // XLA_TOOLS_MULTIHOST_HLO_RUNNER_PROFILER_INTERFACE_H_
