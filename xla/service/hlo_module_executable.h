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

#ifndef XLA_SERVICE_HLO_MODULE_EXECUTABLE_H_
#define XLA_SERVICE_HLO_MODULE_EXECUTABLE_H_

#include <memory>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/executable.h"

namespace xla {

// Base class for executables that keep the HloModule they were compiled from
// and answer metadata queries (name, module config, entry computation layout,
// input/output alias config, ...) from it.
//
// For now this only forwards to `Executable`. In a follow-up the HloModule
// related members of `Executable` move here, and `Executable` becomes an
// abstract interface without data members.
class HloModuleExecutable : public Executable {
 public:
  // The hlo_module parameter may be nullptr, if the given executable type
  // doesn't need it for execution.
  explicit HloModuleExecutable(std::shared_ptr<HloModule> hlo_module);
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_MODULE_EXECUTABLE_H_
