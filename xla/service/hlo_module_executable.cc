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

#include "xla/service/hlo_module_executable.h"

#include <memory>
#include <utility>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/executable.h"

namespace xla {

HloModuleExecutable::HloModuleExecutable(std::shared_ptr<HloModule> hlo_module)
    : Executable(std::move(hlo_module)) {}

}  // namespace xla
