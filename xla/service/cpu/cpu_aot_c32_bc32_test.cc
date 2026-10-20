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

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class CpuAotC32Bc32Test : public HloHardwareIndependentTestBase {};

TEST_F(CpuAotC32Bc32Test, AotCompilationRejectsC32) {
  absl::string_view text = R"(
HloModule c32_aot
ENTRY main {
  ROOT root = c32[2] constant({(0.5, -0.25), (1.5, 2)})
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(text));
  cpu::CpuCompiler cpu_compiler;

  cpu::CpuAotCompilationOptions aot_compilation_options(
      /*triple=*/"i686-none-android", /*cpu_name=*/"", /*features=*/"",
      /*entry_point_name=*/"main",
      cpu::CpuAotCompilationOptions::RelocationModel::BigPic);

  absl::StatusOr<std::vector<std::unique_ptr<CompiledModule>>> result =
      cpu_compiler.CompileAheadOfTime(std::move(hlo_module),
                                      aot_compilation_options);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_NE(result.status().message().find("c32"), std::string::npos)
      << result.status();
}

TEST_F(CpuAotC32Bc32Test, AotCompilationRejectsBc32) {
  absl::string_view text = R"(
HloModule bc32_aot
ENTRY main {
  ROOT root = bc32[2] constant({(0.5, -0.25), (1.5, 2)})
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(text));
  cpu::CpuCompiler cpu_compiler;

  cpu::CpuAotCompilationOptions aot_compilation_options(
      /*triple=*/"i686-none-android", /*cpu_name=*/"", /*features=*/"",
      /*entry_point_name=*/"main",
      cpu::CpuAotCompilationOptions::RelocationModel::BigPic);

  absl::StatusOr<std::vector<std::unique_ptr<CompiledModule>>> result =
      cpu_compiler.CompileAheadOfTime(std::move(hlo_module),
                                      aot_compilation_options);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_NE(result.status().message().find("bc32"), std::string::npos)
      << result.status();
}

}  // namespace
}  // namespace xla
