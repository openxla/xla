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

#include "xla/service/gpu_compilation_environment.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tsl/platform/errors.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ::tsl::testing::StatusIs;
using Flags = std::vector<Flag>;

TEST(CreateGpuCompilationEnvTest, ValidFlags) {
  Flags flags;
  flags.push_back({"xla_gpu_graph_level", "2"});

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuCompilationEnvironment> gpu_comp_env,
      CreateGpuCompEnvFromStringPairs(flags, /*strict=*/true));

  ASSERT_EQ(gpu_comp_env->xla_gpu_graph_level(), 2);
}

TEST(CreateGpuCompilationEnvTest, EmptyFlags) {
  Flags flags;

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuCompilationEnvironment> gpu_comp_env,
      CreateGpuCompEnvFromStringPairs(flags, /*strict=*/true))
}

TEST(CreateGpuCompilationEnvTest, InvalidFlagName) {
  Flags flags;
  flags.push_back({"xla_gpu_invalid_flag", "2"});

  EXPECT_THAT(CreateGpuCompEnvFromStringPairs(flags, /*strict=*/true),
              StatusIs(tsl::error::INVALID_ARGUMENT));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuCompilationEnvironment> gpu_comp_env,
      CreateGpuCompEnvFromStringPairs(flags, /*strict=*/false));
}

TEST(CreateGpuCompilationEnvTest, InvalidFlagValue) {
  Flags flags;
  flags.push_back({"xla_gpu_graph_level", "foo"});

  EXPECT_THAT(CreateGpuCompEnvFromStringPairs(flags, /*strict=*/true),
              StatusIs(tsl::error::INVALID_ARGUMENT));
}

}  // namespace
}  // namespace xla
