/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/executable.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tsl/platform/env.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_execution_profile.h"
#include "xla/service/service_executable_run_options.h"

namespace xla {
namespace {

class TestExecutable : public Executable {
 public:
  explicit TestExecutable(std::shared_ptr<HloModule> module)
      : Executable{std::move(module)} {}

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override {
    return absl::UnimplementedError("Not needed for this test.");
  }
};

class ExecutableTest : public HloHardwareIndependentTestBase {};

TEST_F(ExecutableTest, HloProtoGetterIsThreadCompatible) {
  // Executable::hlo_proto() is doing some lazy initialization of a
  // part of `hlo_proto_`. This test ensures that this is done in a
  // thread-compatible way.
  // Note that this test needs to run with --config=tsan to reliably
  // detect any potential data races.
  constexpr absl::string_view kHloModule = R"(
    HloModule module

    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));

  TestExecutable executable(module);

  auto proto = std::make_unique<HloProto>();
  executable.set_hlo_proto(std::move(proto));

  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "test",
                                 /*num_threads=*/2);
    for (int i = 0; i < 2; ++i) {
      pool.Schedule([&] { executable.hlo_proto()->SerializeAsString(); });
    }
  }
}

constexpr absl::string_view kModuleWithAliasing = R"(
  HloModule module, input_output_alias={ {0}: (0, {}, must-alias) }, buffer_donor={ (1, {}) }

  ENTRY main {
    p0 = f32[4] parameter(0)
    p1 = f32[4] parameter(1)
    add = f32[4] add(p0, p1)
    ROOT result = (f32[4], f32[4]) tuple(add, p1)
  }
)";

TEST_F(ExecutableTest, MetadataDefaultsReadTheModule) {
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleWithAliasing));
  TestExecutable executable(module);

  EXPECT_TRUE(executable.has_module());
  EXPECT_EQ(&executable.module(), module.get());
  EXPECT_EQ(executable.shared_module(), module);
  EXPECT_EQ(&executable.module_config(), &module->config());
  EXPECT_EQ(&executable.input_output_alias_config(),
            &module->input_output_alias_config());
  EXPECT_EQ(&executable.buffer_donor_config(), &module->buffer_donor_config());
  EXPECT_EQ(executable.name(), "module");
  EXPECT_EQ(executable.result_shape(),
            module->config().entry_computation_layout().result_shape());
  // Make sure that the test doesn't compare empty configs.
  EXPECT_TRUE(executable.input_output_alias_config().OutputHasAlias({0}));
  EXPECT_FALSE(executable.buffer_donor_config().buffer_donor().empty());
}

TEST_F(ExecutableTest, WithoutModule) {
  TestExecutable executable(nullptr);

  EXPECT_FALSE(executable.has_module());
  EXPECT_EQ(executable.shared_module(), nullptr);
  EXPECT_EQ(executable.name(), "<unknown executable>");
}

// An executable that keeps its module itself, like `GpuExecutable`.
class ExecutableWithOwnModule : public TestExecutable {
 public:
  explicit ExecutableWithOwnModule(std::shared_ptr<HloModule> module)
      : TestExecutable(nullptr), module_(std::move(module)) {}

  std::shared_ptr<HloModule> shared_module() const override { return module_; }

 protected:
  HloModule* module_ptr() const override { return module_.get(); }

 private:
  std::shared_ptr<HloModule> module_;
};

TEST_F(ExecutableTest, ModuleAccessorsUseTheOverride) {
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleWithAliasing));
  ExecutableWithOwnModule executable(module);

  EXPECT_TRUE(executable.has_module());
  EXPECT_EQ(&executable.module(), module.get());
  EXPECT_EQ(executable.shared_module(), module);
  EXPECT_EQ(&executable.module_config(), &module->config());
  EXPECT_EQ(&executable.input_output_alias_config(),
            &module->input_output_alias_config());
  EXPECT_EQ(&executable.buffer_donor_config(), &module->buffer_donor_config());
  EXPECT_EQ(executable.name(), "module");
}

}  // namespace
}  // namespace xla
