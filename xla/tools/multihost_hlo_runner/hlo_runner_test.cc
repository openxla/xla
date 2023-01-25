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

#include "xla/tools/multihost_hlo_runner/hlo_runner.h"

#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_parser.h"
#include "xla/tests/literal_test_util.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::testing::Contains;
using ::testing::Key;
using ::testing::SizeIs;

TEST(SimpleMultiHostHloRunnerTest, TestSingleCore) {
  constexpr absl::string_view single_core_hlo = R"(
    HloModule hlo_runner_test_0.1

    ENTRY hlo_runner_test_0.1 {
      %lhs = f32[32,24,64,128] parameter(0)
      %rhs = f32[32,1024,64,128] parameter(1)
      ROOT %conv = f32[32,24,1024,1] convolution(%lhs, %rhs), dim_labels=0bf1_0oi1->0bf1, window={size=32x128 stride=31x1 lhs_dilate=32x1}
    }
  )";
  MultiHostHloRunner::Options options;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MultiHostHloRunner> hlo_runner,
                          MultiHostHloRunner::CreateMultiHostHloRunner(
                              options, MultiHostHloRunner::DeviceType::kGpu));
  auto run_status = hlo_runner->ParseAndRun(single_core_hlo);
  TF_EXPECT_OK(run_status.status());
}

TEST(MultiHostHloRunnerTest, TestRepeat) {
  constexpr absl::string_view single_core_hlo = R"(
    HloModule hlo_runner_test_0.1

    ENTRY hlo_runner_test_0.1 {
      %lhs = f32[32,24,64,128] parameter(0)
      %rhs = f32[32,1024,64,128] parameter(1)
      ROOT %conv = f32[32,24,1024,1] convolution(%lhs, %rhs), dim_labels=0bf1_0oi1->0bf1, window={size=32x128 stride=31x1 lhs_dilate=32x1}
    }
  )";
  MultiHostHloRunner::Options options{.num_repeats = 5};
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MultiHostHloRunner> hlo_runner,
                          MultiHostHloRunner::CreateMultiHostHloRunner(
                              options, MultiHostHloRunner::DeviceType::kGpu));
  auto run_status = hlo_runner->ParseAndRun(
      single_core_hlo, absl::btree_map<int, std::vector<Literal>>());
  TF_EXPECT_OK(run_status.status());
}

TEST(MultiHostHloRunnerTest, TestLoadFromText) {
  MultiHostHloRunner::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiHostHloRunner> hlo_runner,
      MultiHostHloRunner::CreateMultiHostHloRunner(
          options, /*device_type=*/MultiHostHloRunner::DeviceType::kGpu));
  auto run_status =
      hlo_runner->LoadAndRun({"third_party/tensorflow/compiler/xla/tools/"
                              "multihost_hlo_runner/test_data/test_hlo.txt"},
                             InputFormat::kText);
  TF_EXPECT_OK(run_status.status());
}

TEST(MultiHostHloRunnerTest, TestLoadFromProtoBinary) {
  MultiHostHloRunner::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiHostHloRunner> hlo_runner,
      MultiHostHloRunner::CreateMultiHostHloRunner(
          options, /*device_type=*/MultiHostHloRunner::DeviceType::kGpu));
  auto run_status =
      hlo_runner->LoadAndRun({"third_party/tensorflow/compiler/xla/tools/"
                              "multihost_hlo_runner/test_data/test_hlo.pb"},
                             InputFormat::kProtoBinary);
  TF_EXPECT_OK(run_status.status());
}

TEST(MultiHostHloRunnerTest, AdditionCorrectness) {
  constexpr absl::string_view addition_hlo = R"(
    HloModule hlo_runner_test_0.1

    ENTRY hlo_runner_test_0.1 {
      %lhs = f32[4,2] parameter(0)
      %rhs = f32[4,2] parameter(1)
      ROOT %conv = f32[4,2] add(%lhs, %rhs)
    }
  )";
  MultiHostHloRunner::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiHostHloRunner> hlo_runner,
      MultiHostHloRunner::CreateMultiHostHloRunner(
          options, /*device_type=*/MultiHostHloRunner::DeviceType::kGpu));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(addition_hlo));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                          hlo_runner->Compile(hlo_module.get()));
  ASSERT_THAT(executable->addressable_devices(), SizeIs(1));
  int device_id = executable->addressable_devices()[0]->id();

  Literal lhs = LiteralUtil::CreateR2<float>({{0, 1}, {2, 8}, {3, 2}, {4, -2}});
  Literal rhs =
      LiteralUtil::CreateR2<float>({{-1, 2}, {4, -2}, {0, 3}, {4, -3}});
  std::vector<Literal> arguments;
  arguments.push_back(std::move(lhs));
  arguments.push_back(std::move(rhs));
  absl::btree_map<int, std::vector<Literal>> device_arguments;
  device_arguments[device_id] = std::move(arguments);

  TF_ASSERT_OK_AND_ASSIGN(
      MultiHostHloRunner::PerDeviceLiteralVecType results,
      hlo_runner->CompileAndRun(hlo_module.get(), device_arguments));
  EXPECT_THAT(results, SizeIs(1));
  EXPECT_THAT(results, Contains(Key(device_id)));
  EXPECT_THAT(results[device_id], SizeIs(1));
  EXPECT_TRUE(LiteralTestUtil::Near(
      LiteralUtil::CreateR2<float>({{-1, 3}, {6, 6}, {3, 5}, {8, -5}}),
      results[device_id][0], ErrorSpec{1e-4, 1e-4}));
}

TEST(MultiHostHloRunnerTest, ArrayParametersWithAliasing) {
  constexpr absl::string_view addition_hlo = R"(
    HloModule hlo_runner_test_0.1, input_output_alias={ {}: (0, {}, may-alias) }

    ENTRY hlo_runner_test_0.1 {
      %lhs = f32[4,2] parameter(0)
      %rhs = f32[4,2] parameter(1)
      ROOT %conv = f32[4,2] add(%lhs, %rhs)
    }
  )";
  MultiHostHloRunner::Options options{.num_repeats = 3};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiHostHloRunner> hlo_runner,
      MultiHostHloRunner::CreateMultiHostHloRunner(
          options, /*device_type=*/MultiHostHloRunner::DeviceType::kGpu));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(addition_hlo));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                          hlo_runner->Compile(hlo_module.get()));
  ASSERT_THAT(executable->addressable_devices(), SizeIs(1));
  int device_id = executable->addressable_devices()[0]->id();

  Literal lhs = LiteralUtil::CreateR2<float>({{0, 1}, {2, 8}, {3, 2}, {4, -2}});
  Literal rhs =
      LiteralUtil::CreateR2<float>({{-1, 2}, {4, -2}, {0, 3}, {4, -3}});
  std::vector<Literal> arguments;
  arguments.push_back(std::move(lhs));
  arguments.push_back(std::move(rhs));
  absl::btree_map<int, std::vector<Literal>> device_arguments;
  device_arguments[device_id] = std::move(arguments);

  TF_ASSERT_OK_AND_ASSIGN(
      MultiHostHloRunner::PerDeviceLiteralVecType results,
      hlo_runner->CompileAndRun(hlo_module.get(), device_arguments));
  EXPECT_THAT(results, SizeIs(1));
  EXPECT_THAT(results, Contains(Key(device_id)));
  EXPECT_THAT(results[device_id], SizeIs(1));
  EXPECT_TRUE(LiteralTestUtil::Near(
      LiteralUtil::CreateR2<float>({{-3, 7}, {14, 2}, {3, 11}, {16, -11}}),
      results[device_id][0], ErrorSpec{1e-4, 1e-4}));
}

TEST(MultiHostHloRunnerTest, TupleParameterWithAliasing) {
  constexpr absl::string_view addition_hlo = R"(
    HloModule hlo_runner_test_0.1, input_output_alias={ {0}: (0, {0}, may-alias), {1}: (0, {1}, may-alias) }

    ENTRY hlo_runner_test_0.1 {
      %param = (f32[4,2], f32[4,2]) parameter(0)
      %lhs = f32[4,2] get-tuple-element(%param), index=0
      %rhs = f32[4,2] get-tuple-element(%param), index=1
      %add = f32[4,2] add(%lhs, %rhs)
      ROOT %tuple = (f32[4,2], f32[4,2]) tuple(%add, %rhs)
    }
  )";
  MultiHostHloRunner::Options options{.num_repeats = 3};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiHostHloRunner> hlo_runner,
      MultiHostHloRunner::CreateMultiHostHloRunner(
          options, /*device_type=*/MultiHostHloRunner::DeviceType::kGpu));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(addition_hlo));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                          hlo_runner->Compile(hlo_module.get()));
  ASSERT_THAT(executable->addressable_devices(), SizeIs(1));
  int device_id = executable->addressable_devices()[0]->id();

  Literal lhs = LiteralUtil::CreateR2<float>({{0, 1}, {2, 8}, {3, 2}, {4, -2}});
  Literal rhs =
      LiteralUtil::CreateR2<float>({{-1, 2}, {4, -2}, {0, 3}, {4, -3}});
  absl::btree_map<int, std::vector<Literal>> device_arguments;
  device_arguments[device_id].push_back(LiteralUtil::MakeTuple({&lhs, &rhs}));
  Literal add_result =
      LiteralUtil::CreateR2<float>({{-3, 7}, {14, 2}, {3, 11}, {16, -11}});
  Literal output = LiteralUtil::MakeTuple({&add_result, &rhs});

  TF_ASSERT_OK_AND_ASSIGN(
      MultiHostHloRunner::PerDeviceLiteralVecType results,
      hlo_runner->CompileAndRun(hlo_module.get(), device_arguments));
  EXPECT_THAT(results, SizeIs(1));
  EXPECT_THAT(results, Contains(Key(device_id)));
  EXPECT_THAT(results[device_id], SizeIs(1));
  EXPECT_TRUE(LiteralTestUtil::Near(output, results[device_id][0],
                                    ErrorSpec{1e-4, 1e-4}));
}

TEST(MultiHostHloRunnerTest, TupledParameterWithPartialAliasing) {
  constexpr absl::string_view addition_hlo = R"(
    HloModule hlo_runner_test_0.1, input_output_alias={ {0}: (0, {0}, may-alias) }

    ENTRY hlo_runner_test_0.1 {
      %param = (f32[4,2], f32[4,2]) parameter(0)
      %lhs = f32[4,2] get-tuple-element(%param), index=0
      %rhs = f32[4,2] get-tuple-element(%param), index=1
      %add = f32[4,2] add(%lhs, %rhs)
      ROOT %tuple = (f32[4,2], f32[4,2]) tuple(%add, %rhs)
    }
  )";
  MultiHostHloRunner::Options options{.num_repeats = 3};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiHostHloRunner> hlo_runner,
      MultiHostHloRunner::CreateMultiHostHloRunner(
          options, /*device_type=*/MultiHostHloRunner::DeviceType::kGpu));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(addition_hlo));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                          hlo_runner->Compile(hlo_module.get()));
  ASSERT_THAT(executable->addressable_devices(), SizeIs(1));
  int device_id = executable->addressable_devices()[0]->id();

  Literal lhs = LiteralUtil::CreateR2<float>({{0, 1}, {2, 8}, {3, 2}, {4, -2}});
  Literal rhs =
      LiteralUtil::CreateR2<float>({{-1, 2}, {4, -2}, {0, 3}, {4, -3}});
  absl::btree_map<int, std::vector<Literal>> device_arguments;
  device_arguments[device_id].push_back(LiteralUtil::MakeTuple({&lhs, &rhs}));
  Literal add_result =
      LiteralUtil::CreateR2<float>({{-3, 7}, {14, 2}, {3, 11}, {16, -11}});
  Literal output = LiteralUtil::MakeTuple({&add_result, &rhs});

  TF_ASSERT_OK_AND_ASSIGN(
      MultiHostHloRunner::PerDeviceLiteralVecType results,
      hlo_runner->CompileAndRun(hlo_module.get(), device_arguments));
  EXPECT_THAT(results, SizeIs(1));
  EXPECT_THAT(results, Contains(Key(device_id)));
  EXPECT_THAT(results[device_id], SizeIs(1));
  EXPECT_TRUE(LiteralTestUtil::Near(output, results[device_id][0],
                                    ErrorSpec{1e-4, 1e-4}));
}

TEST(MultiHostHloRunnerTest, TupledParameterWithoutAliasing) {
  constexpr absl::string_view addition_hlo = R"(
    HloModule hlo_runner_test_0.1

    ENTRY hlo_runner_test_0.1 {
      %param = (f32[4,2], f32[4,2]) parameter(0)
      %lhs = f32[4,2] get-tuple-element(%param), index=0
      %rhs = f32[4,2] get-tuple-element(%param), index=1
      %add = f32[4,2] add(%lhs, %rhs)
      ROOT %tuple = (f32[4,2], f32[4,2]) tuple(%add, %rhs)
    }
  )";
  MultiHostHloRunner::Options options{.num_repeats = 3};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiHostHloRunner> hlo_runner,
      MultiHostHloRunner::CreateMultiHostHloRunner(
          options, /*device_type=*/MultiHostHloRunner::DeviceType::kGpu));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(addition_hlo));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                          hlo_runner->Compile(hlo_module.get()));
  ASSERT_THAT(executable->addressable_devices(), SizeIs(1));
  int device_id = executable->addressable_devices()[0]->id();

  Literal lhs = LiteralUtil::CreateR2<float>({{0, 1}, {2, 8}, {3, 2}, {4, -2}});
  Literal rhs =
      LiteralUtil::CreateR2<float>({{-1, 2}, {4, -2}, {0, 3}, {4, -3}});
  absl::btree_map<int, std::vector<Literal>> device_arguments;
  device_arguments[device_id].push_back(LiteralUtil::MakeTuple({&lhs, &rhs}));
  Literal add_result =
      LiteralUtil::CreateR2<float>({{-1, 3}, {6, 6}, {3, 5}, {8, -5}});
  Literal output = LiteralUtil::MakeTuple({&add_result, &rhs});

  TF_ASSERT_OK_AND_ASSIGN(
      MultiHostHloRunner::PerDeviceLiteralVecType results,
      hlo_runner->CompileAndRun(hlo_module.get(), device_arguments));
  EXPECT_THAT(results, SizeIs(1));
  EXPECT_THAT(results, Contains(Key(device_id)));
  EXPECT_THAT(results[device_id], SizeIs(1));
  EXPECT_TRUE(LiteralTestUtil::Near(output, results[device_id][0],
                                    ErrorSpec{1e-4, 1e-4}));
}

}  // namespace
}  // namespace xla
