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

#include "xla/pjrt/se/stream_executor_executable.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "riegeli/bytes/string_reader.h"
#include "tsl/platform/fingerprint.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"
#include "xla/pjrt/se/stream_executor_platform_id_mapping.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/mock_compiled_module.h"
#include "xla/shape.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/executable_abi_version.pb.h"
#include "xla/stream_executor/platform_id.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::Return;
using ::tsl::proto_testing::EqualsProto;

TEST(StreamExecutorExecutableTest, GetAbiVersion) {
  stream_executor::ExecutableAbiVersionProto executable_abi_version_proto;
  executable_abi_version_proto.mutable_cuda_platform_version()
      ->set_cuda_toolkit_version("1.2.3");
  ASSERT_OK_AND_ASSIGN(
      stream_executor::ExecutableAbiVersion executable_abi_version,
      stream_executor::ExecutableAbiVersion::FromProto(
          executable_abi_version_proto));

  constexpr PjRtPlatformId kPlatformId = 42;

  auto module = std::make_unique<MockCompiledModule>();
  auto hlo_module = std::make_shared<HloModule>("name", HloModuleConfig());
  EXPECT_CALL(*module, shared_optimized_module())
      .WillRepeatedly(Return(hlo_module));
  EXPECT_CALL(*module, GetExecutableAbiVersion())
      .WillOnce(Return(executable_abi_version));
  StreamExecutorExecutable executable(kPlatformId, CompileOptions(),
                                      std::move(module), 0, 0, "name",
                                      "fingerprint", "memory_kind");

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtExecutableAbiVersion> abi_version,
                       executable.GetAbiVersion());
  EXPECT_EQ(abi_version->platform_id(), kPlatformId);

  ASSERT_OK_AND_ASSIGN(PjRtExecutableAbiVersionProto proto,
                       abi_version->ToProto());
  stream_executor::ExecutableAbiVersionProto
      reconstructed_executable_abi_version_proto;
  EXPECT_TRUE(reconstructed_executable_abi_version_proto.ParseFromString(
      proto.version()));
  EXPECT_THAT(reconstructed_executable_abi_version_proto,
              EqualsProto(executable_abi_version_proto));
}

PLATFORM_DEFINE_ID(kFakeAotSePlatformId, fake_aot_se_platform);
constexpr PjRtPlatformId kFakeAotPjRtPlatformId = 424242;

class FakeAotCompiler : public Compiler {
 public:
  stream_executor::Platform::Id PlatformId() const override {
    return kFakeAotSePlatformId;
  }
  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule>, stream_executor::StreamExecutor*,
      const CompileOptions&) override {
    return absl::UnimplementedError("Unimplemented");
  }
  absl::StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule>, stream_executor::StreamExecutor*,
      const CompileOptions&) override {
    return absl::UnimplementedError("Unimplemented");
  }
  absl::StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModule>, std::vector<stream_executor::StreamExecutor*>,
      const CompileOptions&) override {
    return absl::UnimplementedError("Unimplemented");
  }
  absl::StatusOr<std::vector<std::unique_ptr<CompiledModule>>>
  CompileAheadOfTime(std::unique_ptr<HloModule>,
                     const AotCompilationOptions&) override {
    return absl::UnimplementedError("Unimplemented");
  }
  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override {
    return [](const Shape&) { return int64_t{0}; };
  }
  absl::StatusOr<std::unique_ptr<CompiledModule>> LoadAotCompilationResult(
      const std::string& serialized_aot_result) override {
    if (serialized_aot_result != "fake_serialized_aot_module") {
      return absl::InvalidArgumentError("Unexpected serialized payload");
    }
    auto module = std::make_unique<MockCompiledModule>();
    auto hlo_module = std::make_shared<HloModule>("name", HloModuleConfig());
    EXPECT_CALL(*module, shared_optimized_module())
        .WillRepeatedly(Return(hlo_module));
    return module;
  }
};

void RegisterFakeAotCompilerOnce() {
  static absl::once_flag once;
  absl::call_once(once, []() {
    CHECK_OK(StreamExecutorPlatformIdMapping::Global().AddMapping(
        kFakeAotSePlatformId, kFakeAotPjRtPlatformId));
    Compiler::RegisterCompilerFactory(
        kFakeAotSePlatformId,
        []() -> absl::StatusOr<std::unique_ptr<Compiler>> {
          return std::make_unique<FakeAotCompiler>();
        });
  });
}

class FakeTopologyDescription : public PjRtTopologyDescription {
 public:
  PjRtPlatformId platform_id() const override { return kFakeAotPjRtPlatformId; }
  absl::string_view platform_name() const override { return "test"; }
  absl::string_view platform_version() const override { return "1.0"; }
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions()
      const override {
    return {};
  }
  absl::StatusOr<uint64_t> Fingerprint() const override { return 0; }
  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }
  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override {
    return absl::UnimplementedError("Unimplemented");
  }
  absl::Span<const int> GetMemorySpaceKindIds() const override {
    static const int kMemorySpaceKindId =
        static_cast<int>(tsl::Fingerprint32("device"));
    return absl::MakeConstSpan(&kMemorySpaceKindId, 1);
  }

 private:
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_;
};

absl::StatusOr<std::string> SerializeFakeExecutable(
    const CompileOptions& compile_options) {
  RegisterFakeAotCompilerOnce();
  auto module = std::make_unique<MockCompiledModule>();
  auto hlo_module = std::make_shared<HloModule>("name", HloModuleConfig());
  EXPECT_CALL(*module, shared_optimized_module())
      .WillRepeatedly(Return(hlo_module));
  EXPECT_CALL(*module, SerializeAsString())
      .WillOnce(Return(std::string("fake_serialized_aot_module")));

  StreamExecutorExecutable executable(kFakeAotPjRtPlatformId, compile_options,
                                      std::move(module),
                                      /*num_replicas=*/1, /*num_partitions=*/1,
                                      "name", "fingerprint", "device");
  return executable.SerializeExecutable();
}

TEST(StreamExecutorExecutableTest,
     DeserializeOverridesExecutionTimeoutsFromRuntimeFlags) {
  CompileOptions compile_options;
  DebugOptions* build_debug_options =
      compile_options.executable_build_options.mutable_debug_options();
  build_debug_options->set_xla_gpu_execution_terminate_timeout("123s");
  build_debug_options->set_xla_gpu_device_execution_terminate_timeout("456s");
  build_debug_options->set_xla_gpu_execution_progress_tracking(7);
  build_debug_options->set_xla_gpu_executable_warn_stuck_timeout_seconds(99);
  build_debug_options->set_xla_gpu_executable_terminate_timeout_seconds(199);
  // Compile-time env_option_overrides for runtime flags must not override
  // serving flags, whereas non-runtime compile-time overrides are preserved.
  compile_options.env_option_overrides = {
      {"xla_llvm_enable_alias_scope_metadata", false},
      {"xla_gpu_execution_terminate_timeout", std::string("789s")},
      {"xla_gpu_execution_progress_tracking", int64_t{42}},
  };

  ASSERT_OK_AND_ASSIGN(std::string serialized,
                       SerializeFakeExecutable(compile_options));

  FakeTopologyDescription topology;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<StreamExecutorExecutable> deserialized,
                       StreamExecutorExecutable::Deserialize(
                           riegeli::StringReader<>(serialized), topology,
                           /*options=*/std::nullopt));

  const DebugOptions& expected_flag_options = GetDebugOptionsFromFlags();
  const DebugOptions& deserialized_debug_options =
      deserialized->compile_options().executable_build_options.debug_options();
  EXPECT_FALSE(
      deserialized_debug_options.xla_llvm_enable_alias_scope_metadata());
  EXPECT_EQ(deserialized_debug_options.xla_gpu_execution_terminate_timeout(),
            expected_flag_options.xla_gpu_execution_terminate_timeout());
  EXPECT_EQ(
      deserialized_debug_options.xla_gpu_device_execution_terminate_timeout(),
      expected_flag_options.xla_gpu_device_execution_terminate_timeout());
  EXPECT_EQ(deserialized_debug_options.xla_gpu_execution_progress_tracking(),
            expected_flag_options.xla_gpu_execution_progress_tracking());
  EXPECT_EQ(
      deserialized_debug_options
          .xla_gpu_executable_warn_stuck_timeout_seconds(),
      expected_flag_options.xla_gpu_executable_warn_stuck_timeout_seconds());
  EXPECT_EQ(
      deserialized_debug_options.xla_gpu_executable_terminate_timeout_seconds(),
      expected_flag_options.xla_gpu_executable_terminate_timeout_seconds());
}

TEST(StreamExecutorExecutableTest,
     DeserializeWithoutBuildDebugOptionsAppliesEnvAndRuntimeOverrides) {
  CompileOptions compile_options;
  compile_options.env_option_overrides = {
      {"xla_llvm_enable_alias_scope_metadata", false},
      {"xla_gpu_execution_terminate_timeout", std::string("789s")},
      {"xla_gpu_execution_progress_tracking", int64_t{42}},
  };
  ASSERT_FALSE(compile_options.executable_build_options.has_debug_options());

  ASSERT_OK_AND_ASSIGN(std::string serialized,
                       SerializeFakeExecutable(compile_options));

  FakeTopologyDescription topology;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<StreamExecutorExecutable> deserialized,
                       StreamExecutorExecutable::Deserialize(
                           riegeli::StringReader<>(serialized), topology,
                           /*options=*/std::nullopt));

  const DebugOptions& expected_flag_options = GetDebugOptionsFromFlags();
  ASSERT_TRUE(deserialized->compile_options()
                  .executable_build_options.has_debug_options());
  const DebugOptions& deserialized_debug_options =
      deserialized->compile_options().executable_build_options.debug_options();
  EXPECT_FALSE(
      deserialized_debug_options.xla_llvm_enable_alias_scope_metadata());
  EXPECT_EQ(deserialized_debug_options.xla_gpu_execution_terminate_timeout(),
            expected_flag_options.xla_gpu_execution_terminate_timeout());
  EXPECT_EQ(deserialized_debug_options.xla_gpu_execution_progress_tracking(),
            expected_flag_options.xla_gpu_execution_progress_tracking());
}

TEST(StreamExecutorExecutableTest,
     DeserializeWithExplicitOptionsHonorsCallerOverrides) {
  CompileOptions compile_options;
  compile_options.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_execution_terminate_timeout("123s");

  ASSERT_OK_AND_ASSIGN(std::string serialized,
                       SerializeFakeExecutable(compile_options));

  CompileOptions load_options;
  load_options.env_option_overrides = {
      {"xla_gpu_execution_terminate_timeout", std::string("300s")},
      {"xla_gpu_execution_progress_tracking", int64_t{5}},
  };

  FakeTopologyDescription topology;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StreamExecutorExecutable> deserialized,
      StreamExecutorExecutable::Deserialize(riegeli::StringReader<>(serialized),
                                            topology, std::move(load_options)));

  const DebugOptions& deserialized_debug_options =
      deserialized->compile_options().executable_build_options.debug_options();
  EXPECT_EQ(deserialized_debug_options.xla_gpu_execution_terminate_timeout(),
            "300s");
  EXPECT_EQ(deserialized_debug_options.xla_gpu_execution_progress_tracking(),
            5);
}

}  // namespace
}  // namespace xla
