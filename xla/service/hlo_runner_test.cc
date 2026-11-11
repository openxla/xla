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

#include "xla/service/hlo_runner.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "google/protobuf/message.h"
#include "google/protobuf/message_lite.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/path.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/scoped_async_tracking_event.h"
#include "xla/runtime/chip_id.h"
#include "xla/runtime/device_id.h"
#include "xla/service/device_assignment.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/test_compilation_environment.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/util.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::Optional;
using ::testing::StartsWith;

absl::StatusOr<std::unique_ptr<HloModule>> CreateFakeModule() {
  return ParseAndReturnUnverifiedModule(R"(
HloModule constant_s32_module, entry_computation_layout={()->s32[]}

ENTRY %constant_s32 () -> s32[] {
  ROOT %constant = s32[] constant(-42)
}
)");
}

class FakeClient : public PjRtClient {
 public:
  using ExecuteShardedFn =
      std::function<absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>(
          PjRtDevice*)>;

  class Device : public PjRtDevice {
   public:
    explicit Device(int id)
        : id_(id), debug_string_(absl::StrCat("FakeDevice(", id, ")")) {}

    PjRtClient* client() const override { return nullptr; }
    bool IsAddressable() const override { return true; }
    LocalChipId local_hardware_id() const override { return LocalChipId(id_); }
    absl::string_view DebugString() const override { return debug_string_; }
    std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
        absl::string_view description) const override {
      return nullptr;
    }
    absl::Status TransferToInfeed(const LiteralSlice& literal) override {
      return absl::UnimplementedError("Not implemented.");
    }
    absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override {
      return absl::UnimplementedError("Not implemented.");
    }
    absl::Span<PjRtMemorySpace* const> memory_spaces() const override {
      return {};
    }
    absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override {
      return absl::UnimplementedError("Not implemented.");
    }

   private:
    int id_;
    std::string debug_string_;
  };

  class Executable : public PjRtExecutable {
   public:
    explicit Executable(CompileOptions compile_options = {})
        : compile_options_(std::move(compile_options)) {}

    int num_replicas() const override { return 1; }
    int num_partitions() const override { return 1; }
    int64_t SizeOfGeneratedCodeInBytes() const override { return 0; }
    absl::string_view name() const override { return "FakeExecutable"; }
    absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
        const override {
      ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                            CreateFakeModule());
      return std::vector<std::shared_ptr<HloModule>>{std::move(module)};
    }
    absl::StatusOr<CompileOptions> GetCompileOptions() const override {
      return compile_options_;
    }
    absl::StatusOr<std::vector<std::vector<absl::string_view>>>
    GetParameterMemoryKinds() const override {
      return absl::UnimplementedError(
          "GetParameterMemoryKinds is not implemented.");
    }
    absl::StatusOr<std::vector<std::vector<absl::string_view>>>
    GetOutputMemoryKinds() const override {
      return absl::UnimplementedError(
          "GetOutputMemoryKinds is not implemented.");
    }
    absl::StatusOr<std::string> SerializeExecutable() const override {
      return "serialized executable";
    }

   private:
    CompileOptions compile_options_;
  };

  class LoadedExecutable : public PjRtLoadedExecutable {
   public:
    LoadedExecutable(std::shared_ptr<PjRtExecutable> executable,
                     ExecuteShardedFn execute_sharded_fn)
        : executable_(std::move(executable)),
          execute_sharded_fn_(std::move(execute_sharded_fn)) {}

    PjRtClient* client() const override { return nullptr; }
    PjRtExecutable* GetExecutable() const override { return executable_.get(); }
    const DeviceAssignment& device_assignment() const override {
      return device_assignment_;
    }
    absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
        const override {
      return {};
    }
    absl::Span<PjRtDevice* const> addressable_devices() const override {
      return {};
    }
    absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
    Execute(
        absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
        const ExecuteOptions& options,
        std::optional<std::vector<Future<>>>& returned_futures) const override {
      return absl::UnimplementedError("Not implemented.");
    }
    absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
        absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
        const ExecuteOptions& options, std::optional<Future<>>& returned_future,
        bool fill_future) const override {
      return execute_sharded_fn_(device);
    }
    absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
        absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
        const ExecuteOptions& options, std::optional<Future<>>& returned_future,
        bool fill_future) const override {
      return absl::UnimplementedError("Not implemented.");
    }
    void Delete() override {}
    bool IsDeleted() const override { return false; }

   private:
    std::shared_ptr<PjRtExecutable> executable_;
    ExecuteShardedFn execute_sharded_fn_;
    DeviceAssignment device_assignment_;
  };

  FakeClient() : deserialize_callback_([](absl::string_view) {}) {
    owned_devices_.push_back(std::make_unique<Device>(0));
    owned_devices_.push_back(std::make_unique<Device>(1));
  }
  explicit FakeClient(
      std::function<void(absl::string_view)> deserialize_callback)
      : FakeClient() {
    deserialize_callback_ = std::move(deserialize_callback);
  }
  explicit FakeClient(ExecuteShardedFn execute_sharded_fn) : FakeClient() {
    execute_sharded_fn_ = std::move(execute_sharded_fn);
  }

  int process_index() const override { return 0; }
  int device_count() const override { return owned_devices_.size(); }
  int addressable_device_count() const override {
    return owned_devices_.size();
  }
  // HloRunner resolves devices via LookupDevice rather than devices().
  absl::Span<PjRtDevice* const> devices() const override { return {}; }
  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return {};
  }
  absl::StatusOr<PjRtDevice*> LookupDevice(
      GlobalDeviceId global_device_id) const override {
    if (global_device_id.value() < 0 ||
        global_device_id.value() >= owned_devices_.size()) {
      return absl::InvalidArgumentError("Invalid device ID");
    }
    return owned_devices_[global_device_id.value()].get();
  }
  absl::Span<PjRtMemorySpace* const> memory_spaces() const override {
    return {};
  }
  PjRtPlatformId platform_id() const override {
    return tsl::Fingerprint64(platform_name_);
  }
  absl::string_view platform_name() const override { return platform_name_; }
  absl::string_view platform_version() const override { return "0.0.0"; }

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override {
    DeviceAssignment assignment(num_replicas, num_partitions);
    assignment.FillIota(0);
    return assignment;
  }

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override {
    return std::make_unique<Executable>(std::move(options));
  }

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> DeserializeExecutable(
      absl::string_view serialized,
      std::optional<CompileOptions> options) override {
    deserialize_callback_(serialized);
    return std::make_unique<Executable>();
  }

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Load(
      std::shared_ptr<PjRtExecutable> executable,
      const LoadOptions& load_options) override {
    return std::make_unique<LoadedExecutable>(std::move(executable),
                                              execute_sharded_fn_);
  }

 private:
  std::function<void(absl::string_view)> deserialize_callback_;
  ExecuteShardedFn execute_sharded_fn_;
  std::vector<std::unique_ptr<Device>> owned_devices_;
  std::string platform_name_ = "fake";
  PjRtPlatformId platform_id_ = tsl::Fingerprint64(platform_name_);
};

// Fake module with run_hlo_passes=false.
constexpr absl::string_view kModuleSerializedName =
    "a83d1d9a594b0f1fbc4227408abcdc6a.bin.gz";
// Fake module with run_hlo_passes=true.
constexpr absl::string_view kModuleWithRunHloPassesSerializedName =
    "1f51ba5f389ad07cbe268573dd21a94e.bin.gz";
// Fake module with a compilation environment set.
constexpr absl::string_view kModuleWithCompEnvSerializedName =
    "9385c25a58e7d6d47e56af1dd950b7d1.bin.gz";

class ArtifactDirTest : public ::testing::Test {
 public:
  void SetUp() override {
    ASSERT_OK(tsl::Env::Default()->CreateDir(artifact_dir_));
  }
  void TearDown() override {
    int64_t num_files_deleted = 0;
    int64_t num_dirs_deleted = 0;
    ASSERT_OK(tsl::Env::Default()->DeleteRecursively(
        artifact_dir_, &num_files_deleted, &num_dirs_deleted));
  }

  const std::string artifact_dir_ =
      tsl::io::JoinPath(testing::TempDir(), "artifact_dir");
};

using CompilePhaseHloRunnerTest = ArtifactDirTest;

// Tests that a call to CreateExecutable places the file in the right location.
TEST_F(CompilePhaseHloRunnerTest, CreateExecutablePlacesFileCorrectly) {
  CompilePhaseHloRunner runner(std::make_unique<FakeClient>(), artifact_dir_);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m, CreateFakeModule());
  ASSERT_OK(
      runner.CreateExecutable(std::move(m), /*run_hlo_passes=*/false).status());

  std::vector<std::string> children;
  ASSERT_OK(tsl::Env::Default()->GetChildren(artifact_dir_, &children));
  ASSERT_EQ(children.size(), 1);
  ASSERT_EQ(children[0], kModuleSerializedName);
}

// Tests that a CreateExecutable call with different run_hlo_passes value places
// the file in a different location.
TEST_F(CompilePhaseHloRunnerTest,
       CreateExecutablePlacesFilesCorrectlyWithDifferentRunHloPasses) {
  CompilePhaseHloRunner runner(std::make_unique<FakeClient>(), artifact_dir_);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m, CreateFakeModule());
  ASSERT_OK(runner.CreateExecutable(std::move(m), /*run_hlo_passes=*/true));

  std::vector<std::string> children;
  ASSERT_OK(tsl::Env::Default()->GetChildren(artifact_dir_, &children));
  ASSERT_EQ(children.size(), 1);
  ASSERT_EQ(children[0], kModuleWithRunHloPassesSerializedName);
}

// Tests that a CreateExecutable call with a different compilation
// environment places the file in a different location.
TEST_F(CompilePhaseHloRunnerTest,
       CreateExecutablePlacesFilesCorrectlyWithCompilationEnvironment) {
  CompilePhaseHloRunner runner(std::make_unique<FakeClient>(), artifact_dir_);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m, CreateFakeModule());
  m->comp_envs().RegisterProcessNewEnvFn(
      test::TestCompilationEnvironment1::GetDescriptor(),
      [](std::unique_ptr<tsl::protobuf::Message> msg) {
        std::unique_ptr<test::TestCompilationEnvironment1> env(
            google::protobuf::DownCastMessage<
                test::TestCompilationEnvironment1>(msg.release()));
        if (env == nullptr) {
          env = std::make_unique<test::TestCompilationEnvironment1>();
        }
        env->set_some_flag(42);
        return env;
      });
  ASSERT_OK(m->comp_envs().InitializeAllKnownEnvs());
  ASSERT_OK(runner.CreateExecutable(std::move(m), /*run_hlo_passes=*/false));

  std::vector<std::string> children;
  ASSERT_OK(tsl::Env::Default()->GetChildren(artifact_dir_, &children));
  ASSERT_EQ(children.size(), 1);
  ASSERT_EQ(children[0], kModuleWithCompEnvSerializedName);
}

// Tests that a CreateExecutable call with a different device assignment
// places the file in a different location.
TEST_F(CompilePhaseHloRunnerTest,
       CreateExecutablePlacesFilesCorrectlyWithDeviceAssignment) {
  CompilePhaseHloRunner runner(std::make_unique<FakeClient>(), artifact_dir_);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m1, CreateFakeModule());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m2, CreateFakeModule());

  DeviceAssignment da1(1, 2);
  da1.FillIota(0);
  m1->mutable_config().set_static_device_assignment(da1);

  DeviceAssignment da2(1, 2);
  da2.FillIota(1);
  m2->mutable_config().set_static_device_assignment(da2);

  ASSERT_OK(runner.CreateExecutable(std::move(m1), /*run_hlo_passes=*/false));
  ASSERT_OK(runner.CreateExecutable(std::move(m2), /*run_hlo_passes=*/false));

  std::vector<std::string> children;
  ASSERT_OK(tsl::Env::Default()->GetChildren(artifact_dir_, &children));
  EXPECT_EQ(children.size(), 2);
}

using ExecutePhaseHloRunnerTest = ArtifactDirTest;

// Tests that a call to CreateExecutable reads the file from the correct path
// and deserializes the right contents.
TEST_F(ExecutePhaseHloRunnerTest, CreateExecutableReadsFileCorrectly) {
  ASSERT_OK(CompilePhaseHloRunner::WriteCompressedExecutable(
      tsl::io::JoinPath(artifact_dir_, kModuleSerializedName), "hello world"));
  absl::Notification notification;
  std::optional<std::string> serialized_representation_read = std::nullopt;
  ExecutePhaseHloRunner runner(
      std::make_unique<FakeClient>(
          [&notification,
           &serialized_representation_read](absl::string_view serialized) {
            serialized_representation_read = serialized;
            notification.Notify();
          }),
      artifact_dir_);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m, CreateFakeModule());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OpaqueExecutable> executable,
      runner.CreateExecutable(std::move(m), /*run_hlo_passes=*/false));

  ASSERT_TRUE(notification.WaitForNotificationWithTimeout(absl::Seconds(5)));
  ASSERT_TRUE(serialized_representation_read.has_value());
  ASSERT_EQ(*serialized_representation_read, "hello world");
}

TEST_F(ExecutePhaseHloRunnerTest,
       CreateExecutableFailsOnDuplicateLoadIfFeatureEnabled) {
  ASSERT_OK(CompilePhaseHloRunner::WriteCompressedExecutable(
      tsl::io::JoinPath(artifact_dir_, kModuleSerializedName), "hello world"));
  ExecutePhaseHloRunner runner(std::make_unique<FakeClient>(), artifact_dir_);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m, CreateFakeModule());
  ASSERT_OK(runner.CreateExecutable(m->Clone(""), /*run_hlo_passes=*/false));
  EXPECT_THAT(
      runner.CreateExecutable(std::move(m), /*run_hlo_passes=*/false),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          StartsWith(
              "ExecutePhaseHloRunner::CreateExecutable called with a "
              "module that loads an executable that was previously loaded.")));
}

TEST_F(ExecutePhaseHloRunnerTest,
       CreateExecutableSucceedsOnDuplicateLoadIfFeatureDisabled) {
  ASSERT_OK(CompilePhaseHloRunner::WriteCompressedExecutable(
      tsl::io::JoinPath(artifact_dir_, kModuleSerializedName), "hello world"));
  ExecutePhaseHloRunner runner(std::make_unique<FakeClient>(), artifact_dir_,
                               /*compile_if_not_found=*/false,
                               /*fail_duplicate_loads=*/false);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m, CreateFakeModule());
  ASSERT_OK(runner.CreateExecutable(m->Clone(""), /*run_hlo_passes=*/false));
  EXPECT_OK(runner.CreateExecutable(std::move(m), /*run_hlo_passes=*/false));
}

TEST(HloRunnerTest, ExecuteReplicatedAggregatesMultiDeviceErrors) {
  HloRunner runner(std::make_unique<FakeClient>(
      [](PjRtDevice* device)
          -> absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> {
        if (device->local_hardware_id().value() == 0) {
          absl::Status err = absl::InternalError("device 0 error");
          err.SetPayload("type.googleapis.com/test.Payload",
                         absl::Cord("payload_0"));
          return err;
        }
        absl::Status err = absl::UnavailableError("device 1 error");
        err.SetPayload("type.googleapis.com/test.OtherPayload",
                       absl::Cord("payload_1"));
        return err;
      }));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m, CreateFakeModule());
  m->mutable_config().set_replica_count(2);
  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_devices = 2;

  absl::Status status =
      runner.ExecuteReplicated(std::move(m), options).status();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal,
                               "Execution failed on 2 devices:\n\n"
                               "FakeDevice(0): device 0 error\n\n"
                               "FakeDevice(1): device 1 error"));
  EXPECT_THAT(status.GetPayload("type.googleapis.com/test.Payload"),
              Optional(absl::Cord("payload_0")));
}

TEST(HloRunnerTest, ExecuteReplicatedSingleDeviceErrorNotWrapped) {
  HloRunner runner(std::make_unique<FakeClient>(
      [](PjRtDevice* device)
          -> absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> {
        if (device->local_hardware_id().value() == 1) {
          return absl::InternalError("device 1 error");
        }
        return std::vector<std::unique_ptr<PjRtBuffer>>{};
      }));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m, CreateFakeModule());
  m->mutable_config().set_replica_count(2);
  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_devices = 2;

  EXPECT_THAT(runner.ExecuteReplicated(std::move(m), options),
              StatusIs(absl::StatusCode::kInternal, "device 1 error"));
}

}  // namespace
}  // namespace xla
