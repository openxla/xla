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

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/executable.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

class FakeCodegenBackend : public CodegenBackend {
 public:
  FakeCodegenBackend(autotuner::Backend backend, std::string version)
      : backend_(backend), version_(version) {}

  absl::string_view name() const override { return "fake"; }
  autotuner::Backend backend() const override { return backend_; }
  std::string version() const override { return version_; }

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) override {
    return std::vector<std::unique_ptr<BackendConfig>>();
  }
  absl::StatusOr<std::unique_ptr<Executable>> Compile(
      const HloInstruction& instr, const BackendConfig& config) override {
    return absl::UnimplementedError("");
  }
  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) override {
    return absl::OkStatus();
  }

  bool CanProduceWrongResults() const override { return false; }

 private:
  autotuner::Backend backend_;
  std::string version_;
};

TEST(AutotuneCacheContextTest, Create) {
  stream_executor::GpuDeviceInfoProto proto;
  proto.set_name("test_gpu");
  proto.set_core_count(108);
  proto.set_clock_rate_ghz(1.41);
  proto.set_memory_bandwidth(1555000000000);
  proto.set_l2_cache_size(41943040);
  proto.mutable_cuda_compute_capability()->set_major(8);

  ASSERT_OK_AND_ASSIGN(stream_executor::DeviceDescription device_description,
                       stream_executor::DeviceDescription::FromProto(proto));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::make_unique<FakeCodegenBackend>(
      autotuner::Backend::TRITON, "1.2.3"));

  absl::Span<const std::unique_ptr<CodegenBackend>> backends_span(backends);

  AutotuneCacheContext context = AutotuneCacheContext::Create(
      device_description, backends_span, "my_explicit_version");

  EXPECT_EQ(context.device.size(), 16);
  EXPECT_NE(context.device, "unknown");
  EXPECT_EQ(context.explicit_version, "my_explicit_version");
  EXPECT_NE(context.codegen_version, "");
  EXPECT_NE(context.codegen_version, "unknown");
  EXPECT_EQ(context.per_backend_versions.at(autotuner::Backend::TRITON),
            "1.2.3");
}

}  // namespace
}  // namespace xla
