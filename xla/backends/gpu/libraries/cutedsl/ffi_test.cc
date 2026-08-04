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

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_platform.h"  // IWYU pragma: keep
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/resource_loader.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::cutedsl {
namespace {

TEST(CuteDslCustomCallTest, RunVectorAdd) {
  std::string hlo_text;
  ASSERT_OK(tsl::ReadFileToString(
      tsl::Env::Default(),
      tsl::GetDataDependencyFilepath(
          "xla/backends/gpu/libraries/cutedsl/vector_add.hlo"),
      &hlo_text));
  ASSERT_OK_AND_ASSIGN(auto module,
                       xla::ParseAndReturnUnverifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                       xla::GetXlaPjrtGpuClient(/*options=*/{}));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      client->CompileAndLoad(xla::XlaComputation(module->ToProto()),
                             /*options=*/{}));

  constexpr size_t kElementCount = 1024;
  std::vector<float> lhs(kElementCount);
  std::vector<float> rhs(kElementCount);
  for (size_t i = 0; i < kElementCount; ++i) {
    lhs[i] = static_cast<float>(i);
    rhs[i] = static_cast<float>(2 * i);
  }

  xla::Shape shape = xla::ShapeUtil::MakeShape(
      xla::F32, {static_cast<int64_t>(kElementCount)});
  ASSERT_FALSE(client->addressable_devices().empty());
  ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      client->addressable_devices().front()->default_memory_space());
  ASSERT_OK_AND_ASSIGN(
      auto lhs_buffer,
      client->BufferFromHostBuffer(
          lhs.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, memory_space,
          /*device_layout=*/nullptr));
  ASSERT_OK_AND_ASSIGN(
      auto rhs_buffer,
      client->BufferFromHostBuffer(
          rhs.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, memory_space,
          /*device_layout=*/nullptr));

  ASSERT_OK_AND_ASSIGN(
      auto results, executable->Execute({{lhs_buffer.get(), rhs_buffer.get()}},
                                        /*options=*/{}));
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);
  ASSERT_OK_AND_ASSIGN(auto result, results[0][0]->ToLiteral().Await());
  for (size_t i = 0; i < kElementCount; ++i) {
    EXPECT_FLOAT_EQ(result->Get<float>({static_cast<int64_t>(i)}),
                    lhs[i] + rhs[i]);
  }
}

}  // namespace
}  // namespace xla::gpu::cutedsl
