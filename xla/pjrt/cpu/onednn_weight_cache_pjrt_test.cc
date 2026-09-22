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
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/service/cpu/onednn_weight_cache.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla {
namespace {

constexpr char kMatMul[] = R"(
HloModule cached_matmul
ENTRY main {
  lhs = f32[2,3]{1,0} parameter(0)
  rhs = f32[3,2]{1,0} parameter(1)
  ROOT result = f32[2,2]{1,0} custom-call(lhs, rhs),
    custom_call_target="__onednn$matmul",
    backend_config={"onednn_matmul_config":{}}
})";

class OneDnnWeightCachePjRtTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_GT(cache_.capacity_bytes(), 0);
    CpuClientOptions options;
    options.cpu_device_count = 1;
    ASSERT_OK_AND_ASSIGN(client_, GetPjRtCpuClient(std::move(options)));
    ASSERT_OK_AND_ASSIGN(input_, client_->BufferFromHostLiteral(
                                     lhs_, client_->memory_spaces()[0]));
  }

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      absl::string_view hlo = kMatMul) {
    ABSL_ASSIGN_OR_RETURN(auto module, ParseAndReturnUnverifiedModule(hlo, {}));
    return client_->CompileAndLoad(XlaComputation(module->ToProto()), {});
  }

  absl::Status Run(PjRtLoadedExecutable& executable, PjRtBuffer& weights,
                   const Literal& expected) {
    ABSL_ASSIGN_OR_RETURN(auto result,
                          executable.Execute({{input_.get(), &weights}}, {}));
    ABSL_ASSIGN_OR_RETURN(auto actual, result[0][0]->ToLiteral().Await());
    EXPECT_EQ(*actual, expected);
    return absl::OkStatus();
  }

  cpu::OneDnnWeightCache& cache_ = cpu::GlobalOneDnnWeightCache();
  Literal lhs_ = LiteralUtil::CreateR2<float>({{1, 2, 3}, {4, 5, 6}});
  Literal expected_ = LiteralUtil::CreateR2<float>({{58, 64}, {139, 154}});
  std::unique_ptr<PjRtClient> client_;
  std::unique_ptr<PjRtBuffer> input_;
};

TEST_F(OneDnnWeightCachePjRtTest, InvalidatesDonation) {
  ASSERT_OK_AND_ASSIGN(auto executable, Compile());
  auto rhs = LiteralUtil::CreateR2<float>({{7, 8}, {9, 10}, {11, 12}});
  ASSERT_OK_AND_ASSIGN(auto weights, client_->BufferFromHostLiteral(
                                         rhs, client_->memory_spaces()[0]));
  auto before = cache_.stats();
  ASSERT_OK(Run(*executable, *weights, expected_));
  ASSERT_EQ(cache_.stats().insertions, before.insertions + 1);

  // A different executable writes the donated allocation. Its new contents
  // must never reuse the packed representation of the previous generation.
  constexpr char kDonate[] = R"(
HloModule update, input_output_alias={ {}: (0, {}, must-alias) }
ENTRY main {
  weights = f32[3,2]{1,0} parameter(0)
  ones = f32[3,2]{1,0} constant({{1,1},{1,1},{1,1}})
  ROOT updated = f32[3,2]{1,0} add(weights, ones)
})";
  ASSERT_OK_AND_ASSIGN(auto update, Compile(kDonate));
  ASSERT_OK_AND_ASSIGN(auto updated, update->Execute({{weights.get()}}, {}));
  ASSERT_OK(Run(*executable, *updated[0][0],
                LiteralUtil::CreateR2<float>({{64, 70}, {154, 169}})));
  EXPECT_TRUE(weights->IsDeleted());
  EXPECT_EQ(cache_.stats().insertions, before.insertions + 2);
}

TEST_F(OneDnnWeightCachePjRtTest,
       HostWritesInvalidateAndPointerExportsDisableReuse) {
  ASSERT_OK_AND_ASSIGN(auto executable, Compile());
  auto rhs = LiteralUtil::CreateR2<float>({{7, 8}, {9, 10}, {11, 12}});
  ASSERT_OK_AND_ASSIGN(auto weights, client_->BufferFromHostLiteral(
                                         rhs, client_->memory_spaces()[0]));

  auto before = cache_.stats();
  for (int i = 0; i < 2; ++i) {
    ASSERT_OK(Run(*executable, *weights, expected_));
  }
  ASSERT_OK_AND_ASSIGN(auto readback, weights->ToLiteral().Await());
  EXPECT_EQ(*readback, rhs);
  ASSERT_OK(Run(*executable, *weights, expected_));
  EXPECT_EQ(cache_.stats().insertions, before.insertions + 1);
  EXPECT_EQ(cache_.stats().hits, before.hits + 2);

  // Managed writes replace the generation but still allow subsequent reuse.
  ASSERT_OK_AND_ASSIGN(auto raw,
                       PjRtRawBuffer::CreateRawAliasOfBuffer(weights.get()));
  rhs.data<float>()[0] = 17;
  ASSERT_OK(raw->CopyRawHostToDevice(rhs.untyped_data(), 0, rhs.size_bytes())
                .Await());
  for (int i = 0; i < 2; ++i) {
    ASSERT_OK(Run(*executable, *weights,
                  LiteralUtil::CreateR2<float>({{68, 64}, {179, 154}})));
  }
  EXPECT_EQ(cache_.stats().insertions, before.insertions + 2);
  EXPECT_EQ(cache_.stats().hits, before.hits + 3);

  auto* cpu_client = static_cast<CommonPjRtClient*>(client_.get());
  for (bool host_buffer : {false, true}) {
    rhs.data<float>()[0] = host_buffer ? 37 : 27;
    auto semantics = PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall;
    ASSERT_OK_AND_ASSIGN(
        auto written,
        host_buffer
            ? cpu_client->LinearizeHostBufferInto(
                  rhs.untyped_data(), F32, rhs.shape().dimensions(),
                  std::nullopt, semantics, nullptr, rhs.shape(), raw)
            : cpu_client->LinearizeInto(rhs, rhs.shape(), semantics, raw));
    absl::Notification completed;
    written.AndThen([&] { completed.Notify(); });
    ASSERT_TRUE(completed.WaitForNotificationWithTimeout(absl::Seconds(5)));
    ASSERT_FALSE(written.GetErrorIfPresent().has_value());
    auto expected = LiteralUtil::CreateR2<float>(
        {{host_buffer ? 88.f : 78.f, 64}, {host_buffer ? 259.f : 219.f, 154}});
    ASSERT_OK(Run(*executable, *weights, expected));
    ASSERT_OK(Run(*executable, *weights, expected));
  }
  EXPECT_EQ(cache_.stats().insertions, before.insertions + 4);
  EXPECT_EQ(cache_.stats().hits, before.hits + 5);

  // A retained host pointer can write again after another execution completes.
  // Invalidating just once at export must not allow stale packed weights later.
  auto* pointer = static_cast<float*>(raw->GetHostPointer());
  ASSERT_NE(pointer, nullptr);
  for (int i = 0; i < 2; ++i) {
    pointer[0] = 27 + 10 * i;
    ASSERT_OK(Run(*executable, *weights,
                  LiteralUtil::CreateR2<float>(
                      {{78.f + 10 * i, 64}, {219.f + 40 * i, 154}})));
  }
  EXPECT_EQ(cache_.stats().insertions, before.insertions + 4);
  EXPECT_EQ(cache_.stats().hits, before.hits + 5);
}

TEST_F(OneDnnWeightCachePjRtTest, ImportsRespectExternalMutability) {
  ASSERT_OK_AND_ASSIGN(auto executable, Compile());
  struct HostState {
    alignas(64) float rhs[6] = {7, 8, 9, 10, 11, 12};
    absl::Notification released;
  };
  auto state = std::make_shared<HostState>();
  auto* cpu_client = static_cast<CommonPjRtClient*>(client_.get());
  auto* memory_space = client_->memory_spaces()[0];
  auto before = cache_.stats();
  ASSERT_OK_AND_ASSIGN(auto raw,
                       cpu_client->ImportForeignMemory(
                           state->rhs, [state] { state->released.Notify(); },
                           sizeof(state->rhs), memory_space, false));
  ASSERT_OK_AND_ASSIGN(
      auto weights, cpu_client->DefineBuffer(ShapeUtil::MakeShape(F32, {3, 2}),
                                             memory_space, std::move(raw), {}));
  ASSERT_OK(Run(*executable, *weights, expected_));
  state->rhs[0] = 17;
  auto updated = LiteralUtil::CreateR2<float>({{68, 64}, {179, 154}});
  ASSERT_OK(Run(*executable, *weights, updated));
  EXPECT_EQ(cache_.stats().insertions, before.insertions);
  EXPECT_EQ(cache_.stats().hits, before.hits);

  weights.reset();
  EXPECT_TRUE(state->released.WaitForNotificationWithTimeout(absl::Seconds(5)));
  for (auto semantics : {PjRtClient::HostBufferSemantics::kImmutableZeroCopy,
                         PjRtClient::HostBufferSemantics::kMutableZeroCopy}) {
    state = std::make_shared<HostState>();
    ASSERT_OK_AND_ASSIGN(
        weights,
        client_->BufferFromHostBuffer(
            state->rhs, F32, {3, 2}, std::nullopt, semantics,
            [state] { state->released.Notify(); }, memory_space, nullptr));
    ASSERT_OK_AND_ASSIGN(raw,
                         PjRtRawBuffer::CreateRawAliasOfBuffer(weights.get()));
    ASSERT_EQ(raw->GetHostPointerForInternalUse(
                  PjRtRawBufferInterface::HostAccess::kRead),
              state->rhs);
    raw.reset();
    for (int iteration = 0; iteration < 2; ++iteration) {
      ASSERT_OK(Run(*executable, *weights, expected_));
    }
    EXPECT_EQ(cache_.stats().insertions, before.insertions);
    EXPECT_EQ(cache_.stats().hits, before.hits);
    EXPECT_FALSE(state->released.HasBeenNotified());
    weights.reset();
    EXPECT_TRUE(
        state->released.WaitForNotificationWithTimeout(absl::Seconds(5)));
  }
}

}  // namespace
}  // namespace xla
