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

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/barrier.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_common.hpp"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/device_event_utils.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/strided_gather.h"
#include "xla/service/cpu/onednn_weight_cache.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/util/tied_ref.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

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
    ASSERT_OK_AND_ASSIGN(client_, GetXlaPjrtCpuClient(std::move(options)));
    ASSERT_OK_AND_ASSIGN(input_, client_->BufferFromHostLiteral(
                                     lhs_, client_->memory_spaces()[0]));
  }

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      absl::string_view hlo = kMatMul, CompileOptions options = {}) {
    ABSL_ASSIGN_OR_RETURN(auto module, ParseAndReturnUnverifiedModule(hlo, {}));
    return client_->CompileAndLoad(XlaComputation(module->ToProto()),
                                   std::move(options));
  }

  absl::Status Run(PjRtLoadedExecutable& executable, PjRtBuffer& weights,
                   const Literal& expected, ExecuteOptions options = {}) {
    if (options.execution_mode == ExecuteOptions::ExecutionMode::kDefault) {
      options.execution_mode = ExecuteOptions::ExecutionMode::kSynchronous;
    }
    ABSL_ASSIGN_OR_RETURN(
        auto result, executable.Execute({{input_.get(), &weights}}, options));
    ABSL_ASSIGN_OR_RETURN(auto actual, result[0][0]->ToLiteral().Await());
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, *actual));
    return absl::OkStatus();
  }

  absl::Status RunWithBias(PjRtLoadedExecutable& executable,
                           PjRtBuffer& weights, float bias_value) {
    SCOPED_TRACE(bias_value);
    auto bias = LiteralUtil::CreateR2<float>({{bias_value, -bias_value}});
    ABSL_ASSIGN_OR_RETURN(
        auto bias_buffer,
        client_->BufferFromHostLiteral(bias, client_->memory_spaces()[0]));
    ExecuteOptions options;
    options.execution_mode = ExecuteOptions::ExecutionMode::kSynchronous;
    ABSL_ASSIGN_OR_RETURN(
        auto result,
        executable.Execute({{input_.get(), &weights, bias_buffer.get()}},
                           options));
    ABSL_ASSIGN_OR_RETURN(auto actual, result[0][0]->ToLiteral().Await());
    EXPECT_TRUE(LiteralTestUtil::Equal(
        LiteralUtil::CreateR2<float>({{58 + bias_value, 64 - bias_value},
                                      {139 + bias_value, 154 - bias_value}}),
        *actual));
    return absl::OkStatus();
  }

  // Probe after synchronous execution sets oneDNN's calling-thread concurrency.
  // Some implementations use the original layout and need no cached copy.
  dnnl::memory::desc SelectedWeightsDesc(bool transpose_lhs = false,
                                         bool has_bias = false) {
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::matmul::primitive_desc selected(
        engine, {{2, 3}, dt::f32, transpose_lhs ? tag::ba : tag::ab},
        {{3, 2}, dt::f32, tag::any},
        has_bias ? dnnl::memory::desc({1, 2}, dt::f32, tag::ab)
                 : dnnl::memory::desc(),
        {{2, 2}, dt::f32, tag::ab});
    return selected.weights_desc();
  }

  const dnnl::memory::desc source_weights_desc_{
      {3, 2}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab};
  cpu::OneDnnWeightCache& cache_ = cpu::GlobalOneDnnWeightCache();
  Literal lhs_ = LiteralUtil::CreateR2<float>({{1, 2, 3}, {4, 5, 6}});
  Literal expected_ = LiteralUtil::CreateR2<float>({{58, 64}, {139, 154}});
  std::unique_ptr<PjRtClient> client_;
  std::unique_ptr<PjRtBuffer> input_;
};

TEST_F(OneDnnWeightCachePjRtTest, CapacityRejectionPreservesBias) {
  ASSERT_OK_AND_ASSIGN(auto executable, Compile(R"(
HloModule biased_matmul
ENTRY main {
  lhs = f32[2,3]{1,0} parameter(0)
  rhs = f32[3,2]{1,0} parameter(1)
  bias = f32[1,2]{1,0} parameter(2)
  ROOT result = f32[2,2]{1,0} custom-call(lhs, rhs, bias),
    custom_call_target="__onednn$matmul",
    backend_config={"onednn_matmul_config":{"fusions":{"ops":["BIAS"]}}}
})"));
  auto rhs = LiteralUtil::CreateR2<float>({{7, 8}, {9, 10}, {11, 12}});
  ASSERT_OK_AND_ASSIGN(auto weights, client_->BufferFromHostLiteral(
                                         rhs, client_->memory_spaces()[0]));
  auto owner = std::make_shared<tsl::TiedAny>();
  dnnl::memory::desc desc({static_cast<int64_t>(cache_.capacity_bytes())},
                          dnnl::memory::data_type::u8,
                          dnnl::memory::format_tag::x);
  auto held = cache_.LookupOrCreate({owner.get(), desc, desc, owner});
  ASSERT_EQ(held.kind, cpu::OneDnnWeightCache::LookupKind::kMiss);
  // Cancel the entry but retain its allocation to keep the budget full.
  cache_.Complete(held.weights, /*success=*/false);
  ASSERT_EQ(cache_.stats().live_bytes, cache_.capacity_bytes());
  // Changing bias must be honored on fallback and after releasing the budget.
  // If oneDNN selects a packed layout, the second phase must reuse it.
  for (bool reject : {true, false}) {
    SCOPED_TRACE(reject ? "capacity rejection" : "cache reuse");
    if (!reject) {
      held.weights.reset();
    }
    auto before = cache_.stats();
    for (float bias_value : {3.f, -7.f}) {
      ASSERT_OK(RunWithBias(*executable, *weights, bias_value));
    }
    const bool cached = !reject && SelectedWeightsDesc(/*transpose_lhs=*/false,
                                                       /*has_bias=*/true) !=
                                       source_weights_desc_;
    EXPECT_EQ(cache_.stats().insertions, before.insertions + cached);
    EXPECT_EQ(cache_.stats().hits, before.hits + cached);
  }
}

TEST_F(OneDnnWeightCachePjRtTest, SelectedLayoutControlsPacking) {
  // A transposed input can select a primitive that uses the source weights
  // layout directly. Its bias must survive without allocating a cached copy.
  ASSERT_OK_AND_ASSIGN(auto executable, Compile(R"(
HloModule transposed_input
ENTRY main {
  lhs = f32[3,2]{1,0} parameter(0)
  rhs = f32[3,2]{1,0} parameter(1)
  bias = f32[1,2]{1,0} parameter(2)
  ROOT result = f32[2,2]{1,0} custom-call(lhs, rhs, bias),
    custom_call_target="__onednn$matmul",
    backend_config={"onednn_matmul_config":{"transpose_a":true,"fusions":{"ops":["BIAS"]}}}
})"));
  auto lhs = LiteralUtil::CreateR2<float>({{1, 4}, {2, 5}, {3, 6}});
  ASSERT_OK_AND_ASSIGN(
      input_, client_->BufferFromHostLiteral(lhs, client_->memory_spaces()[0]));
  auto rhs = LiteralUtil::CreateR2<float>({{7, 8}, {9, 10}, {11, 12}});
  ASSERT_OK_AND_ASSIGN(auto weights, client_->BufferFromHostLiteral(
                                         rhs, client_->memory_spaces()[0]));

  auto before = cache_.stats();
  for (float bias_value : {3.f, -7.f}) {
    ASSERT_OK(RunWithBias(*executable, *weights, bias_value));
  }
  const auto selected =
      SelectedWeightsDesc(/*transpose_lhs=*/true, /*has_bias=*/true);
  const bool needs_packing = selected != source_weights_desc_;
  EXPECT_EQ(cache_.stats().insertions, before.insertions + needs_packing);
  EXPECT_EQ(cache_.stats().hits, before.hits + needs_packing);
  EXPECT_EQ(cache_.stats().live_bytes,
            before.live_bytes + (needs_packing ? selected.get_size() : 0));
}

TEST_F(OneDnnWeightCachePjRtTest, ConcurrentColdAndWarmRequests) {
  // Cover compiler selection and a transposed RHS as well as concurrent reuse.
  CompileOptions compile_options;
  auto* debug_options =
      compile_options.executable_build_options.mutable_debug_options();
  debug_options->set_xla_cpu_experimental_onednn_custom_call(true);
  debug_options->clear_xla_cpu_experimental_ynn_fusion_type();
  constexpr char kDot[] = R"(
HloModule concurrent_matmul
ENTRY main {
  lhs = f32[256,256]{1,0} parameter(0)
  rhs = f32[256,256]{1,0} parameter(1)
  ROOT result = f32[256,256]{1,0} dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";
  ASSERT_OK_AND_ASSIGN(auto executable,
                       Compile(kDot, std::move(compile_options)));
  // The compiler's eligibility flag must survive executable serialization.
  ASSERT_OK_AND_ASSIGN(auto serialized, executable->SerializeExecutable());
  ASSERT_OK_AND_ASSIGN(executable, client_->LoadSerializedExecutable(
                                       serialized, std::nullopt, {}));
  auto lhs = LiteralUtil::CreateFull<float>({256, 256}, 1);
  auto rhs = LiteralUtil::CreateFull<float>({256, 256}, 0);
  auto expected = LiteralUtil::CreateFull<float>({256, 256}, 0);
  for (int64_t row = 0; row < 256; ++row) {
    for (int64_t column = 0; column < 256; ++column) {
      rhs.Set<float>({row, column}, row + 2 * column);
      // Each output column is the sum of one RHS row.
      expected.Set<float>({row, column}, 256 * (column + 255));
    }
  }
  ASSERT_OK_AND_ASSIGN(
      input_, client_->BufferFromHostLiteral(lhs, client_->memory_spaces()[0]));
  ASSERT_OK_AND_ASSIGN(auto weights, client_->BufferFromHostLiteral(
                                         rhs, client_->memory_spaces()[0]));
  // Separate streams allow overlap while sharing weights. Expect one cold
  // insertion, then two warm hits.
  for (bool warm : {false, true}) {
    SCOPED_TRACE(warm ? "warm cache" : "cold cache");
    auto before = cache_.stats();
    absl::Barrier start(2);
    std::array<std::thread, 2> callers;
    for (int stream_id : {1, 2}) {
      callers[stream_id - 1] = std::thread([&, stream_id] {
        SCOPED_TRACE(warm ? "warm cache" : "cold cache");
        SCOPED_TRACE(stream_id);
        ExecuteOptions options;
        options.execution_mode = ExecuteOptions::ExecutionMode::kAsynchronous;
        options.execution_stream_id = stream_id;
        start.Block();
        EXPECT_OK(Run(*executable, *weights, expected, options));
      });
    }
    for (auto& caller : callers) {
      caller.join();
    }
    EXPECT_EQ(cache_.stats().insertions, before.insertions + (warm ? 0 : 1));
    // A cold caller can bypass a pending fill or hit after it completes.
    if (warm) {
      EXPECT_EQ(cache_.stats().hits, before.hits + callers.size());
    }
  }
}

TEST_F(OneDnnWeightCachePjRtTest, InvalidatesDonation) {
  ASSERT_OK_AND_ASSIGN(auto executable, Compile());
  auto rhs = LiteralUtil::CreateR2<float>({{7, 8}, {9, 10}, {11, 12}});
  ASSERT_OK_AND_ASSIGN(auto weights, client_->BufferFromHostLiteral(
                                         rhs, client_->memory_spaces()[0]));
  auto before = cache_.stats();
  ASSERT_OK(Run(*executable, *weights, expected_));
  const bool needs_packing = SelectedWeightsDesc() != source_weights_desc_;
  ASSERT_EQ(cache_.stats().insertions, before.insertions + needs_packing);

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
  EXPECT_EQ(cache_.stats().insertions, before.insertions + 2 * needs_packing);
}

TEST_F(OneDnnWeightCachePjRtTest,
       SliceCopiesPreserveReuseAndUpdatesInvalidate) {
  ASSERT_OK_AND_ASSIGN(auto executable, Compile());
  auto rhs = LiteralUtil::CreateR2<float>({{7, 8}, {9, 10}, {11, 12}});
  ASSERT_OK_AND_ASSIGN(auto weights, client_->BufferFromHostLiteral(
                                         rhs, client_->memory_spaces()[0]));
  auto before = cache_.stats();
  ASSERT_OK(Run(*executable, *weights, expected_));
  const bool needs_packing = SelectedWeightsDesc() != source_weights_desc_;
  ASSERT_EQ(cache_.stats().insertions, before.insertions + needs_packing);

  StridedGather gather({
      /*element_size=*/sizeof(float),
      /*counts=*/{2, 3},
      /*src_byte_offset=*/0,
      /*src_byte_size=*/6 * sizeof(float),
      /*src_byte_strides=*/{sizeof(float), 2 * sizeof(float)},
  });
  // Copying column-wise preserves source reuse and leaves the copy cacheable.
  ASSERT_OK_AND_ASSIGN(auto copied, weights->CopySlice(gather, rhs.shape()));
  ASSERT_OK(Run(*executable, *copied,
                LiteralUtil::CreateR2<float>({{59, 61}, {143, 148}})));
  ASSERT_OK(Run(*executable, *weights, expected_));
  EXPECT_EQ(cache_.stats().insertions, before.insertions + 2 * needs_packing);
  EXPECT_EQ(cache_.stats().hits, before.hits + needs_packing);

  // Updating replaces the copy's generation; both buffers remain cacheable.
  ASSERT_OK_AND_ASSIGN(auto updated,
                       copied->UpdateSlice(gather, weights.get()));
  EXPECT_TRUE(copied->IsDeleted());
  for (int i = 0; i < 2; ++i) {
    ASSERT_OK(Run(*executable, *updated,
                  LiteralUtil::CreateR2<float>({{50, 68}, {122, 167}})));
  }
  ASSERT_OK(Run(*executable, *weights, expected_));
  EXPECT_EQ(cache_.stats().insertions, before.insertions + 3 * needs_packing);
  EXPECT_EQ(cache_.stats().hits, before.hits + 3 * needs_packing);
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
  EXPECT_TRUE(LiteralTestUtil::Equal(rhs, *readback));
  ASSERT_OK(Run(*executable, *weights, expected_));
  const bool needs_packing = SelectedWeightsDesc() != source_weights_desc_;
  EXPECT_EQ(cache_.stats().insertions, before.insertions + needs_packing);
  EXPECT_EQ(cache_.stats().hits, before.hits + 2 * needs_packing);

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
  EXPECT_EQ(cache_.stats().insertions, before.insertions + 2 * needs_packing);
  EXPECT_EQ(cache_.stats().hits, before.hits + 3 * needs_packing);

  auto* common_client = static_cast<CommonPjRtClient*>(client_.get());
  for (bool host_buffer : {false, true}) {
    rhs.data<float>()[0] = host_buffer ? 37 : 27;
    auto semantics = PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall;
    ASSERT_OK_AND_ASSIGN(
        auto written,
        host_buffer
            ? common_client->LinearizeHostBufferInto(
                  rhs.untyped_data(), F32, rhs.shape().dimensions(),
                  std::nullopt, semantics, nullptr, rhs.shape(), raw)
            : common_client->LinearizeInto(rhs, rhs.shape(), semantics, raw));
    BlockUntilReady(written);
    ASSERT_OK(written.GetErrorIfPresent().value_or(absl::OkStatus()));
    auto expected = LiteralUtil::CreateR2<float>(
        {{host_buffer ? 88.f : 78.f, 64}, {host_buffer ? 259.f : 219.f, 154}});
    ASSERT_OK(Run(*executable, *weights, expected));
    ASSERT_OK(Run(*executable, *weights, expected));
  }
  EXPECT_EQ(cache_.stats().insertions, before.insertions + 4 * needs_packing);
  EXPECT_EQ(cache_.stats().hits, before.hits + 5 * needs_packing);

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
  EXPECT_EQ(cache_.stats().insertions, before.insertions + 4 * needs_packing);
  EXPECT_EQ(cache_.stats().hits, before.hits + 5 * needs_packing);
}

TEST_F(OneDnnWeightCachePjRtTest, ImportsRespectExternalMutability) {
  ASSERT_OK_AND_ASSIGN(auto executable, Compile());
  struct HostState {
    alignas(64) float rhs[6] = {7, 8, 9, 10, 11, 12};
    absl::Notification released;
  };
  auto state = std::make_shared<HostState>();
  auto* common_client = static_cast<CommonPjRtClient*>(client_.get());
  auto* memory_space = client_->memory_spaces()[0];
  auto before = cache_.stats();
  ASSERT_OK_AND_ASSIGN(auto raw,
                       common_client->ImportForeignMemory(
                           state->rhs, [state] { state->released.Notify(); },
                           sizeof(state->rhs), memory_space, false));
  ASSERT_OK_AND_ASSIGN(auto weights, common_client->DefineBuffer(
                                         ShapeUtil::MakeShape(F32, {3, 2}),
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
