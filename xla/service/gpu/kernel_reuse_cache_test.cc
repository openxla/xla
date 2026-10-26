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
#include "xla/service/gpu/kernel_reuse_cache.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/gpu/kernel_reuse_cache.pb.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using KernelReuseTest = ::testing::Test;

TEST_F(KernelReuseTest, ExportAndLoadWork) {
  KernelReuseCache cache;
  EXPECT_TRUE(cache.IsEmpty());

  const std::string fingerprint = "fingerprint1";

  auto [promise, returned] = tsl::MakePromise<KernelReuseCache::Entry>();
  auto [future, was_cached] = cache.GetWithStatus(
      fingerprint, [returned = &returned]() { return *returned; });
  EXPECT_FALSE(was_cached);

  const std::string kernel_name = "kernel_name1";
  EXPECT_FALSE(future.IsReady());
  promise.Set(KernelReuseCache::Entry{kernel_name});

  ASSERT_OK_AND_ASSIGN(const KernelReuseCache::Entry& result, future.Await());
  EXPECT_EQ(result.kernel_name, kernel_name);
  EXPECT_FALSE(cache.IsEmpty());

  const CompilationCacheProto proto = cache.Export();
  cache.Clear();

  EXPECT_TRUE(cache.IsEmpty());

  EXPECT_THAT(proto, tsl::proto_testing::EquivToProto(R"pb(
                entries {
                  key: "kernel_name1"
                  value {
                    fingerprint: "fingerprint1"
                    launch_dimensions { num_blocks: 1 num_threads_per_block: 1 }
                  }
                }
                compatibility_version: 3
              )pb"));

  EXPECT_OK(cache.Load(proto));
  EXPECT_FALSE(cache.IsEmpty());

  {
    auto [future, was_cached] = cache.GetWithStatus(fingerprint, []() {
      return absl::UnimplementedError("Should be cached");
    });
    EXPECT_TRUE(was_cached);
    ASSERT_OK_AND_ASSIGN(const KernelReuseCache::Entry& result, future.Await());
    EXPECT_EQ(result.kernel_name, kernel_name);
  }
}

TEST_F(KernelReuseTest, UpdatingDiskKernelCacheWorks) {
  std::string cache_file_path;
  CHECK(tsl::Env::Default()->LocalTempFilename(&cache_file_path));
  {
    const CompilationCacheProto proto = [](std::string kernel_name) {
      KernelReuseCache cache;
      auto [result, was_cached] = cache.GetWithStatus("fingerprint", [&]() {
        KernelReuseCache::Entry entry;
        entry.kernel_name = kernel_name;
        entry.binary = std::make_shared<const std::vector<uint8_t>>(
            std::vector<uint8_t>{5, 6});
        return entry;
      });
      return cache.Export();
    }("k1");
    EXPECT_OK(
        UpdateDiskKernelCache(cache_file_path, /*do_append=*/false, proto));
  }
  {
    const CompilationCacheProto proto = [](std::string kernel_name) {
      KernelReuseCache cache;
      auto [result, was_cached] = cache.GetWithStatus("fingerprint1", [&]() {
        KernelReuseCache::Entry entry;
        entry.kernel_name = kernel_name;
        entry.binary = std::make_shared<const std::vector<uint8_t>>(
            std::vector<uint8_t>{7, 8});
        return entry;
      });
      return cache.Export();
    }("k2");
    EXPECT_OK(
        UpdateDiskKernelCache(cache_file_path, /*do_append=*/true, proto));
  }
  std::string serialized;
  EXPECT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), cache_file_path, &serialized));
  CompilationCacheProto proto;
  EXPECT_TRUE(proto.ParseFromString(serialized));
  EXPECT_EQ(proto.entries_size(), 2);
}

TEST_F(KernelReuseTest, EntryRemainsValidAfterCacheClearAndDestruction) {
  auto [promise, returned] = tsl::MakePromise<KernelReuseCache::Entry>();
  tsl::Future<KernelReuseCache::Entry> future;
  tsl::Future<KernelReuseCache::Entry> cached_future;
  {
    KernelReuseCache cache;
    bool was_cached = false;
    std::tie(future, was_cached) = cache.GetWithStatus(
        "fingerprint1", [returned = std::move(returned)]() mutable {
          return std::move(returned);
        });
    EXPECT_FALSE(was_cached);

    std::tie(cached_future, was_cached) = cache.GetWithStatus(
        "fingerprint1",
        []() { return absl::UnimplementedError("Should be cached"); });
    EXPECT_TRUE(was_cached);

    cache.Clear();
  }
  {
    auto p = std::move(promise);
    KernelReuseCache::Entry expected;
    expected.kernel_name = "kernel_after_clear";
    expected.binary = std::make_shared<const std::vector<uint8_t>>(
        std::vector<uint8_t>{1, 2, 3});
    p.Set(std::move(expected));
  }

  ASSERT_OK_AND_ASSIGN(const KernelReuseCache::Entry& result, future.Await());
  EXPECT_EQ(result.kernel_name, "kernel_after_clear");
  ASSERT_NE(result.binary, nullptr);
  EXPECT_THAT(*result.binary, testing::ElementsAre(1, 2, 3));

  ASSERT_OK_AND_ASSIGN(const KernelReuseCache::Entry& cached_result,
                       cached_future.Await());
  EXPECT_EQ(cached_result.binary.get(), result.binary.get());
}

TEST_F(KernelReuseTest, DefaultConstructedEntryHasNonNullEmptyBinary) {
  KernelReuseCache::Entry entry{"empty_kernel"};
  ASSERT_NE(entry.binary, nullptr);
  EXPECT_TRUE(entry.binary->empty());
}

TEST_F(KernelReuseTest, MovingFromReturnedEntryDoesNotMutateCachedEntry) {
  KernelReuseCache cache;
  auto [first_future, first_cached] =
      cache.GetWithStatus("fp", []() -> tsl::Future<KernelReuseCache::Entry> {
        KernelReuseCache::Entry entry;
        entry.kernel_name = "shared_kernel";
        entry.binary = std::make_shared<const std::vector<uint8_t>>(
            std::vector<uint8_t>{10, 20, 30});
        return entry;
      });
  EXPECT_FALSE(first_cached);

  // Simulate a consumer (like TritonFusion::Emit) copying Entry into a result
  // struct and then std::move'ing kernel_name and binary out of its copy.
  auto moved_future = first_future.Map([](KernelReuseCache::Entry entry_copy) {
    std::string moved_name = std::move(entry_copy.kernel_name);
    std::shared_ptr<const std::vector<uint8_t>> moved_binary =
        std::move(entry_copy.binary);
    return std::make_pair(std::move(moved_name), std::move(moved_binary));
  });
  ASSERT_OK_AND_ASSIGN(auto first_moved, moved_future.Await());
  EXPECT_EQ(first_moved.first, "shared_kernel");
  ASSERT_NE(first_moved.second, nullptr);

  // A subsequent cache hit must still see the intact kernel_name and share the
  // exact same underlying binary buffer.
  auto [second_future, second_cached] = cache.GetWithStatus(
      "fp", []() { return absl::UnimplementedError("Should be cached"); });
  EXPECT_TRUE(second_cached);
  ASSERT_OK_AND_ASSIGN(const KernelReuseCache::Entry& second_result,
                       second_future.Await());
  EXPECT_EQ(second_result.kernel_name, "shared_kernel");
  EXPECT_EQ(second_result.binary.get(), first_moved.second.get());
}

constexpr absl::string_view kDuplicateFusionsHlo = R"(
HloModule duplicate_fusions

fused_computation.1 {
  p0 = f32[16] parameter(0)
  p1 = f32[16] parameter(1)
  ROOT sub = f32[16] subtract(p0, p1)
}

fused_computation.2 {
  p0 = f32[16] parameter(0)
  p1 = f32[16] parameter(1)
  ROOT sub = f32[16] subtract(p0, p1)
}

ENTRY main {
  p0 = f32[16] parameter(0)
  p1 = f32[16] parameter(1)
  fusion.1 = f32[16] fusion(p0, p1), kind=kLoop, calls=fused_computation.1
  ROOT fusion.2 = f32[16] fusion(fusion.1, p1), kind=kLoop, calls=fused_computation.2
}
)";

TEST_F(KernelReuseTest, IdenticalFusionsShareEntryByDefault) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kDuplicateFusionsHlo));
  const HloInstruction* fusion1 =
      module->entry_computation()->GetInstructionWithName("fusion.1");
  const HloInstruction* fusion2 =
      module->entry_computation()->GetInstructionWithName("fusion.2");
  ASSERT_NE(fusion1, nullptr);
  ASSERT_NE(fusion2, nullptr);

  KernelReuseCache cache;
  auto [first_future, first_cached] = cache.GetWithStatus(
      fusion1->fused_instructions_computation(), /*kernel_arguments=*/{},
      /*discriminator=*/"", []() -> tsl::Future<KernelReuseCache::Entry> {
        return KernelReuseCache::Entry{"loop_subtract_fusion_1"};
      });
  EXPECT_FALSE(first_cached);

  auto [second_future, second_cached] = cache.GetWithStatus(
      fusion2->fused_instructions_computation(), /*kernel_arguments=*/{},
      /*discriminator=*/"", []() -> tsl::Future<KernelReuseCache::Entry> {
        return KernelReuseCache::Entry{"loop_subtract_fusion_2"};
      });
  EXPECT_TRUE(second_cached);
  ASSERT_OK_AND_ASSIGN(const KernelReuseCache::Entry& second_entry,
                       second_future.Await());
  EXPECT_EQ(second_entry.kernel_name, "loop_subtract_fusion_1");
}

TEST_F(KernelReuseTest,
       IdenticalFusionsGetUniqueKernelsWhenGenerateDebugInfoEnabled) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kDuplicateFusionsHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_generate_debug_info(true);

  const HloInstruction* fusion1 =
      module->entry_computation()->GetInstructionWithName("fusion.1");
  const HloInstruction* fusion2 =
      module->entry_computation()->GetInstructionWithName("fusion.2");
  ASSERT_NE(fusion1, nullptr);
  ASSERT_NE(fusion2, nullptr);

  KernelReuseCache cache;
  auto [first_future, first_cached] = cache.GetWithStatus(
      fusion1->fused_instructions_computation(), /*kernel_arguments=*/{},
      /*discriminator=*/"", []() -> tsl::Future<KernelReuseCache::Entry> {
        return KernelReuseCache::Entry{"loop_subtract_fusion_1"};
      });
  EXPECT_FALSE(first_cached);

  auto [second_future, second_cached] = cache.GetWithStatus(
      fusion2->fused_instructions_computation(), /*kernel_arguments=*/{},
      /*discriminator=*/"", []() -> tsl::Future<KernelReuseCache::Entry> {
        return KernelReuseCache::Entry{"loop_subtract_fusion_2"};
      });
  EXPECT_FALSE(second_cached);
  ASSERT_OK_AND_ASSIGN(const KernelReuseCache::Entry& first_entry,
                       first_future.Await());
  ASSERT_OK_AND_ASSIGN(const KernelReuseCache::Entry& second_entry,
                       second_future.Await());
  EXPECT_EQ(first_entry.kernel_name, "loop_subtract_fusion_1");
  EXPECT_EQ(second_entry.kernel_name, "loop_subtract_fusion_2");
}

TEST_F(KernelReuseTest,
       IdenticalFusionsGetUniqueKernelsWhenGenerateLineInfoEnabled) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kDuplicateFusionsHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_generate_line_info(true);

  const HloInstruction* fusion1 =
      module->entry_computation()->GetInstructionWithName("fusion.1");
  const HloInstruction* fusion2 =
      module->entry_computation()->GetInstructionWithName("fusion.2");
  ASSERT_NE(fusion1, nullptr);
  ASSERT_NE(fusion2, nullptr);

  KernelReuseCache cache;
  auto [first_future, first_cached] = cache.GetWithStatus(
      fusion1->fused_instructions_computation(), /*kernel_arguments=*/{},
      /*discriminator=*/"", []() -> tsl::Future<KernelReuseCache::Entry> {
        return KernelReuseCache::Entry{"loop_subtract_fusion_1"};
      });
  EXPECT_FALSE(first_cached);

  auto [second_future, second_cached] = cache.GetWithStatus(
      fusion2->fused_instructions_computation(), /*kernel_arguments=*/{},
      /*discriminator=*/"", []() -> tsl::Future<KernelReuseCache::Entry> {
        return KernelReuseCache::Entry{"loop_subtract_fusion_2"};
      });
  EXPECT_FALSE(second_cached);
  ASSERT_OK_AND_ASSIGN(const KernelReuseCache::Entry& first_entry,
                       first_future.Await());
  ASSERT_OK_AND_ASSIGN(const KernelReuseCache::Entry& second_entry,
                       second_future.Await());
  EXPECT_EQ(first_entry.kernel_name, "loop_subtract_fusion_1");
  EXPECT_EQ(second_entry.kernel_name, "loop_subtract_fusion_2");
}

TEST_F(KernelReuseTest,
       IdenticalFusionsGetUniqueKernelsWhenNsightComputeEnvSet) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kDuplicateFusionsHlo));
  const HloInstruction* fusion1 =
      module->entry_computation()->GetInstructionWithName("fusion.1");
  const HloInstruction* fusion2 =
      module->entry_computation()->GetInstructionWithName("fusion.2");
  ASSERT_NE(fusion1, nullptr);
  ASSERT_NE(fusion2, nullptr);

  setenv("NV_COMPUTE_PROFILER_LOCAL_CONNECTION_OVERRIDE", "1", /*overwrite=*/1);
  KernelReuseCache cache;
  auto [first_future, first_cached] = cache.GetWithStatus(
      fusion1->fused_instructions_computation(), /*kernel_arguments=*/{},
      /*discriminator=*/"", []() -> tsl::Future<KernelReuseCache::Entry> {
        return KernelReuseCache::Entry{"loop_subtract_fusion_1"};
      });
  auto [second_future, second_cached] = cache.GetWithStatus(
      fusion2->fused_instructions_computation(), /*kernel_arguments=*/{},
      /*discriminator=*/"", []() -> tsl::Future<KernelReuseCache::Entry> {
        return KernelReuseCache::Entry{"loop_subtract_fusion_2"};
      });
  unsetenv("NV_COMPUTE_PROFILER_LOCAL_CONNECTION_OVERRIDE");

  EXPECT_FALSE(first_cached);
  EXPECT_FALSE(second_cached);
  ASSERT_OK_AND_ASSIGN(const KernelReuseCache::Entry& first_entry,
                       first_future.Await());
  ASSERT_OK_AND_ASSIGN(const KernelReuseCache::Entry& second_entry,
                       second_future.Await());
  EXPECT_EQ(first_entry.kernel_name, "loop_subtract_fusion_1");
  EXPECT_EQ(second_entry.kernel_name, "loop_subtract_fusion_2");
}

}  // namespace
}  // namespace xla::gpu
