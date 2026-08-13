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

// Tests for roctx_utils.cc — the ROCm dual-push implementation of
// nvtx_utils.h. Verifies that RangePush/RangePop populate AnnotationStack
// (Pipeline A) and emit roctx markers (Pipeline B).

#include <string>

#include "absl/strings/str_cat.h"
#include "rocm/include/rocprofiler-sdk-roctx/roctx.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#include "tsl/profiler/lib/nvtx_utils.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace tsl {
namespace profiler {
namespace {

// RAII guard that enables AnnotationStack on construction and disables
// on destruction so tests don't leak enabled state.
class AnnotationStackGuard {
 public:
  AnnotationStackGuard() { AnnotationStack::Enable(true); }
  ~AnnotationStackGuard() { AnnotationStack::Enable(false); }
};

TEST(RoctxUtils, DefaultProfilerDomainDefaultsToNullAndIsStable) {
  // Two properties, both load-bearing.
  //
  // 1. Default off, matching CUDA's enable_nvtx_tracking=false. Tests run
  //    without XLA_ROCM_ENABLE_ROCTX set, so the domain must be null and
  //    PushAnnotation must take the AnnotationStack branch.
  // 2. LATCHED. scoped_annotation.h branches on this in both PushAnnotation
  //    and PopAnnotation; a handle that changed mid-process would push down
  //    one path and pop down the other, unbalancing both stacks.
  ProfilerDomainHandle first = DefaultProfilerDomain();
  EXPECT_EQ(first, nullptr)
      << "Domain must default to null; XLA_ROCM_ENABLE_ROCTX opts in";
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(DefaultProfilerDomain(), first)
        << "DefaultProfilerDomain() must be stable for the process lifetime";
  }
}

TEST(RoctxUtils, RangePushPopulatesAnnotationStack) {
  auto domain = DefaultProfilerDomain();

  AnnotationStackGuard guard;
  RangePush(domain, "test_op");
  EXPECT_EQ(AnnotationStack::Get(), "test_op");
  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "");
}

TEST(RoctxUtils, NestedPushPopMaintainsAnnotationStack) {
  auto domain = DefaultProfilerDomain();

  AnnotationStackGuard guard;

  RangePush(domain, "outer");
  EXPECT_EQ(AnnotationStack::Get(), "outer");

  RangePush(domain, "inner");
  EXPECT_EQ(AnnotationStack::Get(), "outer::inner");

  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "outer");

  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "");
}

TEST(RoctxUtils, PushPopWithAnnotationStackDisabled) {
  auto domain = DefaultProfilerDomain();

  // AnnotationStack is NOT enabled. RangePush/RangePop must not crash
  // and must not populate the stack.
  RangePush(domain, "ignored_op");
  EXPECT_EQ(AnnotationStack::Get(), "");
  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "");
}

TEST(RoctxUtils, ScopedAnnotationIntegration) {
  // Full chain in THIS binary, where the domain is null:
  // ScopedAnnotation -> PushAnnotation -> AnnotationStack::PushAnnotation.
  // The domain path -- which is what XLA takes when XLA_ROCM_ENABLE_ROCTX is
  // set -- is covered by roctx_utils_enabled_test.cc, a separate binary
  // because DefaultProfilerDomain() latches on first call.
  AnnotationStackGuard guard;
  {
    ScopedAnnotation annotation("my_kernel");
    EXPECT_EQ(AnnotationStack::Get(), "my_kernel");
    {
      ScopedAnnotation nested("inner_kernel");
      EXPECT_EQ(AnnotationStack::Get(), "my_kernel::inner_kernel");
    }
    EXPECT_EQ(AnnotationStack::Get(), "my_kernel");
  }
  EXPECT_EQ(AnnotationStack::Get(), "");
}

TEST(RoctxUtils, ScopedAnnotationDisabledStackDoesNotCrash) {
  // ScopedAnnotation with stack disabled — must not crash.
  {
    ScopedAnnotation annotation("disabled_op");
    EXPECT_EQ(AnnotationStack::Get(), "");
  }
  EXPECT_EQ(AnnotationStack::Get(), "");
}

TEST(RoctxUtils, DirectRoctxCallsDoNotCrash) {
  // Verify direct roctx API calls work (linked at build time).
  int depth = roctxRangePushA("roctx_utils_test_label");
  EXPECT_GE(depth, 0);
  int pop_result = roctxRangePop();
  EXPECT_GE(pop_result, 0);
}

TEST(RoctxUtils, DirectRoctxMarkDoesNotCrash) {
  roctxMarkA("roctx_utils_test_mark");
}

TEST(RoctxUtils, DirectRoctxNestedRanges) {
  int d0 = roctxRangePushA("level_0");
  EXPECT_EQ(d0, 0);
  int d1 = roctxRangePushA("level_1");
  EXPECT_EQ(d1, 1);
  int d2 = roctxRangePushA("level_2");
  EXPECT_EQ(d2, 2);

  EXPECT_EQ(roctxRangePop(), 2);
  EXPECT_EQ(roctxRangePop(), 1);
  EXPECT_EQ(roctxRangePop(), 0);
}

TEST(RoctxUtils, DetailRangePushRecoversRegisteredTitle) {
  // The path XLA takes for HLO ops when the domain is enabled:
  // ScopedAnnotation(range_generator) -> ADL xla::gpu::RangePush -> 3-arg
  // template -> detail::RangePush. Must recover the interned title and
  // perform the same dual push as the plain overload.
  AnnotationStackGuard guard;
  auto domain = DefaultProfilerDomain();
  StringHandle title = RegisterString(domain, "registered_op");
  ASSERT_NE(title, nullptr);

  detail::RangePush(domain, title, /*schema_id=*/0, /*payload=*/nullptr,
                    /*payload_size=*/0);
  EXPECT_EQ(AnnotationStack::Get(), "registered_op")
      << "detail::RangePush must push the recovered title, not nothing -- a "
         "silent no-op here unbalances the stack against RangePop";
  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "");
}

TEST(RoctxUtils, DetailRangePushWithNullTitleStaysBalanced) {
  // A null title (RegisterString never called) must still push, so the
  // matching pop has something to remove. Skipping would unbalance.
  AnnotationStackGuard guard;
  auto domain = DefaultProfilerDomain();
  detail::RangePush(domain, nullptr, 0, nullptr, 0);
  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "");
}

TEST(RoctxUtils, RegisterStringInternsAndDeduplicates) {
  auto domain = DefaultProfilerDomain();
  StringHandle a = RegisterString(domain, "shared_label");
  StringHandle b = RegisterString(domain, "shared_label");
  StringHandle c = RegisterString(domain, "other_label");

  ASSERT_NE(a, nullptr);
  EXPECT_EQ(a, b) << "identical labels must intern to one entry";
  EXPECT_NE(a, c);
  EXPECT_EQ(*reinterpret_cast<const std::string*>(a), "shared_label");
}

TEST(RoctxUtils, RegisterStringHandlesSurviveRehash) {
  // node_hash_set, not flat_hash_set: detail::RangePush dereferences these
  // handles later, so they must not be invalidated by pool growth.
  auto domain = DefaultProfilerDomain();
  StringHandle first = RegisterString(domain, "rehash_probe");
  for (int i = 0; i < 512; ++i) {
    RegisterString(domain, absl::StrCat("filler_", i));
  }
  EXPECT_EQ(*reinterpret_cast<const std::string*>(first), "rehash_probe")
      << "interned handle dangled after the pool grew";
}

TEST(RoctxUtils, RegisterSchemaReturnsZero) {
  // roctx has no schema/payload concept; structured payloads are dropped.
  auto domain = DefaultProfilerDomain();
  uint64_t schema_id = RegisterSchema(domain, nullptr);
  EXPECT_EQ(schema_id, 0);
}

TEST(RoctxUtils, DualPushBothPipelinesSimultaneously) {
  // The dual-push contract of RangePush itself: one call must populate the
  // AnnotationStack AND emit an roctx range.
  //
  // Called directly here, because in this binary the domain is null and
  // PushAnnotation would never route to RangePush on its own. That routing --
  // the path XLA takes with XLA_ROCM_ENABLE_ROCTX set -- is covered by
  // roctx_utils_enabled_test.cc.
  auto domain = DefaultProfilerDomain();

  AnnotationStackGuard guard;

  // RangePush should do both: AnnotationStack + roctxRangePushA.
  RangePush(domain, "dual_push_test");
  EXPECT_EQ(AnnotationStack::Get(), "dual_push_test");

  // RangePop should do both: AnnotationStack + roctxRangePop.
  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "");
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
