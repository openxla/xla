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

// Tests for roctx_utils.cc with the emitter ENABLED.
//
// A separate binary from roctx_utils_test.cc, not a --gtest_filter over it:
// DefaultProfilerDomain() latches its value in a function-local static on the
// first call, so a single process can only ever observe one configuration.
// Bazel sets XLA_ROCM_ENABLE_ROCTX=1 for this target (see the `env` attribute
// on roctx_utils_enabled_test in BUILD), which makes the latch resolve to a
// non-null domain here and to null there.
//
// This is the only place the shipped feature is actually exercised: with a
// null domain, scoped_annotation.h never routes through RangePush at all, so
// every assertion in the disabled binary passes whether the emitter works or
// not.

#include <string>

#include "absl/strings/match.h"
#include "rocm/include/rocprofiler-sdk-roctx/roctx.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#include "tsl/profiler/lib/nvtx_utils.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace tsl {
namespace profiler {
namespace {

class AnnotationStackGuard {
 public:
  AnnotationStackGuard() { AnnotationStack::Enable(true); }
  ~AnnotationStackGuard() { AnnotationStack::Enable(false); }
};

TEST(RoctxUtilsEnabled, DomainIsNonNullWhenEnvVarIsSet) {
  // The premise every other test in this file depends on. If this fails, the
  // Bazel `env` attribute was dropped and the rest are silently testing the
  // disabled path again.
  ASSERT_NE(DefaultProfilerDomain(), nullptr)
      << "XLA_ROCM_ENABLE_ROCTX=1 must produce a non-null domain; check the "
         "env attribute on this test target";
}

TEST(RoctxUtilsEnabled, DomainStaysLatchedAcrossCalls) {
  ProfilerDomainHandle first = DefaultProfilerDomain();
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(DefaultProfilerDomain(), first)
        << "the handle must be stable: scoped_annotation.h branches on it in "
           "both PushAnnotation and PopAnnotation, and an asymmetric read "
           "strands one of the two stacks";
  }
}

// The routing this PR exists to enable. With a non-null domain,
// PushAnnotation takes its `domain != nullptr` branch and calls RangePush,
// whose dual push must still populate the AnnotationStack -- otherwise kTfOp
// disappears from every kernel event, which is the regression that motivated
// returning null in the first place.
TEST(RoctxUtilsEnabled, ScopedAnnotationStillPopulatesAnnotationStack) {
  AnnotationStackGuard guard;
  {
    ScopedAnnotation annotation("enabled_path_op");
    EXPECT_EQ(AnnotationStack::Get(), "enabled_path_op")
        << "the domain path must dual-push; if only roctx is pushed, kTfOp is "
           "lost on every kernel event";
  }
  EXPECT_EQ(AnnotationStack::Get(), "")
      << "PopAnnotation must unwind the domain path symmetrically";
}

TEST(RoctxUtilsEnabled, NestedScopedAnnotationsStayBalanced) {
  AnnotationStackGuard guard;
  {
    ScopedAnnotation outer("outer_op");
    EXPECT_EQ(AnnotationStack::Get(), "outer_op");
    {
      ScopedAnnotation inner("inner_op");
      EXPECT_EQ(AnnotationStack::Get(), "outer_op::inner_op");
    }
    EXPECT_EQ(AnnotationStack::Get(), "outer_op");
  }
  EXPECT_EQ(AnnotationStack::Get(), "");
}

// Pipeline B, asserted from the inside. This is THE test for the feature: it
// fails if ScopedAnnotation stops emitting a roctx range at all.
//
// An earlier version only compared depth before and after the scope, which is
// one-sided -- it catches a leaked push but not an absent one, since "no push
// and no pop" also returns to the starting depth. Measuring INSIDE the scope
// is what distinguishes the two. DirectRoctxNestedRanges in the disabled
// binary establishes that nested pushes report increasing depth, so base+1 is
// a reliable expectation.
//
// Depths are relative, not absolute, so the test does not assume the thread
// arrives with a clean roctx stack.
TEST(RoctxUtilsEnabled, ScopedAnnotationEmitsARoctxRange) {
  // Deliberately no AnnotationStackGuard: only Pipeline B should move.
  const int base = roctxRangePushA("depth_probe");
  ASSERT_GE(base, 0);
  ASSERT_EQ(roctxRangePop(), base);

  {
    ScopedAnnotation annotation("stack_disabled_op");

    const int inner = roctxRangePushA("depth_probe");
    ASSERT_GE(inner, 0);
    EXPECT_EQ(inner, base + 1)
        << "ScopedAnnotation must have pushed a roctx range -- emitting that "
           "range is the entire point of this change, and without this "
           "assertion the whole suite passes with Pipeline B deleted";
    ASSERT_EQ(roctxRangePop(), inner);

    EXPECT_EQ(AnnotationStack::Get(), "")
        << "AnnotationStack is disabled, so Pipeline A must stay empty";
  }

  const int after = roctxRangePushA("depth_probe");
  ASSERT_GE(after, 0);
  EXPECT_EQ(after, base)
      << "roctx depth must return to its starting value; a leaked push would "
         "truncate every subsequent user range";
  ASSERT_EQ(roctxRangePop(), base);
}

// detail::RangePush is the entry point for HLO-op annotations (the two-
// generator ScopedAnnotation overload reaches it via ADL). With the domain
// enabled it must recover the interned title and dual push, not silently do
// nothing while ~ScopedAnnotation goes on to pop.
TEST(RoctxUtilsEnabled, DetailRangePushDualPushesRecoveredTitle) {
  AnnotationStackGuard guard;
  auto domain = DefaultProfilerDomain();
  StringHandle title = RegisterString(domain, "hlo_op_title");
  ASSERT_NE(title, nullptr);

  const int base = roctxRangePushA("depth_probe");
  ASSERT_GE(base, 0);
  ASSERT_EQ(roctxRangePop(), base);

  detail::RangePush(domain, title, /*schema_id=*/0, /*payload=*/nullptr,
                    /*payload_size=*/0);
  EXPECT_EQ(AnnotationStack::Get(), "hlo_op_title");  // Pipeline A

  const int inner = roctxRangePushA("depth_probe");  // Pipeline B
  ASSERT_GE(inner, 0);
  EXPECT_EQ(inner, base + 1)
      << "the HLO-op path must emit a roctx range too, not just populate the "
         "AnnotationStack";
  ASSERT_EQ(roctxRangePop(), inner);

  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "");
  const int after = roctxRangePushA("depth_probe");
  ASSERT_GE(after, 0);
  EXPECT_EQ(after, base) << "detail::RangePush/RangePop must balance on roctx";
  ASSERT_EQ(roctxRangePop(), base);
}

// The intern pool is process-lifetime and never freed, so it is bounded. Past
// the cap RegisterString hands back a diagnostic sentinel rather than growing
// or returning null -- null would push "" onto the AnnotationStack and leave
// stray "::" separators for ParseAnnotationStack to split on.
TEST(RoctxUtilsEnabled, RegisterStringTruncatesOverlongStrings) {
  auto domain = DefaultProfilerDomain();
  // Comfortably past the 4 KiB per-string cap.
  const std::string huge(200000, 'x');
  StringHandle handle = RegisterString(domain, huge);
  ASSERT_NE(handle, nullptr);
  const auto* interned = reinterpret_cast<const std::string*>(handle);

  EXPECT_LE(interned->size(), 4096u)
      << "an uncapped intern would retain the caller's full string forever; "
         "InstructionAnnotation registers whole fusion bodies";
  // Truncation must announce itself, as CUDA's does. A bare prefix looks like
  // a legitimately short label, so a reader cannot tell a real annotation from
  // a cut one.
  EXPECT_TRUE(absl::EndsWith(*interned, "...[truncated]"))
      << "truncated labels must carry the marker, got: " << *interned;
  EXPECT_TRUE(absl::StartsWith(*interned, "xxxx"));
}

// Short strings -- which is every label ROCm actually reads, since only
// nvtx_name_ is dereferenced -- must pass through untouched.
TEST(RoctxUtilsEnabled, RegisterStringLeavesNormalLabelsIntact) {
  auto domain = DefaultProfilerDomain();
  const std::string title = "Thunk:#name=fusion.3,hlo_op=fusion.3#";
  const auto* interned =
      reinterpret_cast<const std::string*>(RegisterString(domain, title));
  ASSERT_NE(interned, nullptr);
  EXPECT_EQ(*interned, title) << "a normal annotation title must not be cut";
}

TEST(RoctxUtilsEnabled, RegisterStringInternsAndDeduplicates) {
  auto domain = DefaultProfilerDomain();
  StringHandle a = RegisterString(domain, "shared");
  StringHandle b = RegisterString(domain, "shared");
  StringHandle c = RegisterString(domain, "distinct");
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(a, b) << "identical labels must share one pool entry";
  EXPECT_NE(a, c);
  EXPECT_EQ(*reinterpret_cast<const std::string*>(a), "shared");
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
