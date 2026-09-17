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

// Tests for roctx_range_annotations.cc — the ROCm dual-push implementation of
// range_annotations.h. Verifies that RangePush/RangePop populate
// AnnotationStack (Pipeline A) and emit roctx markers (Pipeline B).

#include <stdlib.h>  // for setenv, which is POSIX rather than ISO C

#include <cstdint>
#include <string>
#include <thread>  // NOLINT(build/c++11) -- matching the surrounding tsl tests
#include <vector>

#include "absl/strings/str_cat.h"
#include "rocm/include/rocprofiler-sdk-roctx/roctx.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#include "tsl/profiler/lib/range_annotations.h"
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

TEST(RoctxRangeAnnotations, DefaultProfilerDomainDefaultsToNullAndIsStable) {
  // Two properties, both load-bearing.
  //
  // 1. Default off, matching CUDA's enable_nvtx_tracking=false. Tests run
  //    without XLA_ROCM_ENABLE_ROCTX set, so the domain must be null and
  //    PushAnnotation must take the AnnotationStack branch.
  // 2. LATCHED. scoped_annotation.h branches on this in both PushAnnotation
  //    and PopAnnotation; a handle that changed mid-process would push down
  //    one path and pop down the other, unbalancing both stacks.
  ProfilerDomainHandle first = DefaultProfilerDomain();
  ASSERT_EQ(first, nullptr)
      << "Domain must default to null; XLA_ROCM_ENABLE_ROCTX opts in. This "
         "binary pins it to \"0\" via the env attribute on "
         "roctx_range_annotations_test in BUILD -- if that attribute was "
         "dropped, or --test_env leaked a \"1\" in, every other test in this "
         "file is silently exercising the enabled path instead";
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(DefaultProfilerDomain(), first)
        << "DefaultProfilerDomain() must be stable for the process lifetime";
  }
}

// The latch itself, not merely the stability that a constant environment would
// produce anyway. Repeatedly calling DefaultProfilerDomain() under an
// unchanging env cannot distinguish a latched static from a getenv on every
// call, so the property the header calls a correctness requirement would go
// untested. Here the env is changed underneath it and the handle must not move.
//
// Mutating the environment is safe precisely BECAUSE the value is latched: the
// first call below fixes it for the process, so later tests -- in any
// --gtest_shuffle order -- still observe null.
TEST(RoctxRangeAnnotations, DomainIsLatchedAgainstLaterEnvChanges) {
  ProfilerDomainHandle before = DefaultProfilerDomain();
  ASSERT_EQ(before, nullptr);

  ASSERT_EQ(setenv("XLA_ROCM_ENABLE_ROCTX", "1", /*overwrite=*/1), 0);
  EXPECT_EQ(DefaultProfilerDomain(), nullptr)
      << "the handle must be computed once and latched; re-reading the "
         "environment per call would let it flip between a ScopedAnnotation's "
         "push and its pop, stranding one of the two stacks";

  ASSERT_EQ(setenv("XLA_ROCM_ENABLE_ROCTX", "0", /*overwrite=*/1), 0);
}

TEST(RoctxRangeAnnotations, RangePushPopulatesAnnotationStack) {
  auto domain = DefaultProfilerDomain();

  AnnotationStackGuard guard;
  RangePush(domain, "test_op");
  EXPECT_EQ(AnnotationStack::Get(), "test_op");
  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "");
}

TEST(RoctxRangeAnnotations, NestedPushPopMaintainsAnnotationStack) {
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

TEST(RoctxRangeAnnotations, PushPopWithAnnotationStackDisabled) {
  auto domain = DefaultProfilerDomain();

  // AnnotationStack is NOT enabled. RangePush/RangePop must not crash
  // and must not populate the stack.
  RangePush(domain, "ignored_op");
  EXPECT_EQ(AnnotationStack::Get(), "");
  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "");
}

TEST(RoctxRangeAnnotations, ScopedAnnotationIntegration) {
  // Full chain in THIS binary, where the domain is null:
  // ScopedAnnotation -> PushAnnotation -> AnnotationStack::PushAnnotation.
  // The domain path -- which is what XLA takes when XLA_ROCM_ENABLE_ROCTX is
  // set -- is covered by roctx_range_annotations_enabled_test.cc, a separate
  // binary because DefaultProfilerDomain() latches on first call.
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

TEST(RoctxRangeAnnotations, ScopedAnnotationDisabledStackDoesNotCrash) {
  // ScopedAnnotation with stack disabled — must not crash.
  {
    ScopedAnnotation annotation("disabled_op");
    EXPECT_EQ(AnnotationStack::Get(), "");
  }
  EXPECT_EQ(AnnotationStack::Get(), "");
}

TEST(RoctxRangeAnnotations, DirectRoctxCallsDoNotCrash) {
  // The roctx symbols resolve at link time and the calls are safe with no
  // profiler attached. roctxMarkA returns void, so linking and not crashing is
  // the whole assertion available for it.
  const int depth = roctxRangePushA("roctx_range_annotations_test_label");
  EXPECT_GE(depth, 0);
  EXPECT_EQ(roctxRangePop(), depth);
  roctxMarkA("roctx_range_annotations_test_mark");
}

TEST(RoctxRangeAnnotations, DirectRoctxNestedRanges) {
  // Depths are relative to whatever this thread arrives with: tsl_cc_test
  // injects --gtest_shuffle, so this test can run at any point in the binary.
  const int d0 = roctxRangePushA("level_0");
  ASSERT_GE(d0, 0);
  EXPECT_EQ(roctxRangePushA("level_1"), d0 + 1);
  EXPECT_EQ(roctxRangePushA("level_2"), d0 + 2);

  EXPECT_EQ(roctxRangePop(), d0 + 2);
  EXPECT_EQ(roctxRangePop(), d0 + 1);
  EXPECT_EQ(roctxRangePop(), d0);
}

TEST(RoctxRangeAnnotations, DetailRangePushPushesRegisteredTitle) {
  // The path XLA takes for HLO ops when the domain is enabled:
  // ScopedAnnotation(range_generator) -> ADL xla::gpu::RangePush -> 3-arg
  // template -> detail::RangePush. The StringHandle comes from RegisterString,
  // and detail::RangePush must resolve it back to text and perform the same
  // dual push as the plain overload.
  AnnotationStackGuard guard;
  auto domain = DefaultProfilerDomain();

  StringHandle title = RegisterString(domain, "registered_op");
  detail::RangePush(domain, title, /*schema_id=*/0, /*payload=*/nullptr,
                    /*payload_size=*/0);
  EXPECT_EQ(AnnotationStack::Get(), "registered_op")
      << "detail::RangePush must push the registered title, not nothing -- a "
         "silent no-op here unbalances the stack against RangePop";
  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "");
}

TEST(RoctxRangeAnnotations, DetailRangePushWithNullTitleStaysBalanced) {
  // A null title must still push, so the matching pop has something to remove.
  // Asserted from an enclosing range: if the push is skipped, the pop takes the
  // enclosing entry instead, which is the unbalancing this guards against. A
  // bare "stack is empty afterwards" check would pass with the push skipped.
  AnnotationStackGuard guard;
  auto domain = DefaultProfilerDomain();

  RangePush(domain, "outer");
  detail::RangePush(domain, nullptr, /*schema_id=*/0, /*payload=*/nullptr,
                    /*payload_size=*/0);
  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "outer")
      << "the null-title push must have happened; otherwise this RangePop "
         "removed the enclosing range";
  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "");
}

TEST(RoctxRangeAnnotations, DetailRangePushDropsSchemaAndPayload) {
  // roctx has no schema/payload concept, so both are dropped rather than read.
  // Passing a real schema_id and a non-null buffer pins that: the only tests
  // that call this otherwise pass 0/nullptr, where "dropped" and "happens to be
  // unused" are indistinguishable.
  AnnotationStackGuard guard;
  auto domain = DefaultProfilerDomain();

  StringHandle title = RegisterString(domain, "payload_op");
  const char payload[] = "ignored payload bytes";
  detail::RangePush(domain, title, /*schema_id=*/0x1234, payload,
                    sizeof(payload));
  EXPECT_EQ(AnnotationStack::Get(), "payload_op");
  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "");
}

TEST(RoctxRangeAnnotations, RegisterStringInternsAndOutlivesCaller) {
  // roctx has no nvtxDomainRegisterStringA equivalent, so the handle points at
  // an interned copy. Callers pass temporaries (annotation.cc does), so the
  // copy is what makes the handle usable after the caller's string is gone.
  //
  // Both labels are deliberately longer than the libstdc++ SSO threshold and
  // read back only after BOTH registrations. A short label would sit in the
  // temporary's stack slot, and an implementation that returned the caller's
  // c_str() unchanged would still read back correctly from that reused slot --
  // so a short-label version of this test passes with interning removed.
  auto domain = DefaultProfilerDomain();
  constexpr char kLabelA[] = "Thunk:#name=fusion.1,hlo_op=fusion.1#";
  constexpr char kLabelB[] = "Thunk:#name=fusion.2,hlo_op=fusion.2#";

  StringHandle a = RegisterString(domain, std::string(kLabelA));
  StringHandle b = RegisterString(domain, std::string(kLabelB));
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_NE(a, b);
  EXPECT_STREQ(reinterpret_cast<const char*>(a), kLabelA);
  EXPECT_STREQ(reinterpret_cast<const char*>(b), kLabelB);

  // Equal strings share one entry: repeated annotations must not grow the
  // table without bound.
  EXPECT_EQ(RegisterString(domain, std::string(kLabelA)), a);
}

TEST(RoctxRangeAnnotations, NamingAndMarkMemoryLinkAndDoNotCrash) {
  // A link-and-smoke test, and deliberately nothing more. These calls have no
  // in-process observable: the roctx wrappers discard their return value, and
  // rocm_tracer.cc does not subscribe MARKER_NAME_API, so whether they reached
  // roctx at all cannot be asserted from here. In particular this does NOT pin
  // the decision to leave them ungated on the domain -- adding such a gate
  // would still pass. NameStream is excluded: it needs a real hipStream_t.
  NameCurrentThread("roctx_test_thread");
  NameDevice(/*device_id=*/0, "roctx_test_device");

  // Permanently a no-op on ROCm; it must accept a null stream without reading
  // the address it is handed.
  const char buffer[8] = {};
  MarkMemoryInitialized(buffer, sizeof(buffer), /*stream=*/nullptr);
}

TEST(RoctxRangeAnnotations, RegisterStringIsSafeUnderConcurrentCallers) {
  // annotation.cc reaches RegisterString from every XLA thread that annotates
  // an HLO op, so the insert is genuinely concurrent. Without the mutex this is
  // a data race on the node_hash_set. Measured: deleting the lock still passes
  // 10/10 unsanitized runs and fails immediately under -fsanitize=thread, so
  // this case is a vehicle for TSAN rather than a standalone assertion.
  //
  // Asserting handle identity per label also pins the dedup: a torn insert can
  // leave two entries for one string, which would hand back two handles.
  auto domain = DefaultProfilerDomain();
  constexpr int kThreads = 8;
  constexpr int kLabels = 32;

  auto label = [](int i) {
    return absl::StrCat("Thunk:#name=concurrent.", i, ",hlo_op=concurrent.", i,
                        "#");
  };

  // Every thread registers the same label set, so each label is contended.
  std::vector<std::vector<StringHandle>> seen(kThreads);
  {
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t] {
        seen[t].reserve(kLabels);
        for (int i = 0; i < kLabels; ++i) {
          seen[t].push_back(RegisterString(domain, label(i)));
        }
      });
    }
    for (std::thread& thread : threads) thread.join();
  }

  for (int i = 0; i < kLabels; ++i) {
    ASSERT_NE(seen[0][i], nullptr);
    EXPECT_STREQ(reinterpret_cast<const char*>(seen[0][i]), label(i).c_str());
    for (int t = 1; t < kThreads; ++t) {
      EXPECT_EQ(seen[t][i], seen[0][i])
          << "label " << i << " interned to two different addresses; the "
          << "table must dedup across threads";
    }
  }
}

TEST(RoctxRangeAnnotations, RegisterSchemaReturnsZero) {
  // roctx has no schema/payload concept; structured payloads are dropped.
  auto domain = DefaultProfilerDomain();
  uint64_t schema_id = RegisterSchema(domain, nullptr);
  EXPECT_EQ(schema_id, 0);
}

TEST(RoctxRangeAnnotations, DualPushBothPipelinesSimultaneously) {
  // The dual-push contract of RangePush itself: one call must populate the
  // AnnotationStack AND emit an roctx range.
  //
  // Called directly here, because in this binary the domain is null and
  // PushAnnotation would never route to RangePush on its own. That routing --
  // the path XLA takes with XLA_ROCM_ENABLE_ROCTX set -- is covered by
  // roctx_range_annotations_enabled_test.cc.
  auto domain = DefaultProfilerDomain();

  AnnotationStackGuard guard;

  // Probe the roctx depth this thread arrives with, so Pipeline B can be
  // asserted relatively (--gtest_shuffle means the arrival depth is not 0).
  const int base = roctxRangePushA("depth_probe");
  ASSERT_GE(base, 0);
  ASSERT_EQ(roctxRangePop(), base);

  RangePush(domain, "dual_push_test");
  EXPECT_EQ(AnnotationStack::Get(), "dual_push_test");  // Pipeline A

  const int inner = roctxRangePushA("depth_probe");  // Pipeline B
  ASSERT_GE(inner, 0);
  EXPECT_EQ(inner, base + 1)
      << "RangePush must emit a roctx range as well as pushing the "
         "AnnotationStack; without this the test passes with Pipeline B "
         "deleted";
  ASSERT_EQ(roctxRangePop(), inner);

  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "");
  const int after = roctxRangePushA("depth_probe");
  ASSERT_GE(after, 0);
  EXPECT_EQ(after, base) << "RangePop must pop the roctx range too";
  ASSERT_EQ(roctxRangePop(), base);
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
