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

// An unrecognised XLA_ROCM_ENABLE_ROCTX value must leave ROCTX OFF.
//
// ReadBoolFromEnvVar accepts only 0/false/1/true; "yes" and "on" are rejected.
// A fourth binary because, like the default and the two pinned configurations,
// this is a whole-process state that the latch fixes on first call.
//
// Without this target, inverting the error branch so an unrecognised value
// enables the emitter passes every other case: none of them ever hands
// ReadBoolFromEnvVar a value it rejects.

#include <stdlib.h>  // for getenv, which is POSIX rather than ISO C

#include <cstring>

#include "xla/tsl/platform/test.h"
#include "tsl/profiler/lib/range_annotations.h"

namespace tsl {
namespace profiler {
namespace {

TEST(RoctxRangeAnnotationsBadEnv, UnrecognisedValueLeavesRoctxDisabled) {
  const char* raw = getenv("XLA_ROCM_ENABLE_ROCTX");
  ASSERT_NE(raw, nullptr) << "this target pins XLA_ROCM_ENABLE_ROCTX to an "
                             "unparseable value via its env attribute";
  ASSERT_STREQ(raw, "yes");

  EXPECT_EQ(DefaultProfilerDomain(), nullptr)
      << "an unrecognised value must fall back to disabled, not enabled -- "
         "silently turning the emitter ON for a user who wrote \"yes\" "
         "inverts the documented contract";
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
