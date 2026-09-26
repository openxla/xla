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

// The opt-in guarantee: with XLA_ROCM_ENABLE_ROCTX absent, ROCTX stays off.
//
// A third binary, because this is the one configuration the other two cannot
// reach. Both of those pin the variable via the Bazel `env` attribute ("0" and
// "1"), which is deliberate -- it stops a stray --test_env from retargeting
// them -- but it also means ReadBoolFromEnvVar never falls back to its
// default_val in either. Flipping that default from false to true therefore
// leaves all 24 of their cases green while turning the emitter on for every
// ROCm process that has not opted in.
//
// This target intentionally has NO env attribute: Bazel scrubs the test
// environment, so the variable is genuinely unset here.

#include <stdlib.h>  // for unsetenv, which is POSIX rather than ISO C

#include "xla/tsl/platform/test.h"
#include "tsl/profiler/lib/range_annotations.h"

namespace tsl {
namespace profiler {
namespace {

TEST(RoctxRangeAnnotationsDefault, DomainIsNullWhenEnvVarIsAbsent) {
  // Belt and braces: assert the premise rather than trusting the environment
  // to be clean, so this fails loudly if an `env` attribute is ever added to
  // this target or a --test_env leaks a value in.
  ASSERT_EQ(getenv("XLA_ROCM_ENABLE_ROCTX"), nullptr)
      << "this target must run with XLA_ROCM_ENABLE_ROCTX unset; it exists to "
         "cover the default_val path that the pinned binaries never reach";

  EXPECT_EQ(DefaultProfilerDomain(), nullptr)
      << "ROCTX must be off unless explicitly opted in -- this is the whole "
         "safety premise of the feature";
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
