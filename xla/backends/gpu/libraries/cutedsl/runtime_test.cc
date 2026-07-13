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

#include "xla/backends/gpu/libraries/cutedsl/runtime.h"

#include <gtest/gtest.h>

namespace xla::gpu::cutedsl {
namespace {

TEST(CuteDslRuntimeTest, LoadsAndCachesRuntimeByLibraryName) {
  absl::StatusOr<const RuntimeApi*> first = internal::GetRuntimeApi();
  ASSERT_TRUE(first.ok()) << first.status();
  absl::StatusOr<const RuntimeApi*> second = internal::GetRuntimeApi();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(*first, *second);
  EXPECT_NE((*first)->module_create_from_bytes, nullptr);
  EXPECT_NE((*first)->module_get_function, nullptr);
  EXPECT_NE((*first)->function_run, nullptr);
  EXPECT_NE((*first)->module_destroy, nullptr);
  EXPECT_NE((*first)->get_error_name, nullptr);
  EXPECT_NE((*first)->get_error_string, nullptr);
}

}  // namespace
}  // namespace xla::gpu::cutedsl
