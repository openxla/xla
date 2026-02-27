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

#include "xla/lazy.h"

#include <string>

#include <gtest/gtest.h>
#include "xla/tsl/platform/test_benchmark.h"

namespace xla {
namespace {

TEST(LazyTest, Works) {
  Lazy<std::string> lazy_string([]() { return "hello"; });
  EXPECT_FALSE(lazy_string.has_value());
  EXPECT_EQ(lazy_string.get(), "hello");
  EXPECT_TRUE(lazy_string.has_value());
}

struct ClassWithLazyMember {
  [[no_unique_address]] Lazy<int> lazy_int;
  int something_else;
};

TEST(LazyTest, CanReuseTailPadding) {
  EXPECT_EQ(sizeof(ClassWithLazyMember), 48);
}

void BM_InitializeLazy(benchmark::State& state) {
  for (auto _ : state) {
    Lazy<int> lazy_int([]() { return 42; });
    benchmark::DoNotOptimize(lazy_int);
    int i = lazy_int.get();
    benchmark::DoNotOptimize(i);
  }
}
BENCHMARK(BM_InitializeLazy);

}  // namespace
}  // namespace xla
