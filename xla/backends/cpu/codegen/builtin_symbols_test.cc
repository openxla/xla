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

#include "xla/backends/cpu/codegen/builtin_symbols.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

namespace xla::cpu {
namespace {

TEST(BuiltinSymbolsTest, ResolvesCSymbols) {
  EXPECT_EQ(LookupBuiltinSymbol("memcpy"),
            reinterpret_cast<void*>(
                static_cast<void* (*)(void*, const void*, size_t)>(memcpy)));
  EXPECT_NE(LookupBuiltinSymbol("memset"), nullptr);
  EXPECT_NE(LookupBuiltinSymbol("malloc"), nullptr);
  EXPECT_NE(LookupBuiltinSymbol("free"), nullptr);
}

TEST(BuiltinSymbolsTest, ResolvesLibmSymbols) {
  using UnaryF32 = float (*)(float);
  using UnaryF64 = double (*)(double);

  auto expf_ptr = reinterpret_cast<UnaryF32>(LookupBuiltinSymbol("expf"));
  ASSERT_NE(expf_ptr, nullptr);
  EXPECT_FLOAT_EQ(expf_ptr(1.0f), std::exp(1.0f));

  auto log_ptr = reinterpret_cast<UnaryF64>(LookupBuiltinSymbol("log"));
  ASSERT_NE(log_ptr, nullptr);
  EXPECT_DOUBLE_EQ(log_ptr(2.0), std::log(2.0));
}

TEST(BuiltinSymbolsTest, ResolvesCompilerRuntimeSymbols) {
  auto powisf2 =
      reinterpret_cast<float (*)(float, int)>(LookupBuiltinSymbol("__powisf2"));
  ASSERT_NE(powisf2, nullptr);
  EXPECT_FLOAT_EQ(powisf2(2.0f, 10), 1024.0f);

  EXPECT_NE(LookupBuiltinSymbol("__gnu_f2h_ieee"), nullptr);
  EXPECT_NE(LookupBuiltinSymbol("__gnu_h2f_ieee"), nullptr);
  EXPECT_NE(LookupBuiltinSymbol("__truncdfhf2"), nullptr);
  EXPECT_NE(LookupBuiltinSymbol("__truncdfbf2"), nullptr);
  EXPECT_NE(LookupBuiltinSymbol("__truncsfbf2"), nullptr);
}

TEST(BuiltinSymbolsTest, ReturnsNullForUnknownSymbols) {
  EXPECT_EQ(LookupBuiltinSymbol("not_a_builtin_symbol"), nullptr);
  EXPECT_EQ(LookupBuiltinSymbol(""), nullptr);
  // The global prefix (e.g. on Mac OS X) is stripped by the caller, which knows
  // the target data layout.
  EXPECT_EQ(LookupBuiltinSymbol("_memcpy"), nullptr);
}

}  // namespace
}  // namespace xla::cpu
