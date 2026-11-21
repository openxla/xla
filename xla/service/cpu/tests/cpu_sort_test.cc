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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"

namespace xla {
namespace cpu {
namespace {

using ::testing::ElementsAre;

template <typename To, typename From>
std::vector<To> BitcastEach(absl::Span<const From> values) {
  std::vector<To> out;
  out.reserve(values.size());
  absl::c_transform(values, std::back_inserter(out),
                    [](From v) { return absl::bit_cast<To>(v); });
  return out;
}

class CpuSortTest : public HloTestBase {};

// Regression test for https://github.com/openxla/xla/issues/47366: a sort
// comparator containing computation-calling ops (map, reduce) crashed the
// thunk emitter because the called computations were never emitted.
TEST_F(CpuSortTest, ComparatorCallingNestedComputations) {
  const std::string hlo_text = R"(
HloModule sort_comparator_with_nested_computations

add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

double_f32 {
  x = f32[] parameter(0)
  two = f32[] constant(2)
  ROOT mul = f32[] multiply(x, two)
}

compare {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  bcast0 = f32[2] broadcast(p0), dimensions={}
  bcast1 = f32[2] broadcast(p1), dimensions={}
  iota = f32[2] iota(), iota_dimension=0
  sum0 = f32[2] add(bcast0, iota)
  sum1 = f32[2] add(bcast1, iota)
  m0 = f32[2] map(sum0), to_apply=double_f32
  m1 = f32[2] map(sum1), to_apply=double_f32
  zero = f32[] constant(0)
  r0 = f32[] reduce(m0, zero), dimensions={0}, to_apply=add_f32
  r1 = f32[] reduce(m1, zero), dimensions={0}, to_apply=add_f32
  ROOT lt = pred[] compare(r0, r1), direction=LT
}

ENTRY entry {
  p = f32[8] parameter(0)
  ROOT sort = f32[8] sort(p), dimensions={0}, is_stable=true, to_apply=compare
}
)";

  // The comparator orders by 4*p+2, which is equivalent to p0 < p1.
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  Literal input = LiteralUtil::CreateR1<float>({3, -1, 7, 0, -5, 2, 8, 1});
  TF_ASSERT_OK_AND_ASSIGN(const Literal result,
                          Execute(std::move(module), {&input}));
  LiteralTestUtil::ExpectR1Equal<float>({-5, -1, 0, 1, 2, 3, 7, 8}, result);
}

TEST_F(CpuSortTest,
       CanonicalizedTotalOrderAscendingF32SortPreservesBitsAndStability) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule numpy_total_order_ascending_f32_sort

compare_lt_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  lhs_i = s32[] parameter(2)
  rhs_i = s32[] parameter(3)
  lhs_ne = pred[] compare(lhs, lhs), direction=NE
  c_nan = f32[] constant(nan)
  c_zero = f32[] constant(0)
  lhs_eq_zero = pred[] compare(lhs, c_zero), direction=EQ
  lhs_nz = f32[] select(lhs_eq_zero, c_zero, lhs)
  lhs_canon = f32[] select(lhs_ne, c_nan, lhs_nz)
  rhs_ne = pred[] compare(rhs, rhs), direction=NE
  rhs_eq_zero = pred[] compare(rhs, c_zero), direction=EQ
  rhs_nz = f32[] select(rhs_eq_zero, c_zero, rhs)
  rhs_canon = f32[] select(rhs_ne, c_nan, rhs_nz)
  ROOT cmp = pred[] compare(lhs_canon, rhs_canon), direction=LT, order=TOTAL
}

ENTRY entry {
  f32_keys = f32[8] parameter(0)
  indices = s32[8] parameter(1)
  ROOT asc_f32 = (f32[8], s32[8]) sort(f32_keys, indices), dimensions={0}, is_stable=true, to_apply=compare_lt_f32
}
)hlo";

  const uint32_t kF32PosNan = 0x7fc00111u;
  const uint32_t kF32NegNan = 0xffc00222u;
  const uint32_t kF32PosZero = 0x00000000u;
  const uint32_t kF32NegZero = 0x80000000u;
  const uint32_t kF32PosTwo = absl::bit_cast<uint32_t>(2.0f);
  const uint32_t kF32NegTwo = absl::bit_cast<uint32_t>(-2.0f);
  const uint32_t kF32PosInf =
      absl::bit_cast<uint32_t>(std::numeric_limits<float>::infinity());
  const uint32_t kF32NegInf =
      absl::bit_cast<uint32_t>(-std::numeric_limits<float>::infinity());

  // Place +0.0 before -0.0 so a broken uncanonicalized IEEE TOTAL sort
  // (-0.0 < +0.0) would flip their relative order instead of preserving it.
  Literal f32_keys = LiteralUtil::CreateR1<float>(BitcastEach<float, uint32_t>({
      kF32NegNan,
      kF32PosTwo,
      kF32PosZero,
      kF32NegInf,
      kF32NegZero,
      kF32NegTwo,
      kF32PosInf,
      kF32PosNan,
  }));
  Literal indices = LiteralUtil::CreateR1<int32_t>({0, 1, 2, 3, 4, 5, 6, 7});

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(const Literal result,
                       Execute(std::move(module), {&f32_keys, &indices}));

  EXPECT_THAT(BitcastEach<uint32_t>(result.data<float>({0})),
              ElementsAre(kF32NegInf, kF32NegTwo, kF32PosZero, kF32NegZero,
                          kF32PosTwo, kF32PosInf, kF32NegNan, kF32PosNan));
  EXPECT_THAT(result.data<int32_t>({1}), ElementsAre(3, 5, 2, 4, 1, 6, 0, 7));
}

TEST_F(CpuSortTest,
       CanonicalizedTotalOrderDescendingBf16SortPreservesBitsAndStability) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule numpy_total_order_descending_bf16_sort

compare_gt_bf16 {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  lhs_i = s32[] parameter(2)
  rhs_i = s32[] parameter(3)
  lhs_ne = pred[] compare(lhs, lhs), direction=NE
  c_nan = bf16[] constant(nan)
  c_zero = bf16[] constant(0)
  lhs_eq_zero = pred[] compare(lhs, c_zero), direction=EQ
  lhs_nz = bf16[] select(lhs_eq_zero, c_zero, lhs)
  lhs_canon = bf16[] select(lhs_ne, c_nan, lhs_nz)
  rhs_ne = pred[] compare(rhs, rhs), direction=NE
  rhs_eq_zero = pred[] compare(rhs, c_zero), direction=EQ
  rhs_nz = bf16[] select(rhs_eq_zero, c_zero, rhs)
  rhs_canon = bf16[] select(rhs_ne, c_nan, rhs_nz)
  ROOT cmp = pred[] compare(lhs_canon, rhs_canon), direction=GT, order=TOTAL
}

ENTRY entry {
  bf16_keys = bf16[8] parameter(0)
  indices = s32[8] parameter(1)
  ROOT desc_bf16 = (bf16[8], s32[8]) sort(bf16_keys, indices), dimensions={0}, is_stable=true, to_apply=compare_gt_bf16
}
)hlo";

  const uint16_t kBf16PosNan = 0x7fc1u;
  const uint16_t kBf16NegNan = 0xffc2u;
  const uint16_t kBf16PosZero = 0x0000u;
  const uint16_t kBf16NegZero = 0x8000u;
  const uint16_t kBf16PosTwo = absl::bit_cast<uint16_t>(bfloat16(2.0f));
  const uint16_t kBf16NegTwo = absl::bit_cast<uint16_t>(bfloat16(-2.0f));
  const uint16_t kBf16PosInf = 0x7f80u;
  const uint16_t kBf16NegInf = 0xff80u;

  // Place -0.0 before +0.0 so a broken uncanonicalized IEEE TOTAL descending
  // sort (+0.0 > -0.0) would flip their relative order instead of preserving
  // it.
  Literal bf16_keys =
      LiteralUtil::CreateR1<bfloat16>(BitcastEach<bfloat16, uint16_t>({
          kBf16NegNan,
          kBf16PosTwo,
          kBf16NegZero,
          kBf16NegInf,
          kBf16PosZero,
          kBf16NegTwo,
          kBf16PosInf,
          kBf16PosNan,
      }));
  Literal indices = LiteralUtil::CreateR1<int32_t>({0, 1, 2, 3, 4, 5, 6, 7});

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(const Literal result,
                       Execute(std::move(module), {&bf16_keys, &indices}));

  EXPECT_THAT(
      BitcastEach<uint16_t>(result.data<bfloat16>({0})),
      ElementsAre(kBf16NegNan, kBf16PosNan, kBf16PosInf, kBf16PosTwo,
                  kBf16NegZero, kBf16PosZero, kBf16NegTwo, kBf16NegInf));
  EXPECT_THAT(result.data<int32_t>({1}), ElementsAre(0, 7, 6, 1, 2, 4, 5, 3));
}

}  // namespace
}  // namespace cpu
}  // namespace xla
