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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/base/casts.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class C32Bc32Test : public HloHardwareIndependentTestBase {};

template <typename T>
uint16_t BitsOf(T value) {
  return absl::bit_cast<uint16_t>(value);
}

template <typename T>
T FromBits(uint16_t bits) {
  return absl::bit_cast<T>(bits);
}

template <typename Carrier, typename Component>
void ExpectLiteralProtoRoundTrip(PrimitiveType type,
                                 const std::vector<uint16_t>& patterns) {
  Literal literal(
      ShapeUtil::MakeShape(type, {static_cast<int64_t>(patterns.size())}));
  auto data = literal.data<Carrier>();
  for (int i = 0; i < patterns.size(); ++i) {
    data[i] = Carrier(FromBits<Component>(patterns[i]),
                      FromBits<Component>(patterns[patterns.size() - 1 - i]));
  }

  LiteralProto proto = literal.ToProto();
  absl::StatusOr<Literal> roundtrip = Literal::CreateFromProto(proto);
  ASSERT_TRUE(roundtrip.ok()) << roundtrip.status();
  auto roundtrip_data = roundtrip->data<Carrier>();
  ASSERT_EQ(roundtrip_data.size(), patterns.size());
  for (int i = 0; i < patterns.size(); ++i) {
    EXPECT_EQ(BitsOf(roundtrip_data[i].real()), patterns[i]);
    EXPECT_EQ(BitsOf(roundtrip_data[i].imag()),
              patterns[patterns.size() - 1 - i]);
  }
}

TEST_F(C32Bc32Test, CarrierLayout) {
  static_assert(sizeof(complex32) == 4);
  static_assert(sizeof(bcomplex32) == 4);
  static_assert(offsetof(complex32, imag_value) == 2);
  static_assert(offsetof(bcomplex32, imag_value) == 2);
  static_assert(std::is_standard_layout_v<complex32>);
  static_assert(std::is_standard_layout_v<bcomplex32>);
  static_assert(std::is_trivially_copyable_v<complex32>);
  static_assert(std::is_trivially_copyable_v<bcomplex32>);
}

TEST_F(C32Bc32Test, PrimitiveUtilMappings) {
  EXPECT_EQ(primitive_util::NativeToPrimitiveType<complex32>(), C32);
  EXPECT_EQ(primitive_util::NativeToPrimitiveType<bcomplex32>(), BC32);
  EXPECT_EQ(primitive_util::BitWidth(C32), 32);
  EXPECT_EQ(primitive_util::BitWidth(BC32), 32);
  EXPECT_EQ(primitive_util::ByteWidth(C32), 4);
  EXPECT_EQ(primitive_util::ByteWidth(BC32), 4);
  EXPECT_TRUE(primitive_util::IsComplexType(C32));
  EXPECT_TRUE(primitive_util::IsComplexType(BC32));
  EXPECT_EQ(primitive_util::ComplexComponentType(C32), F16);
  EXPECT_EQ(primitive_util::ComplexComponentType(BC32), BF16);
  EXPECT_EQ(primitive_util::ComplexType(F16), C32);
  EXPECT_EQ(primitive_util::ComplexType(BF16), BC32);
  EXPECT_EQ(primitive_util::LowercasePrimitiveTypeName(C32), "c32");
  EXPECT_EQ(primitive_util::LowercasePrimitiveTypeName(BC32), "bc32");
}

TEST_F(C32Bc32Test, LiteralProtoRoundTrip) {
  // Signed zero, subnormals, max finite, infinities, quiet/signaling NaNs.
  const std::vector<uint16_t> f16_patterns = {
      0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x7bff,
      0x7c00, 0xfc00, 0x7e00, 0x7c01, 0x1234, 0xffff};
  ExpectLiteralProtoRoundTrip<complex32, half>(C32, f16_patterns);

  const std::vector<uint16_t> bf16_patterns = {
      0x0000, 0x8000, 0x0001, 0x007f, 0x0080, 0x7f7f,
      0x7f80, 0xff80, 0x7fc0, 0x7f81, 0x1234, 0xffff};
  ExpectLiteralProtoRoundTrip<bcomplex32, bfloat16>(BC32, bf16_patterns);
}

TEST_F(C32Bc32Test, ParserAndPrinterRoundTrip) {
  const absl::string_view kC32Module = R"(
HloModule c32_roundtrip
ENTRY main {
  ROOT root = c32[2] constant({(0.5, -0.25), (1.5, 2)})
}
)";
  absl::StatusOr<std::unique_ptr<HloModule>> c32_module =
      ParseAndReturnVerifiedModule(kC32Module);
  ASSERT_TRUE(c32_module.ok()) << c32_module.status();
  const std::string printed_c32 = (*c32_module)->ToString();
  absl::StatusOr<std::unique_ptr<HloModule>> reparsed_c32 =
      ParseAndReturnVerifiedModule(printed_c32);
  ASSERT_TRUE(reparsed_c32.ok()) << reparsed_c32.status();
  EXPECT_EQ((*reparsed_c32)->ToString(), printed_c32);

  const absl::string_view kBc32Module = R"(
HloModule bc32_roundtrip
ENTRY main {
  ROOT root = bc32[2] constant({(0.5, -0.25), (1.5, 2)})
}
)";
  absl::StatusOr<std::unique_ptr<HloModule>> bc32_module =
      ParseAndReturnVerifiedModule(kBc32Module);
  ASSERT_TRUE(bc32_module.ok()) << bc32_module.status();
  const std::string printed_bc32 = (*bc32_module)->ToString();
  absl::StatusOr<std::unique_ptr<HloModule>> reparsed_bc32 =
      ParseAndReturnVerifiedModule(printed_bc32);
  ASSERT_TRUE(reparsed_bc32.ok()) << reparsed_bc32.status();
  EXPECT_EQ((*reparsed_bc32)->ToString(), printed_bc32);
}

TEST_F(C32Bc32Test, EvaluatorRealImagComplexAndConvert) {
  const absl::string_view kModule = R"(
HloModule c32_eval
ENTRY main {
  real = f16[2] constant({0.5, -1})
  imag = f16[2] constant({0.25, 2})
  ROOT root = c32[2] complex(real, imag)
}
)";
  absl::StatusOr<std::unique_ptr<HloModule>> module =
      ParseAndReturnVerifiedModule(kModule);
  ASSERT_TRUE(module.ok()) << module.status();

  HloEvaluator evaluator;
  absl::StatusOr<Literal> complex_literal = evaluator.Evaluate(**module, {});
  ASSERT_TRUE(complex_literal.ok()) << complex_literal.status();
  ASSERT_EQ(complex_literal->shape().element_type(), C32);
  EXPECT_EQ(complex_literal->Get<complex32>({0}),
            complex32(half(0.5f), half(0.25f)));
  EXPECT_EQ(complex_literal->Get<complex32>({1}),
            complex32(half(-1.0f), half(2.0f)));

  absl::StatusOr<Literal> real_literal =
      evaluator.EvaluateElementwiseUnaryOp(HloOpcode::kReal, *complex_literal);
  ASSERT_TRUE(real_literal.ok()) << real_literal.status();
  EXPECT_EQ(real_literal->shape().element_type(), F16);
  EXPECT_EQ(real_literal->Get<half>({0}), half(0.5f));
  EXPECT_EQ(real_literal->Get<half>({1}), half(-1.0f));

  absl::StatusOr<Literal> imag_literal =
      evaluator.EvaluateElementwiseUnaryOp(HloOpcode::kImag, *complex_literal);
  ASSERT_TRUE(imag_literal.ok()) << imag_literal.status();
  EXPECT_EQ(imag_literal->shape().element_type(), F16);
  EXPECT_EQ(imag_literal->Get<half>({0}), half(0.25f));
  EXPECT_EQ(imag_literal->Get<half>({1}), half(2.0f));

  absl::StatusOr<Literal> converted = complex_literal->Convert(C64);
  ASSERT_TRUE(converted.ok()) << converted.status();
  EXPECT_EQ(converted->Get<complex64>({0}),
            complex64(0.5f, 0.25f));
  EXPECT_EQ(converted->Get<complex64>({1}),
            complex64(-1.0f, 2.0f));
}

TEST_F(C32Bc32Test, EvaluatorRejectsUnsupportedArithmetic) {
  const absl::string_view kModule = R"(
HloModule c32_add
ENTRY main {
  ROOT root = c32[2] constant({(0.5, -0.25), (1.5, 2)})
}
)";
  absl::StatusOr<std::unique_ptr<HloModule>> module =
      ParseAndReturnVerifiedModule(kModule);
  ASSERT_TRUE(module.ok()) << module.status();

  HloEvaluator evaluator;
  absl::StatusOr<Literal> literal = evaluator.Evaluate(**module, {});
  ASSERT_TRUE(literal.ok()) << literal.status();

  absl::StatusOr<Literal> sum =
      evaluator.EvaluateElementwiseBinaryOp(HloOpcode::kAdd, *literal, *literal);
  EXPECT_FALSE(sum.ok());
  EXPECT_EQ(sum.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_NE(sum.status().message().find("C32"), std::string::npos)
      << sum.status();

  // abs changes the result type (c32 -> f16); this also covers the
  // single-instruction evaluation entry.
  absl::StatusOr<Literal> magnitude =
      evaluator.EvaluateElementwiseUnaryOp(HloOpcode::kAbs, *literal);
  EXPECT_FALSE(magnitude.ok());
  EXPECT_NE(magnitude.status().message().find("abs"), std::string::npos)
      << magnitude.status();
}

TEST_F(C32Bc32Test, EvaluatorRejectsWholeModuleUnsupportedOp) {
  // Regression: whole-module eval must not silently compute abs((3,4)) on C32.
  const absl::string_view kModule = R"(
HloModule c32_abs
ENTRY main {
  input = c32[] constant((3, 4))
  ROOT root = f16[] abs(input)
}
)";
  absl::StatusOr<std::unique_ptr<HloModule>> module =
      ParseAndReturnVerifiedModule(kModule);
  ASSERT_TRUE(module.ok()) << module.status();

  HloEvaluator evaluator;
  absl::StatusOr<Literal> result = evaluator.Evaluate(**module, {});
  EXPECT_FALSE(result.ok());
  EXPECT_NE(result.status().message().find("abs"), std::string::npos)
      << result.status();
}

TEST_F(C32Bc32Test, EvaluatorRealImagComplexAndConvertBc32) {
  const absl::string_view kModule = R"(
HloModule bc32_eval
ENTRY main {
  real = bf16[2] constant({0.5, -1})
  imag = bf16[2] constant({0.25, 2})
  ROOT root = bc32[2] complex(real, imag)
}
)";
  absl::StatusOr<std::unique_ptr<HloModule>> module =
      ParseAndReturnVerifiedModule(kModule);
  ASSERT_TRUE(module.ok()) << module.status();

  HloEvaluator evaluator;
  absl::StatusOr<Literal> complex_literal = evaluator.Evaluate(**module, {});
  ASSERT_TRUE(complex_literal.ok()) << complex_literal.status();
  ASSERT_EQ(complex_literal->shape().element_type(), BC32);
  EXPECT_EQ(complex_literal->Get<bcomplex32>({0}),
            bcomplex32(bfloat16(0.5f), bfloat16(0.25f)));
  EXPECT_EQ(complex_literal->Get<bcomplex32>({1}),
            bcomplex32(bfloat16(-1.0f), bfloat16(2.0f)));

  absl::StatusOr<Literal> real_literal =
      evaluator.EvaluateElementwiseUnaryOp(HloOpcode::kReal, *complex_literal);
  ASSERT_TRUE(real_literal.ok()) << real_literal.status();
  EXPECT_EQ(real_literal->shape().element_type(), BF16);
  EXPECT_EQ(real_literal->Get<bfloat16>({0}), bfloat16(0.5f));
  EXPECT_EQ(real_literal->Get<bfloat16>({1}), bfloat16(-1.0f));

  absl::StatusOr<Literal> imag_literal =
      evaluator.EvaluateElementwiseUnaryOp(HloOpcode::kImag, *complex_literal);
  ASSERT_TRUE(imag_literal.ok()) << imag_literal.status();
  EXPECT_EQ(imag_literal->shape().element_type(), BF16);
  EXPECT_EQ(imag_literal->Get<bfloat16>({0}), bfloat16(0.25f));
  EXPECT_EQ(imag_literal->Get<bfloat16>({1}), bfloat16(2.0f));

  absl::StatusOr<Literal> converted = complex_literal->Convert(C128);
  ASSERT_TRUE(converted.ok()) << converted.status();
  EXPECT_EQ(converted->Get<complex128>({0}), complex128(0.5, 0.25));
  EXPECT_EQ(converted->Get<complex128>({1}), complex128(-1.0, 2.0));
}

TEST_F(C32Bc32Test, EvaluatorRejectsUnsupportedArithmeticBc32) {
  Literal literal(ShapeUtil::MakeShape(BC32, {2}));
  auto data = literal.data<bcomplex32>();
  data[0] = bcomplex32(bfloat16(0.5f), bfloat16(-0.25f));
  data[1] = bcomplex32(bfloat16(1.5f), bfloat16(2.0f));

  HloEvaluator evaluator;
  absl::StatusOr<Literal> sum =
      evaluator.EvaluateElementwiseBinaryOp(HloOpcode::kAdd, literal, literal);
  EXPECT_FALSE(sum.ok());
  EXPECT_EQ(sum.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_NE(sum.status().message().find("BC32"), std::string::npos)
      << sum.status();

  absl::StatusOr<Literal> magnitude =
      evaluator.EvaluateElementwiseUnaryOp(HloOpcode::kAbs, literal);
  EXPECT_FALSE(magnitude.ok());
  EXPECT_NE(magnitude.status().message().find("abs"), std::string::npos)
      << magnitude.status();
}

TEST_F(C32Bc32Test, EvaluatorRejectsWholeModuleUnsupportedOpBc32) {
  const absl::string_view kModule = R"(
HloModule bc32_abs
ENTRY main {
  input = bc32[] constant((3, 4))
  ROOT root = bf16[] abs(input)
}
)";
  absl::StatusOr<std::unique_ptr<HloModule>> module =
      ParseAndReturnVerifiedModule(kModule);
  ASSERT_TRUE(module.ok()) << module.status();

  HloEvaluator evaluator;
  absl::StatusOr<Literal> result = evaluator.Evaluate(**module, {});
  EXPECT_FALSE(result.ok());
  EXPECT_NE(result.status().message().find("abs"), std::string::npos)
      << result.status();
}

TEST_F(C32Bc32Test, ShapeInferenceComposesF16AndBF16) {
  const absl::string_view kF16Module = R"(
HloModule c32_compose
ENTRY main {
  real = f16[2] constant({0.5, -1})
  imag = f16[2] constant({0.25, 2})
  ROOT root = c32[2] complex(real, imag)
}
)";
  absl::StatusOr<std::unique_ptr<HloModule>> f16_module =
      ParseAndReturnVerifiedModule(kF16Module);
  ASSERT_TRUE(f16_module.ok()) << f16_module.status();

  const absl::string_view kBf16Module = R"(
HloModule bc32_compose
ENTRY main {
  real = bf16[2] constant({0.5, -1})
  imag = bf16[2] constant({0.25, 2})
  ROOT root = bc32[2] complex(real, imag)
}
)";
  absl::StatusOr<std::unique_ptr<HloModule>> bf16_module =
      ParseAndReturnVerifiedModule(kBf16Module);
  ASSERT_TRUE(bf16_module.ok()) << bf16_module.status();
}

}  // namespace
}  // namespace xla
