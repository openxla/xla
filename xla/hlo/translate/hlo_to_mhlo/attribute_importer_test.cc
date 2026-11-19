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

// Unit tests for HLO-to-StableHLO attribute conversion helpers in
// attribute_importer.cc, including exhaustive PrecisionConfig::Algorithm
// coverage and rejection of unknown enum values.

#include "xla/hlo/translate/hlo_to_mhlo/attribute_importer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "tsl/platform/protobuf.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(AttributeImporterTest, ConvertDotAlgorithmCoversAllDefinedEnumValues) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir::Builder builder(&context);

  const tsl::protobuf::EnumDescriptor* descriptor =
      tsl::protobuf::GetEnumDescriptor<PrecisionConfig::Algorithm>();
  for (int i = 0; i < descriptor->value_count(); ++i) {
    auto algorithm =
        static_cast<PrecisionConfig::Algorithm>(descriptor->value(i)->number());
    ASSERT_OK_AND_ASSIGN(mlir::stablehlo::DotAlgorithmAttr attr,
                         stablehlo::ConvertDotAlgorithm(algorithm, &builder));
    if (algorithm == PrecisionConfig::ALG_UNSET) {
      EXPECT_FALSE(attr);
      continue;
    }
    ASSERT_TRUE(attr) << "Expected non-empty DotAlgorithmAttr for "
                      << PrecisionConfig::Algorithm_Name(algorithm);
    EXPECT_TRUE(mlir::hlo::isKnownDotAlgorithm(
        attr.getLhsPrecisionType(), attr.getRhsPrecisionType(),
        attr.getAccumulationType(), attr.getLhsComponentCount(),
        attr.getRhsComponentCount(), attr.getNumPrimitiveOperations(),
        attr.getAllowImpreciseAccumulation()))
        << "StableHLO did not recognize converted attribute for "
        << PrecisionConfig::Algorithm_Name(algorithm);
  }
}

TEST(AttributeImporterTest, ConvertDotAlgorithmBF16BF16F32X9) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir::Builder builder(&context);

  ASSERT_OK_AND_ASSIGN(
      mlir::stablehlo::DotAlgorithmAttr attr,
      stablehlo::ConvertDotAlgorithm(PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9,
                                     &builder));
  ASSERT_TRUE(attr);
  EXPECT_TRUE(attr.getLhsPrecisionType().isBF16());
  EXPECT_TRUE(attr.getRhsPrecisionType().isBF16());
  EXPECT_TRUE(attr.getAccumulationType().isF32());
  EXPECT_EQ(attr.getLhsComponentCount(), 1);
  EXPECT_EQ(attr.getRhsComponentCount(), 1);
  EXPECT_EQ(attr.getNumPrimitiveOperations(), 9);
  EXPECT_FALSE(attr.getAllowImpreciseAccumulation());
}

TEST(AttributeImporterTest, ConvertDotAlgorithmRejectsUnknownEnumValue) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir::Builder builder(&context);

  auto invalid_algorithm = static_cast<PrecisionConfig::Algorithm>(9999);
  EXPECT_THAT(stablehlo::ConvertDotAlgorithm(invalid_algorithm, &builder),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unknown dot algorithm enum value #9999")));
}

}  // namespace
}  // namespace xla
