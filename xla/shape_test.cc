/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/shape.h"

#include <limits>
#include <optional>

#include "absl/hash/hash_testing.h"
#include "xla/layout.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/test_benchmark.h"

namespace xla {
namespace {

class ShapeTest : public ::testing::Test {
 protected:
  const Shape opaque_ = ShapeUtil::MakeOpaqueShape();
  const Shape token_ = ShapeUtil::MakeTokenShape();
  const Shape scalar_ = ShapeUtil::MakeShape(F32, {});
  const Shape scalar_with_tile_ =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {}, {}, {Tile({256})});
  const Shape matrix_ = ShapeUtil::MakeShape(U32, {1, 2});
  const Shape matrix2_ =
      ShapeUtil::MakeShapeWithDenseLayout(S32, {3, 4}, {0, 1});
  const Shape tuple_ =
      ShapeUtil::MakeTupleShape({opaque_, scalar_, matrix_, matrix2_});
  const Shape nested_tuple_ =
      ShapeUtil::MakeTupleShape({tuple_, matrix_, token_});
  const Shape dynamic_matrix_ =
      ShapeUtil::MakeShape(S32, {5, 2}, {true, false});
  const Shape unbounded_ =
      ShapeUtil::MakeShape(F32, {Shape::kUnboundedSize, 784}, {true, false});
  const Shape quantized_per_tensor_ = ShapeUtil::MakeShape(
      S8, {2, 1},
      QuantizationAttribute(/*expressed_typ*/ F32, /*scales*/ {2.0},
                            /*zero_points*/ {5}, /*storage_min*/ -128,
                            /*storage_max*/ 127));
  const Shape quantized_per_axis_ = ShapeUtil::MakeShape(
      S8, {2, 1},
      QuantizationAttribute(/*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
                            /*zero_points*/ {5, 6}, /*storage_min*/ -128,
                            /*storage_max*/ 127, /*quantization_dimension*/ 0));
  const Shape quantized_shape_with_missing_storage_min_max_types_ =
      ShapeUtil::MakeShape(
          S8, {2, 1},
          QuantizationAttribute(/*expressed_typ*/ F32, /*scales*/ {2.0},
                                /*zero_points*/ {5},
                                /*storage_min*/ std::nullopt,
                                /*storage_max*/ std::nullopt));
};

TEST_F(ShapeTest, ShapeToFromProto) {
  for (const Shape& shape : {opaque_, token_, scalar_, matrix_, matrix2_,
                             tuple_, nested_tuple_, dynamic_matrix_, unbounded_,
                             quantized_per_tensor_, quantized_per_axis_}) {
    Shape shape_copy(shape.ToProto());
    EXPECT_TRUE(ShapeUtil::Equal(shape, shape_copy))
        << shape << " != " << shape_copy;
  }
}

TEST_F(ShapeTest, ShapeToString) {
  EXPECT_EQ("opaque[]", opaque_.ToString());
  EXPECT_EQ("token[]", token_.ToString());
  EXPECT_EQ("f32[]", scalar_.ToString());
  EXPECT_EQ("u32[1,2]", matrix_.ToString());
  EXPECT_EQ("s32[3,4]", matrix2_.ToString());
  EXPECT_EQ("(opaque[], f32[], u32[1,2], s32[3,4])", tuple_.ToString());
  EXPECT_EQ("((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], token[])",
            nested_tuple_.ToString());

  EXPECT_EQ("opaque[]", opaque_.ToString(/*print_layout=*/true));
  EXPECT_EQ("f32[]", scalar_.ToString(/*print_layout=*/true));
  EXPECT_EQ("f32[]{:T(256)}",
            scalar_with_tile_.ToString(/*print_layout=*/true));
  EXPECT_EQ("u32[1,2]{1,0}", matrix_.ToString(/*print_layout=*/true));
  EXPECT_EQ("s32[3,4]{0,1}", matrix2_.ToString(/*print_layout=*/true));
  EXPECT_EQ("(opaque[], f32[], u32[1,2]{1,0}, s32[3,4]{0,1})",
            tuple_.ToString(/*print_layout=*/true));
  EXPECT_EQ(
      "((opaque[], f32[], u32[1,2]{1,0}, s32[3,4]{0,1}), u32[1,2]{1,0}, "
      "token[])",
      nested_tuple_.ToString(/*print_layout=*/true));
  EXPECT_EQ("qint<s8<-128:127>:f32,2:5>[2,1]{1,0}",
            quantized_per_tensor_.ToString(/*print_layout=*/true));
  EXPECT_EQ("qint<s8<-128:127>:f32:0,{2:5,1:6}>[2,1]{1,0}",
            quantized_per_axis_.ToString(/*print_layout=*/true));
  EXPECT_EQ("qint<s8<-128:127>:f32,2:5>[2,1]{1,0}",
            quantized_shape_with_missing_storage_min_max_types_.ToString(
                /*print_layout=*/true));
}

TEST_F(ShapeTest, DynamicShapeToString) {
  Shape array_shape =
      ShapeUtil::MakeShape(F32, {23, 44, 55}, {true, false, true});
  EXPECT_EQ("f32[<=23,44,<=55]", array_shape.ToString());

  array_shape.set_dynamic_dimension(2, false);
  EXPECT_EQ("f32[<=23,44,55]", array_shape.ToString());

  EXPECT_EQ("f32[?,784]", unbounded_.ToString());
}

TEST_F(ShapeTest, EqualityTest) {
  // Different layouts.
  EXPECT_NE(ShapeUtil::MakeShapeWithDenseLayout(F32, {23, 44}, {1, 0}),
            ShapeUtil::MakeShapeWithDenseLayout(F32, {23, 44}, {0, 1}));

  // Different dims.
  EXPECT_NE(ShapeUtil::MakeShapeWithDenseLayout(F32, {44, 23}, {1, 0}),
            ShapeUtil::MakeShapeWithDenseLayout(F32, {23, 44}, {1, 0}));

  // Different elements.
  EXPECT_NE(ShapeUtil::MakeShapeWithDenseLayout(S32, {44, 23}, {1, 0}),
            ShapeUtil::MakeShapeWithDenseLayout(F32, {23, 44}, {1, 0}));

  // Equal shapes.
  EXPECT_EQ(ShapeUtil::MakeShapeWithDenseLayout(F32, {23, 44}, {1, 0}),
            ShapeUtil::MakeShapeWithDenseLayout(F32, {23, 44}, {1, 0}));

  // Equality of two quantized shapes.
  EXPECT_EQ(ShapeUtil::MakeShape(
                S8, {2, 1},
                QuantizationAttribute(
                    /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
                    /*zero_points*/ {5, 6}, /*storage_min*/ -128,
                    /*storage_max*/ 127, /*quantization_dimension*/ 0)),
            ShapeUtil::MakeShape(
                S8, {2, 1},
                QuantizationAttribute(
                    /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
                    /*zero_points*/ {5, 6}, /*storage_min*/ -128,
                    /*storage_max*/ 127, /*quantization_dimension*/ 0)));

  // Non-equality of quantized and non-quantized shapes.
  EXPECT_NE(ShapeUtil::MakeShape(
                S8, {2, 1},
                QuantizationAttribute(
                    /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
                    /*zero_points*/ {5, 6}, /*storage_min*/ -128,
                    /*storage_max*/ 127, /*quantization_dimension*/ 0)),
            ShapeUtil::MakeShape(S8, {2, 1}));

  // Non-equality of two quantized shapes with different storage types.
  EXPECT_NE(ShapeUtil::MakeShape(
                S8, {2, 1},
                QuantizationAttribute(
                    /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
                    /*zero_points*/ {5, 6}, /*storage_min*/ -128,
                    /*storage_max*/ 127, /*quantization_dimension*/ 0)),
            ShapeUtil::MakeShape(
                S32, {2, 1},
                QuantizationAttribute(
                    /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
                    /*zero_points*/ {5, 6}, /*storage_min*/ -128,
                    /*storage_max*/ 127, /*quantization_dimension*/ 0)));

  // Non-equality of two quantized shapes with different expressed types.
  EXPECT_NE(ShapeUtil::MakeShape(
                S8, {2, 1},
                QuantizationAttribute(
                    /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
                    /*zero_points*/ {5, 6}, /*storage_min*/ -128,
                    /*storage_max*/ 127, /*quantization_dimension*/ 0)),
            ShapeUtil::MakeShape(
                S8, {2, 1},
                QuantizationAttribute(
                    /*expressed_typ*/ F16, /*scales*/ {2.0, 1.0},
                    /*zero_points*/ {5, 6}, /*storage_min*/ -128,
                    /*storage_max*/ 127, /*quantization_dimension*/ 0)));

  // Non-equality of two quantized shapes with different scales.
  EXPECT_NE(ShapeUtil::MakeShape(
                S8, {2, 1},
                QuantizationAttribute(
                    /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
                    /*zero_points*/ {5, 6}, /*storage_min*/ -128,
                    /*storage_max*/ 127, /*quantization_dimension*/ 0)),
            ShapeUtil::MakeShape(
                S8, {2, 1},
                QuantizationAttribute(
                    /*expressed_typ*/ F32, /*scales*/ {2.0, 1.5},
                    /*zero_points*/ {5, 6}, /*storage_min*/ -128,
                    /*storage_max*/ 127, /*quantization_dimension*/ 0)));

  // Non-equality of two quantized shapes with different zero-points.
  EXPECT_NE(ShapeUtil::MakeShape(
                S8, {2, 1},
                QuantizationAttribute(
                    /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
                    /*zero_points*/ {5, 6}, /*storage_min*/ -128,
                    /*storage_max*/ 127, /*quantization_dimension*/ 0)),
            ShapeUtil::MakeShape(
                S8, {2, 1},
                QuantizationAttribute(
                    /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
                    /*zero_points*/ {5, 7}, /*storage_min*/ -128,
                    /*storage_max*/ 127, /*quantization_dimension*/ 0)));

  // Non-equality of two quantized shapes with different storage min/max.
  EXPECT_NE(ShapeUtil::MakeShape(
                S8, {2, 1},
                QuantizationAttribute(
                    /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
                    /*zero_points*/ {5, 6}, /*storage_min*/ -128,
                    /*storage_max*/ 127, /*quantization_dimension*/ 0)),
            ShapeUtil::MakeShape(
                S8, {2, 1},
                QuantizationAttribute(
                    /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
                    /*zero_points*/ {5, 6}, /*storage_min*/ -127,
                    /*storage_max*/ 127, /*quantization_dimension*/ 0)));

  // Non-equality of two quantized shapes with different quantization dimension.
  EXPECT_NE(ShapeUtil::MakeShape(
                S8, {2, 1},
                QuantizationAttribute(
                    /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
                    /*zero_points*/ {5, 6}, /*storage_min*/ -128,
                    /*storage_max*/ 127, /*quantization_dimension*/ 0)),
            ShapeUtil::MakeShape(
                S8, {2, 1},
                QuantizationAttribute(
                    /*expressed_typ*/ F32, /*scales*/ {2.0},
                    /*zero_points*/ {5}, /*storage_min*/ -128,
                    /*storage_max*/ 127, /*quantization_dimension*/ 1)));

  // Non-equality of per-tensor vs per-axis quantized shapes.
  EXPECT_NE(ShapeUtil::MakeShape(S8, {2, 1},
                                 QuantizationAttribute(
                                     /*expressed_typ*/ F32, /*scales*/ {2.0},
                                     /*zero_points*/ {5}, /*storage_min*/ -128,
                                     /*storage_max*/ 127)),
            ShapeUtil::MakeShape(
                S8, {2, 1},
                QuantizationAttribute(
                    /*expressed_typ*/ F32, /*scales*/ {2.0},
                    /*zero_points*/ {5}, /*storage_min*/ -128,
                    /*storage_max*/ 127, /*quantization_dimension*/ 1)));
}

TEST_F(ShapeTest, IsStatic) {
  EXPECT_TRUE(opaque_.is_static());
  EXPECT_TRUE(token_.is_static());
  EXPECT_TRUE(matrix_.is_static());
  EXPECT_TRUE(tuple_.is_static());
  EXPECT_TRUE(nested_tuple_.is_static());

  Shape dynamic_matrix = matrix_;
  EXPECT_TRUE(dynamic_matrix.is_static());
  dynamic_matrix.set_dynamic_dimension(1, true);
  EXPECT_FALSE(dynamic_matrix.is_static());

  Shape dynamic_tuple = tuple_;
  EXPECT_TRUE(dynamic_tuple.is_static());
  ShapeUtil::GetMutableSubshape(&dynamic_tuple, {2})
      ->set_dynamic_dimension(1, true);
  EXPECT_FALSE(dynamic_tuple.is_static());

  EXPECT_FALSE(unbounded_.is_static());
  EXPECT_TRUE(quantized_per_tensor_.is_static());
  EXPECT_TRUE(quantized_per_axis_.is_static());
}

TEST_F(ShapeTest, IsDynamic) {
  EXPECT_FALSE(matrix_.is_dynamic());
  EXPECT_FALSE(matrix_.is_unbounded_dynamic());

  EXPECT_TRUE(dynamic_matrix_.is_dynamic());
  EXPECT_FALSE(dynamic_matrix_.is_unbounded_dynamic());

  EXPECT_TRUE(unbounded_.is_dynamic());
  EXPECT_TRUE(unbounded_.is_unbounded_dynamic());

  Shape unbounded_tuple = tuple_;
  EXPECT_FALSE(unbounded_tuple.is_unbounded_dynamic());
  ShapeUtil::GetMutableSubshape(&unbounded_tuple, {2})
      ->set_dynamic_dimension(1, true);
  EXPECT_FALSE(unbounded_tuple.is_unbounded_dynamic());
  ShapeUtil::GetMutableSubshape(&unbounded_tuple, {2})
      ->set_dimensions(1, Shape::kUnboundedSize);
  EXPECT_TRUE(unbounded_tuple.is_unbounded_dynamic());
}

TEST_F(ShapeTest, QuantizedType) {
  const Shape shape_1_ = ShapeUtil::MakeShape(F32, {2});
  const Shape shape_2_ = ShapeUtil::MakeShape(S8, {2});
  const Shape shape_tuple_ = ShapeUtil::MakeTupleShape({shape_1_, shape_2_});
  Shape quantized_tuple = shape_tuple_;

  Shape* mutable_quantized_tuple_1 =
      ShapeUtil::GetMutableSubshape(&quantized_tuple, {0});
  auto status_1 = ShapeUtil::PopulateShapeWithQuantizationAttribute(
      QuantizationAttribute(
          /*expressed_typ*/ F32, /*scales*/ {2.0},
          /*zero_points*/ {5}, /*storage_min*/ std::nullopt,
          /*storage_max*/ std::nullopt),
      mutable_quantized_tuple_1);
  EXPECT_FALSE(status_1.ok());
  EXPECT_THAT(status_1.message(),
              testing::HasSubstr("invalid element type for quantized shape"));

  Shape* mutable_quantized_tuple_2 =
      ShapeUtil::GetMutableSubshape(&quantized_tuple, {1});
  auto status_2 = ShapeUtil::PopulateShapeWithQuantizationAttribute(
      QuantizationAttribute(
          /*expressed_typ*/ U32, /*scales*/ {2.0},
          /*zero_points*/ {5}, /*storage_min*/ std::nullopt,
          /*storage_max*/ std::nullopt),
      mutable_quantized_tuple_2);
  EXPECT_FALSE(status_2.ok());
  EXPECT_THAT(status_2.message(),
              testing::HasSubstr("invalid expressed type for quantized shape"));

  Shape* mutable_quantized_tuple_3 =
      ShapeUtil::GetMutableSubshape(&quantized_tuple, {1});
  auto status_3 = ShapeUtil::PopulateShapeWithQuantizationAttribute(
      QuantizationAttribute(
          /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
          /*zero_points*/ {5}, /*storage_min*/ std::nullopt,
          /*storage_max*/ std::nullopt, /*quantization_dimension*/ 0),
      mutable_quantized_tuple_3);
  EXPECT_FALSE(status_3.ok());
  EXPECT_THAT(status_3.message(),
              testing::HasSubstr("illegal number of scales (2) and zero_points "
                                 "(1) for quantized shape"));

  Shape* mutable_quantized_tuple_4 =
      ShapeUtil::GetMutableSubshape(&quantized_tuple, {1});
  auto status_4 = ShapeUtil::PopulateShapeWithQuantizationAttribute(
      QuantizationAttribute(
          /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
          /*zero_points*/ {5, 0}, /*storage_min*/ std::nullopt,
          /*storage_max*/ std::nullopt),
      mutable_quantized_tuple_4);
  EXPECT_FALSE(status_4.ok());
  EXPECT_THAT(
      status_4.message(),
      testing::HasSubstr(
          "illegal number of scales (2), expected 1, for quantized shape"));

  Shape* mutable_quantized_tuple_5 =
      ShapeUtil::GetMutableSubshape(&quantized_tuple, {1});
  auto status_5 = ShapeUtil::PopulateShapeWithQuantizationAttribute(
      QuantizationAttribute(
          /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
          /*zero_points*/ {5, 0}, /*storage_min*/ std::nullopt,
          /*storage_max*/ std::nullopt, 1),
      mutable_quantized_tuple_5);
  EXPECT_FALSE(status_5.ok());
  EXPECT_THAT(status_5.message(),
              testing::HasSubstr(
                  "illegal quantization dimension (1) for quantized shape"));

  Shape* mutable_quantized_tuple_6 =
      ShapeUtil::GetMutableSubshape(&quantized_tuple, {1});
  auto status_6 = ShapeUtil::PopulateShapeWithQuantizationAttribute(
      QuantizationAttribute(
          /*expressed_typ*/ F32, /*scales*/ {2.0},
          /*zero_points*/ {5}, /*storage_min*/ std::nullopt,
          /*storage_max*/ std::nullopt, -1),
      mutable_quantized_tuple_6);
  EXPECT_FALSE(status_6.ok());
  EXPECT_THAT(status_6.message(),
              testing::HasSubstr(
                  "illegal quantization dimension (-1) for quantized shape"));

  Shape* mutable_quantized_tuple_7 =
      ShapeUtil::GetMutableSubshape(&quantized_tuple, {1});
  auto status_7 = ShapeUtil::PopulateShapeWithQuantizationAttribute(
      QuantizationAttribute(
          /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0, 1.0},
          /*zero_points*/ {5, 0, 0}, /*storage_min*/ std::nullopt,
          /*storage_max*/ std::nullopt, 0),
      mutable_quantized_tuple_7);
  EXPECT_FALSE(status_7.ok());
  EXPECT_THAT(
      status_7.message(),
      testing::HasSubstr(
          "illegal number of scales (3), expected 2, for quantized shape"));

  Shape* mutable_quantized_tuple_8 =
      ShapeUtil::GetMutableSubshape(&quantized_tuple, {1});
  auto status_8 = ShapeUtil::PopulateShapeWithQuantizationAttribute(
      QuantizationAttribute(
          /*expressed_typ*/ F32, /*scales*/ {-2.0, 1.0},
          /*zero_points*/ {5, 0}, /*storage_min*/ std::nullopt,
          /*storage_max*/ std::nullopt, 0),
      mutable_quantized_tuple_8);
  EXPECT_FALSE(status_8.ok());
  EXPECT_THAT(status_8.message(),
              testing::HasSubstr("illegal scale value (-2"));

  Shape* mutable_quantized_tuple_9 =
      ShapeUtil::GetMutableSubshape(&quantized_tuple, {1});
  auto status_9 = ShapeUtil::PopulateShapeWithQuantizationAttribute(
      QuantizationAttribute(
          /*expressed_typ*/ F32,
          /*scales*/ {std::numeric_limits<float>::infinity(), 1.0},
          /*zero_points*/ {5, 0}, /*storage_min*/ std::nullopt,
          /*storage_max*/ std::nullopt, 0),
      mutable_quantized_tuple_9);
  EXPECT_FALSE(status_9.ok());
  EXPECT_THAT(
      status_9.message(),
      testing::HasSubstr("illegal scale value (inf) for quantized shape"));

  Shape* mutable_quantized_tuple_10 =
      ShapeUtil::GetMutableSubshape(&quantized_tuple, {1});
  auto status_10 = ShapeUtil::PopulateShapeWithQuantizationAttribute(
      QuantizationAttribute(
          /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
          /*zero_points*/ {5, 0}, /*storage_min*/ -129,
          /*storage_max*/ std::nullopt, 0),
      mutable_quantized_tuple_10);
  EXPECT_FALSE(status_10.ok());
  EXPECT_THAT(
      status_10.message(),
      testing::HasSubstr("value of storage_type_min (-129) does not fit into "
                         "the storage_type (s8) for quantized shape"));

  Shape* mutable_quantized_tuple_11 =
      ShapeUtil::GetMutableSubshape(&quantized_tuple, {1});
  auto status_11 = ShapeUtil::PopulateShapeWithQuantizationAttribute(
      QuantizationAttribute(
          /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
          /*zero_points*/ {-129, 0}, /*storage_min*/ -128,
          /*storage_max*/ std::nullopt, 0),
      mutable_quantized_tuple_11);
  EXPECT_FALSE(status_11.ok());
  EXPECT_THAT(
      status_11.message(),
      testing::HasSubstr("illegal value of zero point (-129) less than "
                         "storage_type_min (-128) for quantized shape"));

  Shape* mutable_quantized_tuple_12 =
      ShapeUtil::GetMutableSubshape(&quantized_tuple, {1});
  auto status_12 = ShapeUtil::PopulateShapeWithQuantizationAttribute(
      QuantizationAttribute(
          /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
          /*zero_points*/ {5, 0}, /*storage_min*/ std::nullopt,
          /*storage_max*/ 128, 0),
      mutable_quantized_tuple_12);
  EXPECT_FALSE(status_12.ok());
  EXPECT_THAT(
      status_12.message(),
      testing::HasSubstr("value of storage_type_max (128) does not fit into "
                         "the storage_type (s8) for quantized shape"));

  Shape* mutable_quantized_tuple_13 =
      ShapeUtil::GetMutableSubshape(&quantized_tuple, {1});
  auto status_13 = ShapeUtil::PopulateShapeWithQuantizationAttribute(
      QuantizationAttribute(
          /*expressed_typ*/ F32, /*scales*/ {2.0, 1.0},
          /*zero_points*/ {128, 0}, /*storage_min*/ std::nullopt,
          /*storage_max*/ 127, 0),
      mutable_quantized_tuple_13);
  EXPECT_FALSE(status_13.ok());
  EXPECT_THAT(
      status_13.message(),
      testing::HasSubstr("illegal value of zero point (128) greater than "
                         "storage_type_max (127) for quantized shape"));
}

TEST_F(ShapeTest, IsQuantized) {
  Shape quantized_tuple = tuple_;
  EXPECT_FALSE(quantized_tuple.is_quantized());
  Shape* mutable_quantized_tuple =
      ShapeUtil::GetMutableSubshape(&quantized_tuple, {2});
  auto status = ShapeUtil::PopulateShapeWithQuantizationAttribute(
      QuantizationAttribute(
          /*expressed_typ*/ F32, /*scales*/ {2.0},
          /*zero_points*/ {5}, /*storage_min*/ 0,
          /*storage_max*/ 127),
      mutable_quantized_tuple);
  EXPECT_TRUE(status.ok() && quantized_tuple.is_quantized());
}

TEST_F(ShapeTest, IsDynamicDimension) {
  Shape dynamic_matrix = matrix_;
  dynamic_matrix.set_dynamic_dimension(1, true);
  EXPECT_FALSE(dynamic_matrix.is_dynamic_dimension(0));
  EXPECT_TRUE(dynamic_matrix.is_dynamic_dimension(1));

  Shape dynamic_tuple = tuple_;
  EXPECT_TRUE(dynamic_tuple.is_static());
  ShapeUtil::GetMutableSubshape(&dynamic_tuple, {2})
      ->set_dynamic_dimension(1, true);
  EXPECT_FALSE(dynamic_tuple.is_static());

  EXPECT_TRUE(unbounded_.is_dynamic_dimension(0));
  EXPECT_FALSE(unbounded_.is_dynamic_dimension(1));
}

TEST_F(ShapeTest, ProgramShapeToFromProto) {
  ProgramShape program_shape;
  *program_shape.add_parameters() = ShapeUtil::MakeShape(F32, {1, 2, 3});
  *program_shape.add_parameters() = ShapeUtil::MakeTokenShape();
  *program_shape.add_parameters() = ShapeUtil::MakeShape(S64, {});
  *program_shape.add_parameters() = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(S32, {}),
       ShapeUtil::MakeTupleShape({ShapeUtil::MakeTokenShape()}),
       ShapeUtil::MakeShape(F32, {42, 42})});

  *program_shape.mutable_result() = ShapeUtil::MakeShape(F32, {7});

  program_shape.add_parameter_names("foo");
  program_shape.add_parameter_names("bar");
  program_shape.add_parameter_names("baz");
  program_shape.add_parameter_names("qux qux");

  // Create a copy of the program shape by round-tripping through a proto.
  ProgramShape program_shape_copy(program_shape.ToProto());
  ASSERT_EQ(program_shape.parameters_size(),
            program_shape_copy.parameters_size());
  for (int i = 0; i < program_shape.parameters_size(); ++i) {
    EXPECT_TRUE(ShapeUtil::Equal(program_shape.parameters(i),
                                 program_shape_copy.parameters(i)));
  }

  EXPECT_TRUE(
      ShapeUtil::Equal(program_shape.result(), program_shape_copy.result()));

  ASSERT_EQ(program_shape.parameter_names_size(),
            program_shape_copy.parameter_names_size());
  for (int i = 0; i < program_shape.parameter_names_size(); ++i) {
    EXPECT_EQ(program_shape.parameter_names(i),
              program_shape_copy.parameter_names(i));
  }
}

TEST_F(ShapeTest, ProgramShapeToString) {
  ProgramShape prog = ShapeUtil::MakeProgramShape(
      {opaque_, scalar_, matrix_, matrix2_, tuple_, nested_tuple_},
      nested_tuple_);
  EXPECT_EQ(
      "((unknown): opaque[], "
      "(unknown): f32[], "
      "(unknown): u32[1,2], "
      "(unknown): s32[3,4], "
      "(unknown): (opaque[], f32[], u32[1,2], s32[3,4]), "
      "(unknown): ((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], token[])) "
      "-> "
      "((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], token[])",
      prog.ToString());

  prog.add_parameter_names("arg0");
  prog.add_parameter_names("scalar");
  prog.add_parameter_names("matrix");
  prog.add_parameter_names("matrix2");
  prog.add_parameter_names("tuple");
  prog.add_parameter_names("nested_tuple");
  EXPECT_EQ(
      "(arg0: opaque[], "
      "scalar: f32[], "
      "matrix: u32[1,2], "
      "matrix2: s32[3,4], "
      "tuple: (opaque[], f32[], u32[1,2], s32[3,4]), "
      "nested_tuple: ((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], "
      "token[])) "
      "-> "
      "((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], token[])",
      prog.ToString());
}

TEST_F(ShapeTest, SupportsAbslHash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      {opaque_, token_, scalar_, scalar_with_tile_, matrix_, matrix2_, tuple_,
       nested_tuple_, dynamic_matrix_}));
}

void BM_ShapeCopy(::testing::benchmark::State& state) {
  // Create different shapes based on benchmark parameters:
  Shape shape;
  switch (state.range(0)) {
    case 0: {
      // Shape()
      break;
    }
    case 1: {
      // f32[1,2,2]{2,1,0}
      shape = Shape(F32, {1, 2, 2}, {false, false, false}, {});
      *shape.mutable_layout() = Layout({2, 1, 0});
      break;
    }
    case 2: {
      // f32[1,2,2]{2,1,0:T(2,128)}
      shape = Shape(F32, {1, 2, 2}, {false, false, false}, {});
      *shape.mutable_layout() = Layout({2, 1, 0}, {}, {}, {}, {Tile({2, 128})});
      break;
    }
  }
  state.SetLabel(shape.ToString(true));

  for (auto s : state) {
    Shape copy(shape);
  }
}
BENCHMARK(BM_ShapeCopy)->Arg(0)->Arg(1)->Arg(2);

}  // namespace
}  // namespace xla
