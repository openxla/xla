/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SHAPE_H_
#define XLA_SHAPE_H_

#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/primitive_util.h"
#include "xla/printer.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace xla {

// A quantization attribute describes various properties of a quantized shape
// other than the internal type used for storage, which is expressed using
// Shape::element_type. The properties include the following:
//
// expressed type_: The type that the quantized shape approximates.
// scales_ and zero points_: The quantization parameters of a quantized shape.
// storage_min_ and storage_max_: The minimum and maximum value the internal
//                                storage type can take.
// quantization_dimension_: The axis along which the scales and zero_points may
//                          vary. This is relevant for per-axis quantized shape.
//
// zero_points_, storage_min_ and storage_max_ can be omitted in the quantized
// shape specification.
class QuantizationAttribute {
 public:
  QuantizationAttribute();
  QuantizationAttribute(
      PrimitiveType expressed_type, std::vector<double> scales,
      std::vector<int64_t> zero_points = {},
      std::optional<int64_t> storage_type_min = std::nullopt,
      std::optional<int64_t> storage_type_max = std::nullopt,
      std::optional<int32_t> quantization_dimension = std::nullopt)
      : expressed_type_(expressed_type),
        scales_(std::move(scales)),
        zero_points_(zero_points),
        storage_type_min_(storage_type_min),
        storage_type_max_(storage_type_max),
        quantization_dimension_(quantization_dimension) {
    if (zero_points_.empty()) {
      zero_points_.resize(scales_.size(), 0);
    }
  }
  QuantizationAttribute(const QuantizationAttribute& other);
  QuantizationAttribute(QuantizationAttribute&& other);
  QuantizationAttribute& operator=(const QuantizationAttribute&);
  ~QuantizationAttribute();

  // Construct QuantizedAttribute from a QuantizedAttributesProto.
  static QuantizationAttribute CreateFromProto(
      const QuantizationAttributeProto& proto);

  // Returns a QuantizedAttributesProto representation of the
  // QuantizedAttribute.
  QuantizationAttributeProto ToProto() const;

  // Prints a human-readable string that represents this QuantizedAttribute.
  void Print(Printer* printer) const;

  // Returns a human-readable string that represents this QuantizedAttribute.
  std::string ToString() const;

  // Equal is a configurable functor to check the equality of two quantization
  // attribute.
  class Equal {
   public:
    Equal() = default;

    bool operator()(const QuantizationAttribute& lhs,
                    const QuantizationAttribute& rhs);
  };

  // Methods to access various components of QuantizationAttribute.
  PrimitiveType expressed_type() const { return expressed_type_; }
  void set_expressed_type(PrimitiveType value) { expressed_type_ = value; }

  const std::vector<double>& scales() const { return scales_; }
  int64_t scale_size() const { return scales_.size(); }
  void set_scales(const std::vector<double>& scales) { scales_ = scales; }
  void add_scale(double scale) { scales_.push_back(scale); }

  // bool has_zero_points() const { return !zero_points_.empty(); }
  const std::vector<int64_t>& zero_points() const { return zero_points_; }
  void set_zero_points(const std::vector<int64_t>& zero_points) {
    zero_points_ = zero_points;
  }
  void add_zero_point(int64_t zero_point) {
    zero_points_.push_back(zero_point);
  }

  bool has_storage_type_min() const {
    return storage_type_min_ != std::nullopt;
  }
  int64_t storage_type_min() const {
    CHECK(has_storage_type_min());
    return *storage_type_min_;
  }
  void set_storage_type_min(int64_t storage_type_min) {
    storage_type_min_ = storage_type_min;
  }

  bool has_storage_type_max() const {
    return storage_type_max_ != std::nullopt;
  }
  int64_t storage_type_max() const {
    CHECK(has_storage_type_max());
    return *storage_type_max_;
  }
  void set_storage_type_max(int64_t storage_type_max) {
    storage_type_max_ = storage_type_max;
  }

  bool has_quantization_dimension() const {
    return quantization_dimension_ != std::nullopt;
  }
  int32_t quantization_dimension() const {
    CHECK(has_quantization_dimension());
    return *quantization_dimension_;
  }
  void set_quantization_dimension(int32_t value) {
    quantization_dimension_ = value;
  }

  bool is_per_axis_quantized() const { return has_quantization_dimension(); }
  bool is_per_tensor_quantized() const { return !has_quantization_dimension(); }

 private:
  // Floating point type that the quantized shape approximates.
  PrimitiveType expressed_type_ = PRIMITIVE_TYPE_INVALID;

  // The qunatization parameters for the quantized shape.
  std::vector<double> scales_;
  std::vector<int64_t> zero_points_;

  // The minimum value the internal storage type, Shape::element_type, can take.
  std::optional<int64_t> storage_type_min_;

  // The maximum value the internal storage type, Shape::element_type, can take.
  std::optional<int64_t> storage_type_max_;

  // The axis along which the scales and zero_points of the quantized shape can
  // vary.
  std::optional<int32_t> quantization_dimension_;
};

// A shape describes the number of dimensions in a array, the bounds of each
// dimension, and the primitive component type. For tuples, shape describes the
// structure (number of elements and nesting).
class Shape {
 public:
  Shape();
  ~Shape();
  Shape(const Shape&);
  Shape(Shape&&);
  Shape& operator=(const Shape&);

  // Construct a shape from a ShapeProto.
  explicit Shape(const ShapeProto& shape_proto);

  Shape(PrimitiveType element_type, absl::Span<const int64_t> dimensions,
        absl::Span<const bool> dynamic_dimensions,
        std::vector<Shape> tuple_shapes)
      : element_type_(element_type),
        dimensions_(dimensions.begin(), dimensions.end()),
        dynamic_dimensions_(dynamic_dimensions.begin(),
                            dynamic_dimensions.end()),
        tuple_shapes_(std::move(tuple_shapes)) {}

  // Returns a ShapeProto representation of the Shape.
  ShapeProto ToProto() const;

  // Prints a human-readable string that represents the given shape, with or
  // without layout. e.g. "F32[42,12] {0, 1}" or "F32[64]".
  void Print(Printer* printer, bool print_layout = false) const;

  // Returns a human-readable string that represents the given shape, with or
  // without layout. e.g. "F32[42,12] {0, 1}" or "F32[64]".
  std::string ToString(bool print_layout = false) const;

  // Returns the rank (number of dimensions) of the given shape. Shape must be
  // an array.
  int64_t rank() const {
    DCHECK(IsArray()) << "Non-arrays do not have a rank, shape: " << ToString();
    return dimensions_.size();
  }

  // Returns whether the shape is of the specified type (array, tuple, etc).
  bool IsArray() const { return primitive_util::IsArrayType(element_type()); }
  bool IsTuple() const { return element_type() == TUPLE; }
  bool IsToken() const { return element_type() == TOKEN; }
  bool IsOpaque() const { return element_type() == OPAQUE_TYPE; }
  // bool IsQuantized() const { return has_quantization_attribute(); }

  // Returns whether all elements in the shape are integer.
  // A nested tuple of integers is considered as integer.
  bool IsInteger() const;

  // Returns true if no array dimension in the shape is dynamically sized. Tuple
  // shapes are traversed recursively.
  bool is_static() const;

  bool is_dynamic() const { return !is_static(); }

  // Unbounded dynamism.
  // If `dimensions(axis) == kUnboundedSize && is_dynamic_dimension(axis)`,
  // this means that the axis has unbounded dynamic size.
  // The sentinel value for kUnboundedSize is chosen to be exactly the same
  // as the sentinel value mlir::ShapedType::kDynamic.
  static constexpr int64_t kUnboundedSize = std::numeric_limits<int64_t>::min();

  // Returns true if the shape has one or more dimensions with unbounded sizes.
  // Tuple shapes are traversed recursively.
  bool is_unbounded_dynamic() const;

  // Returns true if the given dimension is unbounded dynamic.
  bool is_unbounded_dynamic_dimension(int dimension) const {
    return dimensions_[dimension] == kUnboundedSize;
  }

  // Sets a given dimension as unbounded dynamic.
  void set_unbounded_dynamic_dimension(int dimension) {
    dynamic_dimensions_[dimension] = true;
    dimensions_[dimension] = kUnboundedSize;
  }

  // Returns true if the given dimension is bounded dynamic.
  bool is_bounded_dynamic_dimension(int dimension) const {
    return is_dynamic_dimension(dimension) &&
           !is_unbounded_dynamic_dimension(dimension);
  }

  // Returns true if the given dimension is dynamically-sized.
  bool is_dynamic_dimension(int dimension) const {
    return dynamic_dimensions_[dimension];
  }

  // Sets whether or not the given dimension is dynamically-sized.
  void set_dynamic_dimension(int dimension, bool is_dynamic) {
    dynamic_dimensions_[dimension] = is_dynamic;
  }

  absl::Span<const bool> dynamic_dimensions() const {
    return dynamic_dimensions_;
  }

  absl::Span<bool> mutable_dynamic_dimensions() {
    return absl::MakeSpan(dynamic_dimensions_);
  }

  // Add dimension_upper_bound().

  // Removes the given dimension from the shape. Layout, if it exists, is
  // adjusted to match the modified shape.
  void DeleteDimension(int64_t dim_to_delete);

  // The following methods mirror the protobuf generated code interface for the
  // message ShapeProto. This enabled easy migration of this data structure
  // from a proto to a proper C++ class.
  // TODO(b/29771030): Replace or augment these methods with a more ergonomic
  // interface.

  // Methods for accessing the primitive type.
  PrimitiveType element_type() const { return element_type_; }
  void set_element_type(PrimitiveType value) { element_type_ = value; }

  // Methods for accessing the dimensions array.
  int dimensions_size() const { return dimensions_.size(); }
  int64_t dimensions(int index) const { return dimensions_[index]; }

  int64_t dimensions_minor(int index) const {
    CHECK(has_layout());
    return dimensions_[layout_->minor_to_major(index)];
  }
  void set_dimensions(int index, int64_t value) { dimensions_[index] = value; }
  void set_dimensions_minor(int index, int64_t value) {
    CHECK(has_layout());
    dimensions_[layout_->minor_to_major(index)] = value;
  }
  void add_dimensions(int64_t value) {
    dimensions_.push_back(value);
    dynamic_dimensions_.push_back(false);
  }
  void clear_dimensions() {
    dimensions_.clear();
    dynamic_dimensions_.clear();
  }
  absl::Span<const int64_t> dimensions() const { return dimensions_; }
  absl::Span<int64_t> mutable_dimensions() {
    return absl::MakeSpan(dimensions_);
  }

  // Methods for accessing the tuple subshapes. This field only non-empty for
  // tuple shapes.
  int tuple_shapes_size() const { return tuple_shapes_.size(); }
  const Shape& tuple_shapes(int index) const;
  Shape* mutable_tuple_shapes(int index) { return &tuple_shapes_[index]; }
  Shape* add_tuple_shapes();
  void clear_tuple_shapes() { tuple_shapes_.clear(); }
  const std::vector<Shape>& tuple_shapes() const { return tuple_shapes_; }
  std::vector<Shape>* mutable_tuple_shapes() { return &tuple_shapes_; }

  // Methods for accessing the layout field.
  bool has_layout() const { return layout_ != std::nullopt; }
  const Layout& layout() const {
    CHECK(has_layout()) << ShortDebugString();
    return *layout_;
  }
  Layout* mutable_layout() {
    CHECK(IsArray()) << ShortDebugString();
    if (layout_ == std::nullopt) {
      layout_.emplace();
    }
    return &(*layout_);
  }
  void clear_layout() { layout_ = std::nullopt; }

  // Returns true if the shape containing quantization attribute .
  // Tuple shapes are traversed recursively.
  bool is_quantized() const;

  // Methods for accessing the quantization attribute.
  bool has_quantization_attribute() const {
    return quantization_attribute_ != std::nullopt;
  }
  const QuantizationAttribute& quantization_attribute() const {
    CHECK(has_quantization_attribute()) << ShortDebugString();
    return *quantization_attribute_;
  }
  QuantizationAttribute* mutable_quantization_attribute() {
    if (quantization_attribute_ == std::nullopt) {
      quantization_attribute_.emplace();
    }
    return &(*quantization_attribute_);
  }
  void clear_quantization_attribute() {
    quantization_attribute_ = std::nullopt;
  }

  // Recursively clear all dynamic dimension of a shape, including bounded and
  // unbounded dynamic dimensions.
  void clear_dynamic_dimensions() {
    if (!IsTuple()) {
      if (is_dynamic()) {
        mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(0);
      }
      for (int64_t i = 0; i < dynamic_dimensions_.size(); ++i) {
        dynamic_dimensions_[i] = false;
      }
      return;
    }
    for (auto& subshape : tuple_shapes_) {
      subshape.clear_dynamic_dimensions();
    }
  }

  void Swap(Shape* other) {
    using std::swap;
    swap(*this, *other);
  }

  void Clear() {
    element_type_ = PRIMITIVE_TYPE_INVALID;
    clear_dimensions();
    tuple_shapes_.clear();
    clear_layout();
  }

  std::string SerializeAsString() const {
    return ToProto().SerializeAsString();
  }
  std::string ShortDebugString() const { return ToProto().ShortDebugString(); }
  std::string DebugString() const { return ToProto().DebugString(); }

  // Equal is a configurable functor to check the equality of two shapes.
  //
  // Examples:
  //
  // - Comparing two shapes ignoring their layout difference:
  //   Equal().IgnoreLayout()(shape1, shape2);
  //
  // - Comparing two shapes ignoring their layout and element type difference:
  //   Equal().IgnoreLayout().IgnoreElementType()(shape1, shape2);
  class Equal {
   public:
    Equal() = default;

    bool operator()(const Shape& lhs, const Shape& rhs);

    Equal& IgnoreLayout() {
      ignore_layout_ = true;
      return *this;
    }
    Equal& IgnoreTilesInLayout() {
      ignore_tiles_in_layout_ = true;
      return *this;
    }
    Equal& IgnoreElementSizeInLayout() {
      ignore_element_size_in_layout_ = true;
      return *this;
    }
    Equal& IgnoreMemorySpaceInLayout() {
      ignore_memory_space_in_layout_ = true;
      return *this;
    }
    Equal& MinorToMajorOnlyInLayout() {
      ignore_tiles_in_layout_ = true;
      ignore_element_size_in_layout_ = true;
      ignore_memory_space_in_layout_ = true;
      return *this;
    }
    Equal& IgnoreElementType() {
      ignore_element_type_ = true;
      return *this;
    }
    Equal& IgnoreFpPrecision() {
      ignore_fp_precision_ = true;
      return *this;
    }
    Equal& IgnoreDynamicDimension() {
      ignore_dynamic_dimension_ = true;
      return *this;
    }
    Equal& IgnoreDimensions() {
      ignore_dimensions_ = true;
      return *this;
    }
    Equal& IgnoreQuantizationAttribute() {
      ignore_quantization_attribute_ = true;
      return *this;
    }

   private:
    bool ignore_layout_ = false;
    bool ignore_tiles_in_layout_ = false;
    bool ignore_element_size_in_layout_ = false;
    bool ignore_memory_space_in_layout_ = false;
    bool ignore_element_type_ = false;
    bool ignore_fp_precision_ = false;
    bool ignore_dynamic_dimension_ = false;
    bool ignore_dimensions_ = false;
    bool ignore_quantization_attribute_ = false;
  };

  // Test that all fields of the shape are the same, equivalent to Equal().
  bool operator==(const Shape& other) const { return Equal()(*this, other); }
  bool operator!=(const Shape& other) const { return !(*this == other); }

  template <typename H, bool kIsLayoutSensitive = true>
  static H Hash(H h, const Shape& s) {
    if (s.IsTuple()) {
      for (const Shape& subshape : s.tuple_shapes_) {
        h = Shape::Hash<H, kIsLayoutSensitive>(std::move(h), subshape);
      }
      return H::combine(std::move(h), s.tuple_shapes_size());
    }
    h = H::combine(std::move(h), s.element_type_, s.dimensions_,
                   s.dynamic_dimensions_);
    if (kIsLayoutSensitive) {
      h = H::combine(std::move(h), s.layout_);
    }
    return std::move(h);
  }

  template <typename H>
  friend H AbslHashValue(H h, const Shape& s) {
    return Shape::Hash(std::move(h), s);
  }

 private:
  // The element type of this shape (tuple, array, etc).
  PrimitiveType element_type_ = PRIMITIVE_TYPE_INVALID;

  // The array bounds of the dimensions. This is nonempty only for array
  // shapes. For a dynamically-sized dimension, the respective value in this
  // vector is an inclusive upper limit of the array bound.
  DimensionVector dimensions_;

  // This vector is the same size as 'dimensions_' and indicates whether the
  // respective dimension is dynamically sized.
  absl::InlinedVector<bool, InlineRank()> dynamic_dimensions_;

  // The tuple element subshapes. This is nonempty only for tuple shapes.
  std::vector<Shape> tuple_shapes_;

  // The layout of the shape. Only relevant for arrays.
  std::optional<Layout> layout_;

  std::optional<QuantizationAttribute> quantization_attribute_;
};

// Shape of the parameters and output of an XLA computation. This is analogous
// to a traditional function signature.
class ProgramShape {
 public:
  ProgramShape();
  ~ProgramShape();
  ProgramShape(const ProgramShape&);
  ProgramShape(ProgramShape&&);
  ProgramShape& operator=(const ProgramShape&);

  // Creates a ProgramShape from a ProgramShapeProto protobuf.
  explicit ProgramShape(const ProgramShapeProto& program_shape_proto);

  // Returns a proto representation of the object.
  ProgramShapeProto ToProto() const;

  void Print(Printer* printer) const;

  std::string ToString() const;

  // The following methods mirror the protobuf generated code interface for the
  // message ProgramShapeProto. This enabled easy migration of this data
  // structure from a proto to a proper C++ class.
  // TODO(b/29771030): Replace or augment these methods with a more ergonomic
  // interface.

  // Methods for accessing and manipulating the Shape of the parameters.
  int parameters_size() const { return parameters_.size(); }
  const Shape& parameters(int index) const { return parameters_[index]; }
  Shape* mutable_parameters(int index) { return &parameters_[index]; }
  Shape* add_parameters() {
    parameters_.emplace_back();
    return &parameters_.back();
  }
  void clear_parameters() { parameters_.clear(); }
  const std::vector<Shape>& parameters() const { return parameters_; }
  std::vector<Shape>* mutable_parameters() { return &parameters_; }

  // Methods for accessing and manipulating the Shape of the result.
  const Shape& result() const { return result_; }
  Shape* mutable_result() { return &result_; }

  // Methods for accessing and manipulating the names of the parameters.
  int parameter_names_size() const { return parameter_names_.size(); }
  const std::string& parameter_names(int index) const {
    return parameter_names_[index];
  }
  void set_parameter_names(int index, const std::string& value) {
    parameter_names_[index] = value;
  }
  std::string* mutable_parameter_names(int index) {
    return &parameter_names_[index];
  }
  void add_parameter_names(const std::string& value) {
    parameter_names_.push_back(value);
  }
  std::string* add_parameter_names() {
    parameter_names_.push_back("");
    return &parameter_names_.back();
  }
  void clear_parameter_names() { parameter_names_.clear(); }
  const std::vector<std::string>& parameter_names() const {
    return parameter_names_;
  }
  std::vector<std::string>* mutable_parameter_names() {
    return &parameter_names_;
  }

  std::string ShortDebugString() const { return ToProto().ShortDebugString(); }
  std::string DebugString() const { return ToProto().DebugString(); }

 private:
  // The shapes of the parameters of the computation represented by this object.
  std::vector<Shape> parameters_;

  // The names of the parameters of the computation represented by this object.
  std::vector<std::string> parameter_names_;

  // The shape of the result of the computation represented by this object.
  Shape result_;
};

std::ostream& operator<<(std::ostream& out, const Shape& shape);
std::ostream& operator<<(std::ostream& out, const ProgramShape& program_shape);

}  // namespace xla

#endif  // XLA_SHAPE_H_
