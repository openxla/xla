/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/nanort/nanort_executable.h"

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/base/dynamic_annotations.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tsl/platform/platform.h"
#include "xla/backends/cpu/buffer_allocation_info.h"
#include "xla/backends/cpu/buffer_allocation_info_util.h"
#include "xla/backends/cpu/constant_allocation.h"
#include "xla/backends/cpu/runtime/dot_lib.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/kernel.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/backends/cpu/runtime/rng_state_lib.h"
#include "xla/backends/cpu/runtime/sort_lib.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/backends/cpu/runtime/topk_lib.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/service/cpu/cpu_aot_loader.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/shaped_slice.pb.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

std::atomic<NanoRtExecutable::AotCompilationResultImporter>& GetAotImporter() {
  static std::atomic<NanoRtExecutable::AotCompilationResultImporter> importer{
      nullptr};
  return importer;
}

struct LiteSlice {
  int64_t index = -1;
  int64_t offset = 0;
  int64_t size = 0;
};

LiteSlice ParseSlice(
    const xla::buffer_assignment::BufferAllocationSliceProto& proto) {
  return LiteSlice{proto.buffer_allocation_index(), proto.offset(),
                   proto.size()};
}

absl::StatusOr<se::DeviceAddressBase> ResolveSlice(
    absl::Span<const se::DeviceAddressBase> buffers, const LiteSlice& slice) {
  if (ABSL_PREDICT_FALSE(slice.index < 0 ||
                         static_cast<size_t>(slice.index) >= buffers.size())) {
    return InvalidArgument("Invalid buffer allocation index %d", slice.index);
  }
  const se::DeviceAddressBase& base = buffers[slice.index];
  if (ABSL_PREDICT_FALSE(slice.offset < 0 || slice.size < 0 ||
                         static_cast<uint64_t>(slice.offset + slice.size) >
                             base.size())) {
    return InvalidArgument(
        "Slice offset %d size %d out of bounds for buffer %d of size %d",
        slice.offset, slice.size, slice.index, base.size());
  }
  return base.GetByteSlice(slice.offset, slice.size);
}

internal::SortDims ComputeSortDims(const Shape& shape, int64_t dimension) {
  int64_t sort_dimension =
      dimension >= 0 ? dimension : shape.dimensions().size() + dimension;
  Shape physical_shape =
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(shape);
  auto logical_to_physical = LayoutUtil::MakeLogicalToPhysical(shape.layout());
  sort_dimension = logical_to_physical[sort_dimension];

  auto product = [](absl::Span<const int64_t> dims) {
    return absl::c_accumulate(dims, int64_t{1}, std::multiplies<>());
  };
  absl::Span<const int64_t> dimensions = physical_shape.dimensions();
  int64_t outer_dim_size = product(dimensions.subspan(0, sort_dimension));
  int64_t sort_dim_size = dimensions[sort_dimension];
  int64_t inner_dim_size = product(dimensions.subspan(sort_dimension + 1));
  return internal::SortDims{outer_dim_size, sort_dim_size, inner_dim_size};
}

template <PrimitiveType Type>
constexpr bool IsSupportedSortKeyType() {
  using NativeT = primitive_util::NativeTypeOf<Type>;
  return (std::is_arithmetic_v<NativeT> && !std::is_same_v<NativeT, bool>) ||
         std::is_same_v<NativeT, bfloat16> || std::is_same_v<NativeT, half>;
}

class LiteOp {
 public:
  virtual ~LiteOp() = default;
  virtual tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> Execute(
      absl::Span<const se::DeviceAddressBase> buffers,
      FunctionLibrary* function_library,
      const NanoRtExecutable::ExecuteOptions& options) = 0;
};

class LiteSequence {
 public:
  explicit LiteSequence(std::vector<std::unique_ptr<LiteOp>> ops)
      : ops_(std::move(ops)) {}

  tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> Execute(
      absl::Span<const se::DeviceAddressBase> buffers,
      FunctionLibrary* function_library,
      const NanoRtExecutable::ExecuteOptions& options) {
    return ExecuteFrom(0, buffers, function_library, options);
  }

 private:
  tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> ExecuteFrom(
      size_t start_idx, absl::Span<const se::DeviceAddressBase> buffers,
      FunctionLibrary* function_library,
      const NanoRtExecutable::ExecuteOptions& options) {
    for (size_t i = start_idx; i < ops_.size(); ++i) {
      auto event = ops_[i]->Execute(buffers, function_library, options);
      if (ABSL_PREDICT_FALSE(!event.IsAvailable())) {
        auto out =
            tsl::MakeConstructedAsyncValueRef<NanoRtExecutable::ExecuteEvent>();
        std::vector<se::DeviceAddressBase> saved_buffers(buffers.begin(),
                                                         buffers.end());
        event.AndThen([this, i, saved_buffers = std::move(saved_buffers),
                       function_library, &options,
                       out](absl::Status status) mutable {
          if (!status.ok()) {
            out.SetError(std::move(status));
            return;
          }
          auto next =
              ExecuteFrom(i + 1, saved_buffers, function_library, options);
          next.AndThen([out](absl::Status s) mutable {
            if (!s.ok()) {
              out.SetError(std::move(s));
            } else {
              out.SetStateConcrete();
            }
          });
        });
        return out;
      }
      if (ABSL_PREDICT_FALSE(event.IsError())) {
        return event;
      }
    }
    return tsl::MakeAvailableAsyncValueRef<NanoRtExecutable::ExecuteEvent>();
  }

  std::vector<std::unique_ptr<LiteOp>> ops_;
};

class LiteKernelOp final : public LiteOp {
 public:
  static absl::StatusOr<std::unique_ptr<LiteOp>> Create(
      const KernelThunkProto& proto) {
    std::vector<LiteSlice> slices;
    slices.reserve(proto.arguments_buffers_size() +
                   proto.results_buffers_size());
    for (const auto& arg : proto.arguments_buffers()) {
      slices.push_back(ParseSlice(arg.slice()));
    }
    for (const auto& res : proto.results_buffers()) {
      slices.push_back(ParseSlice(res.slice()));
    }
    NumWorkGroups num_workgroups{
        static_cast<uint64_t>(proto.num_workgroups().x()),
        static_cast<uint64_t>(proto.num_workgroups().y()),
        static_cast<uint64_t>(proto.num_workgroups().z())};
    std::optional<uint64_t> min_alignment;
    if (proto.min_alignment().contains_value()) {
      min_alignment = static_cast<uint64_t>(proto.min_alignment().value());
    }
    return std::make_unique<LiteKernelOp>(proto.kernel_name(), num_workgroups,
                                          min_alignment, std::move(slices));
  }

  LiteKernelOp(std::string kernel_name, NumWorkGroups num_workgroups,
               std::optional<uint64_t> min_alignment,
               std::vector<LiteSlice> slices)
      : kernel_name_(std::move(kernel_name)),
        num_workgroups_(num_workgroups),
        min_alignment_(min_alignment),
        call_once_(num_workgroups_ == NumWorkGroups()),
        slices_(std::move(slices)) {}

  tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> Execute(
      absl::Span<const se::DeviceAddressBase> buffers,
      FunctionLibrary* function_library,
      const NanoRtExecutable::ExecuteOptions& options) override {
    absl::call_once(init_flag_, [&] {
      absl::StatusOr<FunctionLibrary::Kernel*> fn =
          function_library->ResolveFunction<FunctionLibrary::Kernel>(
              kernel_name_);
      if (fn.ok()) {
        kernel_.emplace(slices_.size(), *fn);
      } else {
        kernel_ = fn.status();
      }
    });
    if (ABSL_PREDICT_FALSE(!kernel_.ok())) {
      return kernel_.status();
    }

    absl::InlinedVector<XLA_CPU_KernelArg, 16> args(slices_.size());
    for (size_t i = 0; i < slices_.size(); ++i) {
      ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase mem,
                            ResolveSlice(buffers, slices_[i]));
      args[i] = XLA_CPU_KernelArg{mem.opaque(), mem.size()};
    }

    if constexpr (tsl::kIsDebugBuild) {
      uint64_t min_align = min_alignment_.value_or(0);
      if (min_align > 0) {
        for (size_t i = 0; i < args.size(); ++i) {
          auto ptr = reinterpret_cast<uintptr_t>(args[i].data);
          if (ABSL_PREDICT_FALSE((ptr & (min_align - 1)) != 0)) {
            return Internal(
                "Host kernel %s buffer argument #%d (%p) is not aligned to a "
                "required minimum alignment of %d bytes",
                kernel_name_, i, args[i].data, min_align);
          }
        }
      }
    }

    if (ABSL_PREDICT_TRUE(call_once_)) {
      ABSL_RETURN_IF_ERROR(kernel_->CallOnce(args));
      return tsl::MakeAvailableAsyncValueRef<NanoRtExecutable::ExecuteEvent>();
    }
    if (options.intra_op_thread_pool() != nullptr) {
      return kernel_->Launch(num_workgroups_, args,
                             options.intra_op_thread_pool());
    }
    ABSL_RETURN_IF_ERROR(kernel_->Launch(num_workgroups_, args));
    return tsl::MakeAvailableAsyncValueRef<NanoRtExecutable::ExecuteEvent>();
  }

 private:
  std::string kernel_name_;
  NumWorkGroups num_workgroups_;
  std::optional<uint64_t> min_alignment_;
  bool call_once_;
  std::vector<LiteSlice> slices_;
  absl::once_flag init_flag_;
  absl::StatusOr<Kernel> kernel_;
};

class LiteCopyOp final : public LiteOp {
 public:
  static absl::StatusOr<std::unique_ptr<LiteOp>> Create(
      const CopyThunkProto& proto) {
    ABSL_ASSIGN_OR_RETURN(Shape src_shape,
                          Shape::FromProto(proto.src_buffer_shape().shape()));
    ABSL_ASSIGN_OR_RETURN(Shape dst_shape,
                          Shape::FromProto(proto.dst_buffer_shape().shape()));
    LiteSlice src_slice = ParseSlice(proto.src_buffer_shape().slice());
    LiteSlice dst_slice = ParseSlice(proto.dst_buffer_shape().slice());

    int64_t size_in_bytes = ShapeUtil::ByteSizeOf(src_shape);
    return std::make_unique<LiteCopyOp>(src_slice, dst_slice, size_in_bytes,
                                        std::move(src_shape),
                                        std::move(dst_shape));
  }

  LiteCopyOp(LiteSlice src_slice, LiteSlice dst_slice, int64_t size_in_bytes,
             Shape src_shape, Shape dst_shape)
      : src_slice_(src_slice),
        dst_slice_(dst_slice),
        size_in_bytes_(size_in_bytes),
        src_shape_(std::move(src_shape)),
        dst_shape_(std::move(dst_shape)) {}

  tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> Execute(
      absl::Span<const se::DeviceAddressBase> buffers,
      FunctionLibrary* /*function_library*/,
      const NanoRtExecutable::ExecuteOptions& /*options*/) override {
    ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase src,
                          ResolveSlice(buffers, src_slice_));
    ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase dst,
                          ResolveSlice(buffers, dst_slice_));
    if (size_in_bytes_ == 0) {
      return tsl::MakeAvailableAsyncValueRef<NanoRtExecutable::ExecuteEvent>();
    }
    if (src_shape_ != dst_shape_) {
      auto src_strides = ShapeUtil::ByteStrides(src_shape_);
      auto dst_strides = ShapeUtil::ByteStrides(dst_shape_);
      if (!src_strides.has_value() || !dst_strides.has_value()) {
        return InvalidArgument("Copy shapes must have valid byte strides");
      }
      int64_t elem_size =
          ShapeUtil::ByteSizeOfPrimitiveType(src_shape_.element_type());
      const char* src_base = static_cast<const char*>(src.opaque());
      char* dst_base = static_cast<char*>(dst.opaque());
      ShapeUtil::ForEachIndexNoStatus(
          src_shape_, [&](absl::Span<const int64_t> index) {
            int64_t src_offset = 0;
            int64_t dst_offset = 0;
            for (size_t d = 0; d < index.size(); ++d) {
              src_offset += index[d] * (*src_strides)[d];
              dst_offset += index[d] * (*dst_strides)[d];
            }
            std::memcpy(dst_base + dst_offset, src_base + src_offset,
                        elem_size);
            return true;
          });
    } else {
      std::memcpy(dst.opaque(), src.opaque(), size_in_bytes_);
    }
    return tsl::MakeAvailableAsyncValueRef<NanoRtExecutable::ExecuteEvent>();
  }

 private:
  LiteSlice src_slice_;
  LiteSlice dst_slice_;
  int64_t size_in_bytes_;
  Shape src_shape_;
  Shape dst_shape_;
};

struct LiteDotShape {
  int64_t batch_size;
  Shape lhs_matmul_shape;
  Shape rhs_matmul_shape;
  Shape out_matmul_shape;
};

struct LiteDotCanonicalDims {
  int64_t m;
  int64_t k;
  int64_t n;
  bool lhs_column_major;
  bool lhs_canonical;
  bool rhs_column_major;
  bool rhs_canonical;
  bool output_column_major;
};

absl::StatusOr<LiteDotShape> GetLiteDotShape(
    const DotDimensionNumbers& dot_dimensions, const Shape& lhs_shape,
    const Shape& rhs_shape, const Shape& out_shape) {
  std::vector<int64_t> batch_dims(dot_dimensions.lhs_batch_dimensions().size());
  absl::c_iota(batch_dims, 0);
  if (!absl::c_equal(dot_dimensions.lhs_batch_dimensions(), batch_dims) ||
      !absl::c_equal(dot_dimensions.rhs_batch_dimensions(), batch_dims)) {
    return InvalidArgument(
        "Batch dimensions must be contiguous and start at 0");
  }
  int64_t num_batch_dims = batch_dims.size();
  int64_t batch_size =
      std::accumulate(out_shape.dimensions().begin(),
                      out_shape.dimensions().begin() + num_batch_dims, 1LL,
                      std::multiplies<int64_t>());
  Shape lhs_matmul_shape = ShapeUtil::DeleteDimensions(batch_dims, lhs_shape);
  Shape rhs_matmul_shape = ShapeUtil::DeleteDimensions(batch_dims, rhs_shape);
  Shape out_matmul_shape = ShapeUtil::DeleteDimensions(batch_dims, out_shape);
  return LiteDotShape{batch_size, std::move(lhs_matmul_shape),
                      std::move(rhs_matmul_shape), std::move(out_matmul_shape)};
}

absl::StatusOr<LiteDotCanonicalDims> GetLiteDotCanonicalDims(
    const DotDimensionNumbers& dot_dimensions, const LiteDotShape& dot_shape) {
  absl::InlinedVector<int64_t, 2> lhs_contracting_dims(
      dot_dimensions.lhs_contracting_dimensions().begin(),
      dot_dimensions.lhs_contracting_dimensions().end());
  absl::InlinedVector<int64_t, 2> rhs_contracting_dims(
      dot_dimensions.rhs_contracting_dimensions().begin(),
      dot_dimensions.rhs_contracting_dimensions().end());
  for (int64_t& dim : lhs_contracting_dims) {
    dim -= dot_dimensions.lhs_batch_dimensions_size();
  }
  for (int64_t& dim : rhs_contracting_dims) {
    dim -= dot_dimensions.rhs_batch_dimensions_size();
  }
  auto is_column_major = [](const Shape& shape) {
    return shape.dimensions().size() > 1 &&
           LayoutUtil::Minor(shape.layout(), 0) == 0;
  };
  return LiteDotCanonicalDims{
      /*m=*/dot_shape.lhs_matmul_shape.dimensions().size() <= 1
          ? int64_t{1}
          : dot_shape.lhs_matmul_shape.dimensions(1 - lhs_contracting_dims[0]),
      /*k=*/dot_shape.lhs_matmul_shape.dimensions(lhs_contracting_dims[0]),
      /*n=*/dot_shape.rhs_matmul_shape.dimensions().size() <= 1
          ? int64_t{1}
          : dot_shape.rhs_matmul_shape.dimensions(1 - rhs_contracting_dims[0]),
      /*lhs_column_major=*/is_column_major(dot_shape.lhs_matmul_shape),
      /*lhs_canonical=*/dot_shape.lhs_matmul_shape.dimensions().size() <= 1 ||
          lhs_contracting_dims[0] == 1,
      /*rhs_column_major=*/is_column_major(dot_shape.rhs_matmul_shape),
      /*rhs_canonical=*/rhs_contracting_dims[0] == 0,
      /*output_column_major=*/is_column_major(dot_shape.out_matmul_shape)};
}

class LiteDotOp final : public LiteOp {
 public:
  static absl::StatusOr<std::unique_ptr<LiteOp>> Create(
      const DotThunkProto& proto) {
    ABSL_ASSIGN_OR_RETURN(Shape lhs_shape,
                          Shape::FromProto(proto.lhs_buffer_shape().shape()));
    ABSL_ASSIGN_OR_RETURN(Shape rhs_shape,
                          Shape::FromProto(proto.rhs_buffer_shape().shape()));
    ABSL_ASSIGN_OR_RETURN(Shape out_shape,
                          Shape::FromProto(proto.out_buffer_shape().shape()));
    ABSL_ASSIGN_OR_RETURN(LiteDotShape dot_shape,
                          GetLiteDotShape(proto.dot_dimensions(), lhs_shape,
                                          rhs_shape, out_shape));
    ABSL_ASSIGN_OR_RETURN(
        LiteDotCanonicalDims dot_canonical_dims,
        GetLiteDotCanonicalDims(proto.dot_dimensions(), dot_shape));
    return std::make_unique<LiteDotOp>(
        ParseSlice(proto.lhs_buffer_shape().slice()),
        ParseSlice(proto.rhs_buffer_shape().slice()),
        ParseSlice(proto.out_buffer_shape().slice()), std::move(dot_shape),
        dot_canonical_dims);
  }

  LiteDotOp(LiteSlice lhs_slice, LiteSlice rhs_slice, LiteSlice out_slice,
            LiteDotShape dot_shape, LiteDotCanonicalDims dot_canonical_dims)
      : lhs_slice_(lhs_slice),
        rhs_slice_(rhs_slice),
        out_slice_(out_slice),
        dot_shape_(std::move(dot_shape)),
        dot_canonical_dims_(dot_canonical_dims) {}

  tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> Execute(
      absl::Span<const se::DeviceAddressBase> buffers,
      FunctionLibrary* /*function_library*/,
      const NanoRtExecutable::ExecuteOptions& options) override {
    ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase lhs_data,
                          ResolveSlice(buffers, lhs_slice_));
    ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase rhs_data,
                          ResolveSlice(buffers, rhs_slice_));
    ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase out_data,
                          ResolveSlice(buffers, out_slice_));

    void* out = out_data.opaque();
    void* lhs = lhs_data.opaque();
    void* rhs = rhs_data.opaque();

    int64_t m = dot_canonical_dims_.m;
    int64_t n = dot_canonical_dims_.n;
    int64_t k = dot_canonical_dims_.k;

    bool transpose_lhs = (dot_canonical_dims_.lhs_canonical !=
                          dot_canonical_dims_.lhs_column_major);
    bool transpose_rhs = (dot_canonical_dims_.rhs_canonical !=
                          dot_canonical_dims_.rhs_column_major);

    if (!dot_canonical_dims_.output_column_major) {
      std::swap(m, n);
      std::swap(lhs, rhs);
      std::swap(transpose_lhs, transpose_rhs);
      transpose_lhs = !transpose_lhs;
      transpose_rhs = !transpose_rhs;
    }

    PrimitiveType lhs_dtype = dot_shape_.lhs_matmul_shape.element_type();
    PrimitiveType rhs_dtype = dot_shape_.rhs_matmul_shape.element_type();
    PrimitiveType out_dtype = dot_shape_.out_matmul_shape.element_type();

    int64_t lhs_stride = m * k * primitive_util::ByteWidth(lhs_dtype);
    int64_t rhs_stride = k * n * primitive_util::ByteWidth(rhs_dtype);
    int64_t out_stride = m * n * primitive_util::ByteWidth(out_dtype);

    auto batch_ptr = [](void* ptr, int64_t stride, int64_t index) -> void* {
      return static_cast<uint8_t*>(ptr) + stride * index;
    };

    tsl::CountDownAsyncValueRef<NanoRtExecutable::ExecuteEvent> state(
        dot_shape_.batch_size);

    auto dispatch = [&](auto lhs_type, auto rhs_type, auto out_type) {
      using LhsType = decltype(lhs_type);
      using RhsType = decltype(rhs_type);
      using OutType = decltype(out_type);
      for (int64_t i = 0; i < dot_shape_.batch_size; ++i) {
        internal::TypedMatMul<LhsType, RhsType, OutType>(
            options.intra_op_thread_pool(), batch_ptr(out, out_stride, i),
            batch_ptr(lhs, lhs_stride, i), batch_ptr(rhs, rhs_stride, i), m, n,
            k, transpose_lhs, transpose_rhs,
            [state]() mutable { state.CountDown(); });
      }
    };

    auto dispatch_same_type = [&](auto type_tag) {
      dispatch(type_tag, type_tag, type_tag);
    };

    if (lhs_dtype == rhs_dtype && lhs_dtype == out_dtype) {
      switch (lhs_dtype) {
        case BF16:
          dispatch_same_type(bfloat16{});
          break;
        case F16:
          dispatch_same_type(half{});
          break;
        case F32:
          dispatch_same_type(float{});
          break;
        case F64:
          dispatch_same_type(double{});
          break;
        case S32:
          dispatch_same_type(int32_t{});
          break;
        case C64:
          dispatch_same_type(std::complex<float>{});
          break;
        case C128:
          dispatch_same_type(std::complex<double>{});
          break;
        default:
          return Unimplemented("Unsupported element type for Dot: %s",
                               PrimitiveType_Name(lhs_dtype));
      }
    } else if (lhs_dtype == S8 && rhs_dtype == S8 && out_dtype == S32) {
      dispatch(int8_t{}, int8_t{}, int32_t{});
    } else {
      return Unimplemented("Unsupported element types for Dot: %s x %s = %s",
                           PrimitiveType_Name(lhs_dtype),
                           PrimitiveType_Name(rhs_dtype),
                           PrimitiveType_Name(out_dtype));
    }

    return state.AsRef();
  }

 private:
  LiteSlice lhs_slice_;
  LiteSlice rhs_slice_;
  LiteSlice out_slice_;
  LiteDotShape dot_shape_;
  LiteDotCanonicalDims dot_canonical_dims_;
};

class LiteTopKOp final : public LiteOp {
 public:
  explicit LiteTopKOp(const TopKThunkProto& proto)
      : batch_size_(proto.batch_size()),
        input_size_(proto.input_size()),
        k_(proto.k()),
        values_slice_(ParseSlice(proto.values_buffer())),
        output_slice_(ParseSlice(proto.output_buffer())),
        indices_slice_(ParseSlice(proto.indices_buffer())) {}

  tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> Execute(
      absl::Span<const se::DeviceAddressBase> buffers,
      FunctionLibrary* /*function_library*/,
      const NanoRtExecutable::ExecuteOptions& /*options*/) override {
    ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase values,
                          ResolveSlice(buffers, values_slice_));
    ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase output,
                          ResolveSlice(buffers, output_slice_));
    ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase indices,
                          ResolveSlice(buffers, indices_slice_));
    internal::TopK<float>(batch_size_, input_size_, k_,
                          static_cast<const float*>(values.opaque()),
                          static_cast<float*>(output.opaque()),
                          static_cast<int32_t*>(indices.opaque()));
    return tsl::MakeAvailableAsyncValueRef<NanoRtExecutable::ExecuteEvent>();
  }

 private:
  int64_t batch_size_;
  int64_t input_size_;
  int64_t k_;
  LiteSlice values_slice_;
  LiteSlice output_slice_;
  LiteSlice indices_slice_;
};

class LiteSortOp final : public LiteOp {
 public:
  struct Input {
    LiteSlice slice;
    Shape shape;
  };

  static absl::StatusOr<std::unique_ptr<LiteOp>> Create(
      const SortThunkProto& proto) {
    if (proto.inputs_shapes_size() == 0) {
      return Internal("Sort inputs must not be empty");
    }
    std::vector<Input> inputs;
    inputs.reserve(proto.inputs_shapes_size());
    for (const auto& in : proto.inputs_shapes()) {
      ABSL_ASSIGN_OR_RETURN(Shape shape, Shape::FromProto(in.shape()));
      inputs.push_back(Input{ParseSlice(in.slice()), std::move(shape)});
    }
    internal::SortDims sort_dims =
        ComputeSortDims(inputs[0].shape, proto.dimension());
    std::optional<internal::SortDirection> direction;
    if (proto.direction().contains_value()) {
      if (proto.direction().value() == SortDirectionProto::ASCENDING) {
        direction = internal::SortDirection::kAscending;
      } else if (proto.direction().value() == SortDirectionProto::DESCENDING) {
        direction = internal::SortDirection::kDescending;
      }
    }
    return std::make_unique<LiteSortOp>(std::move(inputs), proto.is_stable(),
                                        proto.comparator_name(), sort_dims,
                                        direction);
  }

  LiteSortOp(std::vector<Input> inputs, bool is_stable,
             std::string comparator_name, internal::SortDims sort_dims,
             std::optional<internal::SortDirection> direction)
      : inputs_(std::move(inputs)),
        is_stable_(is_stable),
        comparator_name_(std::move(comparator_name)),
        sort_dims_(sort_dims),
        direction_(direction) {}

  tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> Execute(
      absl::Span<const se::DeviceAddressBase> buffers,
      FunctionLibrary* function_library,
      const NanoRtExecutable::ExecuteOptions& /*options*/) override {
    absl::InlinedVector<std::byte*, 8> raw_data;
    absl::InlinedVector<size_t, 8> primitive_sizes;
    raw_data.reserve(inputs_.size());
    primitive_sizes.reserve(inputs_.size());
    for (const Input& input : inputs_) {
      ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase mem,
                            ResolveSlice(buffers, input.slice));
      raw_data.push_back(reinterpret_cast<std::byte*>(mem.opaque()));
      primitive_sizes.push_back(
          primitive_util::ByteWidth(input.shape.element_type()));
    }

    absl::call_once(init_flag_, [&]() {
      absl::StatusOr<FunctionLibrary::Comparator*> comparator =
          function_library->ResolveFunction<FunctionLibrary::Comparator>(
              comparator_name_);
      if (comparator.ok()) {
        less_than_ = [comparator](const void** data) {
          bool result;
          (*comparator)(&result, nullptr, data, nullptr, nullptr, nullptr);
          ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&result, sizeof(result));
          return result;
        };
      } else {
        less_than_ = comparator.status();
      }
    });
    if (ABSL_PREDICT_FALSE(!less_than_.ok())) {
      return less_than_.status();
    }
    internal::LessThan* less_than = &less_than_.value();
    int64_t num_slices = sort_dims_.outer_dim_size * sort_dims_.inner_dim_size;
    PrimitiveType first_element_type = inputs_[0].shape.element_type();

    if (raw_data.size() == 1 && direction_.has_value()) {
      primitive_util::ArrayTypeSwitch(
          [&](auto type) {
            if constexpr (IsSupportedSortKeyType<type>()) {
              using T = primitive_util::NativeTypeOf<type>;
              internal::SortInplace<T>(sort_dims_, 0, num_slices,
                                       reinterpret_cast<T*>(raw_data[0]),
                                       is_stable_, *direction_);
            } else {
              internal::SortInplace(sort_dims_, 0, num_slices, raw_data,
                                    primitive_sizes, is_stable_, less_than);
            }
          },
          first_element_type);
    } else if (raw_data.size() == 2 && direction_.has_value() &&
               (sort_dims_.inner_dim_size == 1 ||
                sort_dims_.sort_dim_size <= 1000)) {
      size_t val_size = primitive_sizes[1];
      primitive_util::ArrayTypeSwitch(
          [&](auto key_type) {
            if constexpr (IsSupportedSortKeyType<key_type>()) {
              using Key = primitive_util::NativeTypeOf<key_type>;
              auto* keys = reinterpret_cast<Key*>(raw_data[0]);
              switch (val_size) {
                case 1:
                  internal::Sort2DKeyValue<Key, uint8_t>(
                      sort_dims_, 0, num_slices, keys,
                      reinterpret_cast<uint8_t*>(raw_data[1]), is_stable_,
                      *direction_);
                  break;
                case 2:
                  internal::Sort2DKeyValue<Key, uint16_t>(
                      sort_dims_, 0, num_slices, keys,
                      reinterpret_cast<uint16_t*>(raw_data[1]), is_stable_,
                      *direction_);
                  break;
                case 4:
                  internal::Sort2DKeyValue<Key, uint32_t>(
                      sort_dims_, 0, num_slices, keys,
                      reinterpret_cast<uint32_t*>(raw_data[1]), is_stable_,
                      *direction_);
                  break;
                case 8:
                  internal::Sort2DKeyValue<Key, uint64_t>(
                      sort_dims_, 0, num_slices, keys,
                      reinterpret_cast<uint64_t*>(raw_data[1]), is_stable_,
                      *direction_);
                  break;
                default:
                  internal::SortInplace(sort_dims_, 0, num_slices, raw_data,
                                        primitive_sizes, is_stable_, less_than);
                  break;
              }
            } else {
              internal::SortInplace(sort_dims_, 0, num_slices, raw_data,
                                    primitive_sizes, is_stable_, less_than);
            }
          },
          first_element_type);
    } else {
      internal::SortInplace(sort_dims_, 0, num_slices, raw_data,
                            primitive_sizes, is_stable_, less_than);
    }

    return tsl::MakeAvailableAsyncValueRef<NanoRtExecutable::ExecuteEvent>();
  }

 private:
  std::vector<Input> inputs_;
  bool is_stable_;
  std::string comparator_name_;
  internal::SortDims sort_dims_;
  std::optional<internal::SortDirection> direction_;
  absl::once_flag init_flag_;
  absl::StatusOr<internal::LessThan> less_than_;
};

class LiteRngStateOp final : public LiteOp {
 public:
  explicit LiteRngStateOp(const RngGetAndUpdateStateThunkProto& proto)
      : state_slice_(ParseSlice(proto.state_buffer())),
        rng_state_(proto.delta()) {}

  tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> Execute(
      absl::Span<const se::DeviceAddressBase> buffers,
      FunctionLibrary* /*function_library*/,
      const NanoRtExecutable::ExecuteOptions& /*options*/) override {
    ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase state_data,
                          ResolveSlice(buffers, state_slice_));
    if (state_data.size() != sizeof(absl::int128)) {
      return InvalidArgument("Invalid state buffer size: %d",
                             state_data.size());
    }
    rng_state_.GetAndUpdateState(static_cast<uint64_t*>(state_data.opaque()));
    return tsl::MakeAvailableAsyncValueRef<NanoRtExecutable::ExecuteEvent>();
  }

 private:
  LiteSlice state_slice_;
  RngState rng_state_;
};

class LiteRngSeedOp final : public LiteOp {
 public:
  explicit LiteRngSeedOp(const RngSeedThunkProto& proto)
      : dest_slice_(ParseSlice(proto.dest_buffer())) {}

  tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> Execute(
      absl::Span<const se::DeviceAddressBase> buffers,
      FunctionLibrary* /*function_library*/,
      const NanoRtExecutable::ExecuteOptions& /*options*/) override {
    ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase dest_data,
                          ResolveSlice(buffers, dest_slice_));
    if (dest_data.size() != sizeof(uint64_t)) {
      return InvalidArgument("Invalid seed buffer size: %u", dest_data.size());
    }
    static std::atomic<uint64_t> state{0x9e3779b97f4a7c15ULL};
    uint64_t z =
        state.fetch_add(0x9e3779b97f4a7c15ULL, std::memory_order_relaxed);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    uint64_t seed = (z ^ (z >> 31)) | 1ULL;
    std::memcpy(dest_data.opaque(), &seed, sizeof(uint64_t));
    return tsl::MakeAvailableAsyncValueRef<NanoRtExecutable::ExecuteEvent>();
  }

 private:
  LiteSlice dest_slice_;
};

class LiteLogicalIdOp final : public LiteOp {
 public:
  explicit LiteLogicalIdOp(LiteSlice slice) : slice_(slice) {}

  tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> Execute(
      absl::Span<const se::DeviceAddressBase> buffers,
      FunctionLibrary* /*function_library*/,
      const NanoRtExecutable::ExecuteOptions& /*options*/) override {
    ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase data,
                          ResolveSlice(buffers, slice_));
    if (data.size() != sizeof(int32_t)) {
      return InvalidArgument("Invalid logical id buffer size: %u", data.size());
    }
    int32_t id = 0;
    std::memcpy(data.opaque(), &id, sizeof(int32_t));
    return tsl::MakeAvailableAsyncValueRef<NanoRtExecutable::ExecuteEvent>();
  }

 private:
  LiteSlice slice_;
};

class LiteCallOp final : public LiteOp {
 public:
  explicit LiteCallOp(std::unique_ptr<LiteSequence> sequence)
      : sequence_(std::move(sequence)) {}

  tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> Execute(
      absl::Span<const se::DeviceAddressBase> buffers,
      FunctionLibrary* function_library,
      const NanoRtExecutable::ExecuteOptions& options) override {
    return sequence_->Execute(buffers, function_library, options);
  }

 private:
  std::unique_ptr<LiteSequence> sequence_;
};

class LiteConditionalOp final : public LiteOp {
 public:
  LiteConditionalOp(LiteSlice branch_index_slice,
                    std::vector<std::unique_ptr<LiteSequence>> branches)
      : branch_index_slice_(branch_index_slice),
        branches_(std::move(branches)) {}

  tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> Execute(
      absl::Span<const se::DeviceAddressBase> buffers,
      FunctionLibrary* function_library,
      const NanoRtExecutable::ExecuteOptions& options) override {
    ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase index_data,
                          ResolveSlice(buffers, branch_index_slice_));
    size_t branch_idx = 0;
    if (branch_index_slice_.size == sizeof(bool)) {
      bool pred = *reinterpret_cast<const bool*>(index_data.opaque());
      branch_idx = pred ? 0 : 1;
    } else if (branch_index_slice_.size == sizeof(int32_t)) {
      int32_t idx = *reinterpret_cast<const int32_t*>(index_data.opaque());
      branch_idx = (idx < 0 || static_cast<size_t>(idx) >= branches_.size())
                       ? branches_.size() - 1
                       : static_cast<size_t>(idx);
    } else {
      return Internal("Unsupported branch index buffer size %d",
                      branch_index_slice_.size);
    }
    return branches_[branch_idx]->Execute(buffers, function_library, options);
  }

 private:
  LiteSlice branch_index_slice_;
  std::vector<std::unique_ptr<LiteSequence>> branches_;
};

class LiteWhileOp final : public LiteOp {
 public:
  LiteWhileOp(LiteSlice cond_slice, std::optional<int64_t> trip_count,
              std::unique_ptr<LiteSequence> cond_seq,
              std::unique_ptr<LiteSequence> body_seq)
      : cond_slice_(cond_slice),
        trip_count_(trip_count),
        cond_seq_(std::move(cond_seq)),
        body_seq_(std::move(body_seq)) {}

  tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> Execute(
      absl::Span<const se::DeviceAddressBase> buffers,
      FunctionLibrary* function_library,
      const NanoRtExecutable::ExecuteOptions& options) override {
    if (trip_count_.has_value()) {
      for (int64_t i = 0; i < *trip_count_; ++i) {
        auto event = body_seq_->Execute(buffers, function_library, options);
        tsl::BlockUntilReady(event);
        if (event.IsError()) return event;
      }
      return tsl::MakeAvailableAsyncValueRef<NanoRtExecutable::ExecuteEvent>();
    }

    ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase cond_data,
                          ResolveSlice(buffers, cond_slice_));
    const bool* condition = reinterpret_cast<const bool*>(cond_data.opaque());

    auto init_event = cond_seq_->Execute(buffers, function_library, options);
    tsl::BlockUntilReady(init_event);
    if (init_event.IsError()) return init_event;

    while (*condition) {
      auto body_event = body_seq_->Execute(buffers, function_library, options);
      tsl::BlockUntilReady(body_event);
      if (body_event.IsError()) return body_event;

      auto cond_event = cond_seq_->Execute(buffers, function_library, options);
      tsl::BlockUntilReady(cond_event);
      if (cond_event.IsError()) return cond_event;
    }
    return tsl::MakeAvailableAsyncValueRef<NanoRtExecutable::ExecuteEvent>();
  }

 private:
  LiteSlice cond_slice_;
  std::optional<int64_t> trip_count_;
  std::unique_ptr<LiteSequence> cond_seq_;
  std::unique_ptr<LiteSequence> body_seq_;
};

absl::StatusOr<std::unique_ptr<LiteSequence>> BuildLiteSequence(
    const ThunkSequenceProto& proto);

absl::StatusOr<std::unique_ptr<LiteOp>> BuildLiteOp(const ThunkProto& proto) {
  switch (proto.impl_case()) {
    case ThunkProto::kKernelThunk:
      return LiteKernelOp::Create(proto.kernel_thunk());
    case ThunkProto::kCopyThunk:
      return LiteCopyOp::Create(proto.copy_thunk());
    case ThunkProto::kDotThunk:
      return LiteDotOp::Create(proto.dot_thunk());
    case ThunkProto::kTopKThunk:
      return std::make_unique<LiteTopKOp>(proto.top_k_thunk());
    case ThunkProto::kSortThunk:
      return LiteSortOp::Create(proto.sort_thunk());
    case ThunkProto::kRngGetAndUpdateStateThunk:
      return std::make_unique<LiteRngStateOp>(
          proto.rng_get_and_update_state_thunk());
    case ThunkProto::kRngSeedThunk:
      return std::make_unique<LiteRngSeedOp>(proto.rng_seed_thunk());
    case ThunkProto::kPartitionIdThunk:
      return std::make_unique<LiteLogicalIdOp>(
          ParseSlice(proto.partition_id_thunk().logical_id_buffer()));
    case ThunkProto::kReplicaIdThunk:
      return std::make_unique<LiteLogicalIdOp>(
          ParseSlice(proto.replica_id_thunk().logical_id_buffer()));
    case ThunkProto::kCallThunk: {
      ABSL_ASSIGN_OR_RETURN(
          auto seq, BuildLiteSequence(proto.call_thunk().called_sequence()));
      return std::make_unique<LiteCallOp>(std::move(seq));
    }
    case ThunkProto::kConditionalThunk: {
      std::vector<std::unique_ptr<LiteSequence>> branches;
      branches.reserve(proto.conditional_thunk().branch_sequences_size());
      for (const auto& branch_proto :
           proto.conditional_thunk().branch_sequences()) {
        ABSL_ASSIGN_OR_RETURN(auto seq, BuildLiteSequence(branch_proto));
        branches.push_back(std::move(seq));
      }
      return std::make_unique<LiteConditionalOp>(
          ParseSlice(proto.conditional_thunk().branch_index_buffer()),
          std::move(branches));
    }
    case ThunkProto::kWhileThunk: {
      ABSL_ASSIGN_OR_RETURN(
          auto cond_seq,
          BuildLiteSequence(proto.while_thunk().cond_sequence()));
      ABSL_ASSIGN_OR_RETURN(
          auto body_seq,
          BuildLiteSequence(proto.while_thunk().body_sequence()));
      std::optional<int64_t> trip_count;
      if (proto.while_thunk().trip_count().contains_value()) {
        trip_count = proto.while_thunk().trip_count().value();
      }
      return std::make_unique<LiteWhileOp>(
          ParseSlice(proto.while_thunk().cond_buffer()), trip_count,
          std::move(cond_seq), std::move(body_seq));
    }
    default:
      return Unimplemented("Unsupported thunk kind in Lite AOT mode: %s (%d)",
                           proto.kind(), proto.impl_case());
  }
}

absl::StatusOr<std::unique_ptr<LiteSequence>> BuildLiteSequence(
    const ThunkSequenceProto& proto) {
  std::vector<std::unique_ptr<LiteOp>> ops;
  ops.reserve(proto.thunks_size());
  for (const ThunkProto& thunk_proto : proto.thunks()) {
    ABSL_ASSIGN_OR_RETURN(auto op, BuildLiteOp(thunk_proto));
    ops.push_back(std::move(op));
  }
  return std::make_unique<LiteSequence>(std::move(ops));
}

class LiteExecutableRunner final : public NanoRtExecutable::ExecutableRunner {
 public:
  LiteExecutableRunner(std::string /*module_name*/,
                       std::unique_ptr<FunctionLibrary> function_library,
                       std::vector<ConstantAllocation> constants,
                       std::unique_ptr<LiteSequence> sequence)
      : function_library_(std::move(function_library)),
        constants_(std::move(constants)),
        sequence_(std::move(sequence)) {}

  tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> Execute(
      std::vector<se::DeviceAddressBase> buffers,
      const NanoRtExecutable::ExecuteOptions& options) override {
    for (const auto& constant : constants_) {
      if (constant.index >= 0) {
        buffers[constant.index] = constant.AsDeviceAddress();
      }
    }
    return sequence_->Execute(buffers, function_library_.get(), options);
  }

 private:
  std::unique_ptr<FunctionLibrary> function_library_;
  std::vector<ConstantAllocation> constants_;
  std::unique_ptr<LiteSequence> sequence_;
};

using ArgumentIndex = std::pair<size_t, ShapeIndex>;

absl::StatusOr<std::vector<size_t>> ResolveArgumentsMappingFromProto(
    const ProgramShape& program_shape,
    const BufferAssignmentProto& buffer_assignment_proto) {
  absl::flat_hash_map<ArgumentIndex, size_t> executable_arg_index;
  for (size_t i = 0; i < program_shape.parameters_size(); ++i) {
    ShapeUtil::ForEachLeafShape(
        program_shape.parameters(i),
        [&](const Shape& shape, const ShapeIndex& index) {
          if (shape.IsToken()) {
            return;
          }
          size_t arg_index = executable_arg_index.size();
          executable_arg_index[ArgumentIndex{i, index}] = arg_index;
        });
  }

  std::vector<size_t> argument_to_allocation_index(executable_arg_index.size());
  for (const BufferAllocationProto& alloc_proto :
       buffer_assignment_proto.buffer_allocations()) {
    if (alloc_proto.is_entry_computation_parameter()) {
      ArgumentIndex idx{static_cast<size_t>(alloc_proto.parameter_number()),
                        ShapeIndex(alloc_proto.parameter_shape_index().begin(),
                                   alloc_proto.parameter_shape_index().end())};
      auto arg_idx = executable_arg_index.find(idx);
      if (arg_idx == executable_arg_index.end()) continue;
      argument_to_allocation_index[arg_idx->second] =
          static_cast<size_t>(alloc_proto.index());
    }
  }
  return argument_to_allocation_index;
}

absl::StatusOr<std::vector<size_t>> ResolveResultMappingFromProto(
    const HloModuleProto& module_proto,
    const BufferAssignmentProto& buffer_assignment_proto) {
  ABSL_ASSIGN_OR_RETURN(
      ShapeTree<int64_t> result_index_tree,
      CreateResultAllocationIndexTree(module_proto, buffer_assignment_proto));
  std::vector<size_t> result_to_allocation_index;
  for (const auto& [index, alloc_idx] : result_index_tree) {
    if (!result_index_tree.IsLeaf(index)) continue;
    const Shape& subshape =
        ShapeUtil::GetSubshape(result_index_tree.shape(), index);
    if (subshape.IsToken()) continue;
    if (alloc_idx < 0) {
      return Internal("Unresolved result buffer allocation at shape index %s",
                      index.ToString());
    }
    result_to_allocation_index.push_back(static_cast<size_t>(alloc_idx));
  }
  return result_to_allocation_index;
}

absl::StatusOr<std::optional<size_t>> ResolveTempAllocationIndexFromInfos(
    absl::Span<const BufferAllocationInfo> allocation_infos) {
  std::optional<size_t> temp_allocation_index;
  for (size_t i = 0; i < allocation_infos.size(); ++i) {
    if (allocation_infos[i].is_temp()) {
      if (temp_allocation_index.has_value()) {
        return Internal("Multiple temp buffer allocations found");
      }
      temp_allocation_index = i;
    }
  }
  return temp_allocation_index;
}

}  // namespace

const HloModuleConfig& NanoRtExecutable::ExecutableRunner::module_config()
    const {
  LOG(FATAL) << "HloModuleConfig is not available in Lite NanoRtExecutable";
}

void NanoRtExecutable::RegisterAotImporter(
    AotCompilationResultImporter aot_importer) {
  GetAotImporter().store(aot_importer, std::memory_order_release);
}

NanoRtExecutable::ExecuteOptions::ExecuteOptions()
    : intra_op_thread_pool_(nullptr),
      local_device_id_(0),
      global_device_id_(0),
      device_assignment_(nullptr),
      launch_id_(0),
      ffi_context_(nullptr) {}

NanoRtExecutable::ExecuteOptions::~ExecuteOptions() = default;

NanoRtExecutable::ExecuteOptions::ExecuteOptions(ExecuteOptions&&) noexcept =
    default;
NanoRtExecutable::ExecuteOptions& NanoRtExecutable::ExecuteOptions::operator=(
    ExecuteOptions&&) noexcept = default;

NanoRtExecutable::ExecuteOptions&
NanoRtExecutable::ExecuteOptions::set_intra_op_thread_pool(
    const Eigen::ThreadPoolDevice* intra_op_thread_pool) {
  intra_op_thread_pool_ = intra_op_thread_pool;
  return *this;
}

NanoRtExecutable::ExecuteOptions&
NanoRtExecutable::ExecuteOptions::set_ffi_context(
    const ffi::ExecutionContext* ffi_context) {
  ffi_context_ = ffi_context;
  return *this;
}

NanoRtExecutable::ExecuteOptions&
NanoRtExecutable::ExecuteOptions::set_launch_id(int32_t launch_id) {
  launch_id_ = launch_id;
  return *this;
}

NanoRtExecutable::ExecuteOptions&
NanoRtExecutable::ExecuteOptions::set_local_device_id(
    LocalDeviceId local_device_id) {
  local_device_id_ = local_device_id;
  return *this;
}

NanoRtExecutable::ExecuteOptions&
NanoRtExecutable::ExecuteOptions::set_global_device_id(
    GlobalDeviceId global_device_id) {
  global_device_id_ = global_device_id;
  return *this;
}

NanoRtExecutable::ExecuteOptions&
NanoRtExecutable::ExecuteOptions::set_device_assignment(
    DeviceAssignment* device_assignment) {
  device_assignment_ = device_assignment;
  return *this;
}

const Eigen::ThreadPoolDevice*
NanoRtExecutable::ExecuteOptions::intra_op_thread_pool() const {
  return intra_op_thread_pool_;
}

absl::StatusOr<std::unique_ptr<NanoRtExecutable>> NanoRtExecutable::Create(
    CompilationResultProto aot_compilation_result,
    std::optional<ProgramShape> program_shape) {
  if (auto* aot_importer = GetAotImporter().load(std::memory_order_acquire);
      aot_importer != nullptr) {
    return aot_importer(std::move(aot_compilation_result),
                        std::move(program_shape));
  }

  const HloModuleProto& hlo_module_proto =
      aot_compilation_result.hlo_module().hlo_module();
  const BufferAssignmentProto& buffer_assignment_proto =
      aot_compilation_result.buffer_assignment();

  if (!program_shape.has_value()) {
    ABSL_ASSIGN_OR_RETURN(
        program_shape,
        ProgramShape::FromProto(hlo_module_proto.host_program_shape()));
  }

  std::vector<BufferAllocationInfo> allocation_infos =
      CreateBufferAllocationInfos(hlo_module_proto, buffer_assignment_proto);
  ABSL_ASSIGN_OR_RETURN(std::vector<size_t> argument_to_allocation_index,
                        ResolveArgumentsMappingFromProto(
                            *program_shape, buffer_assignment_proto));
  ABSL_ASSIGN_OR_RETURN(
      std::vector<size_t> result_to_allocation_index,
      ResolveResultMappingFromProto(hlo_module_proto, buffer_assignment_proto));
  ABSL_ASSIGN_OR_RETURN(std::optional<size_t> temp_allocation_index,
                        ResolveTempAllocationIndexFromInfos(allocation_infos));

  std::vector<size_t> allocation_sizes(allocation_infos.size());
  for (size_t i = 0; i < allocation_infos.size(); ++i) {
    allocation_sizes[i] = allocation_infos[i].size();
  }

  ABSL_ASSIGN_OR_RETURN(
      std::vector<ConstantAllocation> constants,
      CreateConstantAllocations(buffer_assignment_proto, hlo_module_proto));

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<LiteSequence> sequence,
      BuildLiteSequence(aot_compilation_result.thunk_sequence()));

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<FunctionLibrary> function_library,
      CpuAotLoader::LoadFunctionLibrary(aot_compilation_result));

  auto runner = std::make_unique<LiteExecutableRunner>(
      hlo_module_proto.name(), std::move(function_library),
      std::move(constants), std::move(sequence));

  return std::make_unique<NanoRtExecutable>(
      std::move(runner), std::move(allocation_sizes),
      std::move(argument_to_allocation_index),
      std::move(result_to_allocation_index), temp_allocation_index,
      std::move(program_shape));
}

NanoRtExecutable::NanoRtExecutable(
    std::unique_ptr<ExecutableRunner> runner,
    std::vector<size_t> allocation_sizes,
    std::vector<size_t> argument_to_allocation_index,
    std::vector<size_t> result_to_allocation_index,
    std::optional<size_t> temp_allocation_index,
    std::optional<ProgramShape> program_shape)
    : runner_(std::move(runner)),
      allocation_sizes_(std::move(allocation_sizes)),
      argument_to_allocation_index_(std::move(argument_to_allocation_index)),
      result_to_allocation_index_(std::move(result_to_allocation_index)),
      temp_allocation_index_(temp_allocation_index),
      program_shape_(std::move(program_shape)) {}

NanoRtExecutable::~NanoRtExecutable() = default;

static se::DeviceAddressBase ToDeviceMemory(
    const NanoRtExecutable::Argument& argument) {
  return se::DeviceAddressBase(
      const_cast<void*>(reinterpret_cast<const void*>(argument.data().data())),
      argument.data().size());
}

static se::DeviceAddressBase ToDeviceMemory(
    const NanoRtExecutable::Result& result) {
  return se::DeviceAddressBase(reinterpret_cast<void*>(result.data().data()),
                               result.data().size());
}

static se::DeviceAddressBase ToDeviceMemory(
    const NanoRtExecutable::PreallocatedTemp& temp) {
  return se::DeviceAddressBase(reinterpret_cast<void*>(temp.data()),
                               temp.size());
}

tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> NanoRtExecutable::Execute(
    absl::Span<const Argument> arguments, absl::Span<const Result> results,
    PreallocatedTemp temp, const ExecuteOptions& options) {
  size_t num_arguments = argument_to_allocation_index_.size();
  size_t num_results = result_to_allocation_index_.size();

  if (ABSL_PREDICT_FALSE(arguments.size() != num_arguments)) {
    return InvalidArgument("Expected %d arguments, got %d", num_arguments,
                           arguments.size());
  }

  if (ABSL_PREDICT_FALSE(results.size() != num_results)) {
    return InvalidArgument("Expected %d results, got %d", num_results,
                           results.size());
  }

  std::vector<se::DeviceAddressBase> buffers(allocation_sizes_.size());

  for (size_t i = 0; i < num_arguments; ++i) {
    size_t idx = argument_to_allocation_index_[i];
    buffers[idx] = ToDeviceMemory(arguments[i]);

    if (ABSL_PREDICT_FALSE(buffers[idx].size() != allocation_sizes_[idx])) {
      return InvalidArgument("Argument %d size mismatch: expected %d, got %d",
                             i, allocation_sizes_[idx], buffers[idx].size());
    }
  }

  for (size_t i = 0; i < num_results; ++i) {
    size_t idx = result_to_allocation_index_[i];
    buffers[idx] = ToDeviceMemory(results[i]);

    if (ABSL_PREDICT_FALSE(buffers[idx].size() != allocation_sizes_[idx])) {
      return InvalidArgument("Result %d size mismatch: expected %d, got %d", i,
                             allocation_sizes_[idx], buffers[idx].size());
    }
  }

  if (temp_allocation_index_) {
    size_t idx = *temp_allocation_index_;
    buffers[idx] = ToDeviceMemory(temp);

    if (ABSL_PREDICT_FALSE(buffers[idx].size() != allocation_sizes_[idx])) {
      return InvalidArgument("Temp size mismatch: expected %d, got %d",
                             allocation_sizes_[idx], buffers[idx].size());
    }
  }

  return runner_->Execute(std::move(buffers), options);
}

size_t NanoRtExecutable::temp_buffer_size() const {
  if (temp_allocation_index_.has_value()) {
    return allocation_sizes_[*temp_allocation_index_];
  }
  return 0;
}

}  // namespace xla::cpu
