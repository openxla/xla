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

#include "xla/pjrt/common_pjrt_client.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/infer_dispatch_info.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/runtime/device_id.h"
#include "xla/service/device_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

TEST(TransferIncarnationsTest, EmptySnapshotReturnsNullopt) {
  ASSERT_OK_AND_ASSIGN(auto transfer_incarnations,
                       TransferIncarnations(/*src_process_index=*/1,
                                            /*dst_process_index=*/2, {}));
  EXPECT_FALSE(transfer_incarnations.has_value());
}

TEST(TransferIncarnationsTest, SnapshotKeepsSourceThenDestination) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(2), IncarnationId(20)},
      {TaskId(1), IncarnationId(10)},
  };

  ASSERT_OK_AND_ASSIGN(
      auto forward,
      TransferIncarnations(/*src_process_index=*/2, /*dst_process_index=*/1,
                           incarnations));
  ASSERT_TRUE(forward.has_value());
  EXPECT_THAT(*forward, ElementsAre(IncarnationId(20), IncarnationId(10)));

  ASSERT_OK_AND_ASSIGN(
      auto swapped,
      TransferIncarnations(/*src_process_index=*/1, /*dst_process_index=*/2,
                           incarnations));
  ASSERT_TRUE(swapped.has_value());
  EXPECT_THAT(*swapped, ElementsAre(IncarnationId(10), IncarnationId(20)));
}

TEST(TransferIncarnationsTest, SameTaskKeepsBothIncarnations) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(3), IncarnationId(7)},
  };

  ASSERT_OK_AND_ASSIGN(
      auto transfer_incarnations,
      TransferIncarnations(/*src_process_index=*/3, /*dst_process_index=*/3,
                           incarnations));
  ASSERT_TRUE(transfer_incarnations.has_value());
  EXPECT_THAT(*transfer_incarnations,
              ElementsAre(IncarnationId(7), IncarnationId(7)));
}

TEST(TransferIncarnationsTest, MissingTaskFails) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(1), IncarnationId(10)},
  };

  EXPECT_THAT(TransferIncarnations(/*src_process_index=*/1,
                                   /*dst_process_index=*/2, incarnations),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

std::vector<Shape> MakeCanonicalDynamicParameterShapes(
    const HloModule& module) {
  std::vector<Shape> shapes =
      GetParameterShapes(module.entry_computation_layout());
  for (Shape& shape : shapes) {
    ShapeUtil::ForEachMutableSubshape(
        &shape, [](Shape* subshape, const ShapeIndex&) {
          if (subshape->IsArray() && subshape->is_dynamic() &&
              subshape->has_layout()) {
            subshape->mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(
                1024);
          }
        });
  }
  return shapes;
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsPadRealToStaticAndPreservesPadToStatic) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule pad_real_vs_pad_std

ENTRY main {
  p0 = bf16[<=16,32]{1,0} parameter(0)
  p1 = f32[<=16,32]{1,0} parameter(1)
  p2 = s32[<=16,32]{1,0} parameter(2)
  d0 = s32[] parameter(3)
  d1 = s32[] parameter(4)
  tok = token[] after-all()
  bitcast_p2 = s32[<=16,32]{1,0} bitcast(p2)
  barrier_p2 = s32[<=16,32]{1,0} opt-barrier(bitcast_p2)
  dep_p2 = s32[<=16,32]{1,0} add-dependency(barrier_p2, tok)
  pad_real_0 = bf16[16,32]{1,0} custom-call(p0, d0, d1),
      custom_call_target="PadRealToStatic"
  pad_std_1 = (f32[16,32]{1,0}, s32[], s32[]) custom-call(p1),
      custom_call_target="PadToStatic"
  pad_real_2 = s32[16,32]{1,0} custom-call(dep_p2, d0, d1),
      custom_call_target="PadRealToStatic"
  gte_1 = f32[16,32]{1,0} get-tuple-element(pad_std_1), index=0
  ROOT out = (bf16[16,32]{1,0}, f32[16,32]{1,0}, s32[16,32]{1,0})
      tuple(pad_real_0, gte_1, pad_real_2)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
  EXPECT_EQ(param_shapes[2].layout().dynamic_shape_metadata_prefix_bytes(), 0);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsWideParametersReadByPlaneCustomCalls) {
  // After the TPU X64/complex rewriters, 64-bit and 128-bit `PadRealToStatic`
  // inputs are read directly by one 32-bit `PadRealToStatic` per plane.
  constexpr absl::string_view kHlo = R"hlo(
HloModule wide_pad_real_to_static

ENTRY main {
  p0_s64 = s64[<=16,32]{1,0} parameter(0)
  p1_f64 = f64[<=16,32]{1,0} parameter(1)
  p2_c128 = c128[<=16,32]{1,0} parameter(2)
  p3_s64_std = s64[<=16,32]{1,0} parameter(3)
  d0 = s32[] parameter(4)
  d1 = s32[] parameter(5)

  pad_s64_lo = u32[16,32]{1,0} custom-call(p0_s64, d0, d1),
      custom_call_target="PadRealToStatic",
      backend_config={"pad_real_to_static_plane_index":0}
  pad_s64_hi = u32[16,32]{1,0} custom-call(p0_s64, d0, d1),
      custom_call_target="PadRealToStatic",
      backend_config={"pad_real_to_static_plane_index":1}
  pad_f64_lo = f32[16,32]{1,0} custom-call(p1_f64, d0, d1),
      custom_call_target="PadRealToStatic",
      backend_config={"pad_real_to_static_plane_index":1}
  pad_c128 = f32[16,32]{1,0} custom-call(p2_c128, d0, d1),
      custom_call_target="PadRealToStatic",
      backend_config={"pad_real_to_static_plane_index":3}

  std_lo = u32[<=16,32]{1,0} custom-call(p3_s64_std),
      custom_call_target="X64SplitLow"
  pad_std = (u32[16,32]{1,0}, s32[], s32[]) custom-call(std_lo),
      custom_call_target="PadToStatic"
  pad_std_data = u32[16,32]{1,0} get-tuple-element(pad_std), index=0

  ROOT out = (u32[16,32]{1,0}, u32[16,32]{1,0}, f32[16,32]{1,0},
              f32[16,32]{1,0}, u32[16,32]{1,0})
      tuple(pad_s64_lo, pad_s64_hi, pad_f64_lo, pad_c128, pad_std_data)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[2].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[3].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsThroughSparseCoreOffloadAsyncStartAndAsyncUpdate) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule sparsecore_offload_pad_real_to_static

%leaf_lo (leaf_p0: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> u32[16,32] {
  %leaf_p0 = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p0, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic",
      backend_config={"pad_real_to_static_plane_index":0}
}, execution_thread="sparsecore"

%leaf_hi (leaf_p0: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> u32[16,32] {
  %leaf_p0 = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p0, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic",
      backend_config={"pad_real_to_static_plane_index":1}
}, execution_thread="sparsecore"

%outer_async_lo (out_p0: s64[<=16,32], out_d0: s32[], out_d1: s32[]) -> u32[16,32] {
  %out_p0 = s64[<=16,32]{1,0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %out_d1 = s32[] parameter(2)
  ROOT %call = u32[16,32]{1,0} call(%out_p0, %out_d0, %out_d1),
      to_apply=%leaf_lo
}, execution_thread="sparsecore"

%outer_async_hi (out_p0: s64[<=16,32], out_d0: s32[], out_d1: s32[]) -> u32[16,32] {
  %out_p0 = s64[<=16,32]{1,0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %out_d1 = s32[] parameter(2)
  ROOT %call = u32[16,32]{1,0} call(%out_p0, %out_d0, %out_d1),
      to_apply=%leaf_hi
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], p1: s64[<=16,32], d0: s32[], d1: s32[]) -> (u32[16,32], u32[16,32], u32[16,32]) {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %p1 = s64[<=16,32]{1,0} parameter(1)
  %d0 = s32[] parameter(2)
  %d1 = s32[] parameter(3)
  %start_lo = ((), (), u32[]) async-start(),
      async_execution_thread="sparsecore", calls=%outer_async_lo
  %update_lo = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0}, u32[])
      async-update(%start_lo, %p0, %d0, %d1),
      async_execution_thread="sparsecore"
  %done_lo = u32[16,32]{1,0} async-done(%update_lo),
      async_execution_thread="sparsecore"

  %start_hi = ((), (), u32[]) async-start(),
      async_execution_thread="sparsecore", calls=%outer_async_hi
  %update_hi = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0}, u32[])
      async-update(%start_hi, %p0, %d0, %d1),
      async_execution_thread="sparsecore"
  %done_hi = u32[16,32]{1,0} async-done(%update_hi),
      async_execution_thread="sparsecore"

  %std_lo = u32[<=16,32]{1,0} custom-call(%p1),
      custom_call_target="X64SplitLow"
  %pad_std = (u32[16,32]{1,0}, s32[], s32[]) custom-call(%std_lo),
      custom_call_target="PadToStatic"
  %std_data = u32[16,32]{1,0} get-tuple-element(%pad_std), index=0

  ROOT %out = (u32[16,32]{1,0}, u32[16,32]{1,0}, u32[16,32]{1,0})
      tuple(%done_lo, %done_hi, %std_data)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsTupledEntryParameterElement) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule tupled_entry_pad_real_to_static

ENTRY main {
  arg_tuple = (s64[<=16,32]{1,0}, f32[<=16,32]{1,0}, s64[<=16,32]{1,0},
               s32[], s32[]) parameter(0)
  arg_copy = (s64[<=16,32]{1,0}, f32[<=16,32]{1,0}, s64[<=16,32]{1,0},
              s32[], s32[]) copy(arg_tuple)
  p0 = s64[<=16,32]{1,0} get-tuple-element(arg_copy), index=0
  p1 = f32[<=16,32]{1,0} get-tuple-element(arg_copy), index=1
  p2 = s64[<=16,32]{1,0} get-tuple-element(arg_copy), index=2
  d0 = s32[] get-tuple-element(arg_copy), index=3
  d1 = s32[] get-tuple-element(arg_copy), index=4
  nested = (s64[<=16,32]{1,0}, s32[]) tuple(p2, d0)
  nested_p2 = s64[<=16,32]{1,0} get-tuple-element(nested), index=0
  pad_real = u32[16,32]{1,0} custom-call(p0, d0, d1),
      custom_call_target="PadRealToStatic"
  pad_nested = u32[16,32]{1,0} custom-call(nested_p2, d0, d1),
      custom_call_target="PadRealToStatic"
  pad_std = (f32[16,32]{1,0}, s32[], s32[]) custom-call(p1),
      custom_call_target="PadToStatic"
  gte = f32[16,32]{1,0} get-tuple-element(pad_std), index=0
  ROOT out = (u32[16,32]{1,0}, u32[16,32]{1,0}, f32[16,32]{1,0})
      tuple(pad_real, pad_nested, gte)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_EQ(param_shapes.size(), 5);
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
  EXPECT_EQ(param_shapes[2].layout().dynamic_shape_metadata_prefix_bytes(), 0);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsThroughNestedTuplesAndSingleElementTupleParameter) {
  // A single 1-level tuple entry parameter is flattened by `GetParameterShapes`
  // into one non-tuple shape per top-level element.
  constexpr absl::string_view kHlo = R"hlo(
HloModule nested_tuple_pad_real_to_static

ENTRY main {
  arg = (s64[<=16,32]{1,0}) parameter(0)
  p = s64[<=16,32]{1,0} get-tuple-element(arg), index=0
  d0 = s32[] constant(8)
  d1 = s32[] constant(32)
  fwd = (s64[<=16,32]{1,0}, s32[]) tuple(p, d0)
  fwd2 = ((s64[<=16,32]{1,0}, s32[]), s32[]) tuple(fwd, d1)
  fwd_inner = (s64[<=16,32]{1,0}, s32[]) get-tuple-element(fwd2), index=0
  p_fwd = s64[<=16,32]{1,0} get-tuple-element(fwd_inner), index=0
  ROOT pad = u32[16,32]{1,0} custom-call(p_fwd, d0, d1),
      custom_call_target="PadRealToStatic"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_EQ(param_shapes.size(), 1);
  ASSERT_FALSE(param_shapes[0].IsTuple());
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsParameterWithOtherUsersWithoutPartialMutation) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule shared_pad_real_to_static_parameter

ENTRY main {
  p_valid = bf16[<=16,32]{1,0} parameter(0)
  p_shared = s64[<=16,32]{1,0} parameter(1)
  d0 = s32[] parameter(2)
  d1 = s32[] parameter(3)
  pad_valid = bf16[16,32]{1,0} custom-call(p_valid, d0, d1),
      custom_call_target="PadRealToStatic"
  pad_shared = u32[16,32]{1,0} custom-call(p_shared, d0, d1),
      custom_call_target="PadRealToStatic"
  dyn_sum = s64[<=16,32]{1,0} add(p_shared, p_shared)
  ROOT out = (bf16[16,32]{1,0}, u32[16,32]{1,0}, s64[<=16,32]{1,0})
      tuple(pad_valid, pad_shared, dyn_sum)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *module, absl::MakeSpan(param_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 1"),
                     HasSubstr("PadRealToStatic"), HasSubstr("dyn_sum"))));
  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsTupleParameterElementWithOtherUsers) {
  // The same tuple element read through a second get-tuple-element.
  constexpr absl::string_view kHlo = R"hlo(
HloModule shared_tuple_pad_real_to_static_parameter

ENTRY main {
  arg = (f32[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  p = f32[<=16,32]{1,0} get-tuple-element(arg), index=0
  p_again = f32[<=16,32]{1,0} get-tuple-element(arg), index=0
  d0 = s32[] get-tuple-element(arg), index=1
  d1 = s32[] get-tuple-element(arg), index=2
  pad = f32[16,32]{1,0} custom-call(p, d0, d1),
      custom_call_target="PadRealToStatic"
  neg = f32[<=16,32]{1,0} negate(p_again)
  ROOT out = (f32[16,32]{1,0}, f32[<=16,32]{1,0}) tuple(pad, neg)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *module, absl::MakeSpan(param_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0 {0}"), HasSubstr("neg"))));
  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsAliasedPadRealToStaticParameter) {
  constexpr absl::string_view kDirectAliasHlo = R"hlo(
HloModule aliased_pad_real_to_static,
    input_output_alias={ {}: (0, {}, may-alias) }

ENTRY main {
  p0 = s64[<=16,32]{1,0} parameter(0)
  d0 = s32[] parameter(1)
  d1 = s32[] parameter(2)
  ROOT pad = s64[16,32]{1,0} custom-call(p0, d0, d1),
      custom_call_target="PadRealToStatic"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> direct_module,
                       ParseAndReturnUnverifiedModule(kDirectAliasHlo));
  std::vector<Shape> direct_shapes =
      MakeCanonicalDynamicParameterShapes(*direct_module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *direct_module, absl::MakeSpan(direct_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0"),
                     HasSubstr("PadRealToStatic"), HasSubstr("aliased"))));
  EXPECT_EQ(direct_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);

  constexpr absl::string_view kTupleLeafAliasHlo = R"hlo(
HloModule aliased_tuple_pad_real_to_static,
    input_output_alias={ {0}: (0, {0}, must-alias) }

ENTRY main {
  arg = (f32[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  p = f32[<=16,32]{1,0} get-tuple-element(arg), index=0
  d0 = s32[] get-tuple-element(arg), index=1
  d1 = s32[] get-tuple-element(arg), index=2
  pad = f32[16,32]{1,0} custom-call(p, d0, d1),
      custom_call_target="PadRealToStatic"
  ROOT out = (f32[16,32]{1,0}, s32[]) tuple(pad, d0)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> tuple_module,
                       ParseAndReturnUnverifiedModule(kTupleLeafAliasHlo));
  std::vector<Shape> tuple_shapes =
      MakeCanonicalDynamicParameterShapes(*tuple_module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *tuple_module, absl::MakeSpan(tuple_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0 {0}"),
                     HasSubstr("PadRealToStatic"), HasSubstr("aliased"))));
  EXPECT_EQ(tuple_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     AllowsStaticAliasedPadRealToStaticParameter) {
  constexpr absl::string_view kStaticAliasHlo = R"hlo(
HloModule static_aliased_pad_real_to_static,
    input_output_alias={ {}: (0, {}, may-alias) }

ENTRY main {
  p0 = s32[16,32]{1,0} parameter(0)
  d0 = s32[] parameter(1)
  d1 = s32[] parameter(2)
  ROOT pad = s32[16,32]{1,0} custom-call(p0, d0, d1),
      custom_call_target="PadRealToStatic"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kStaticAliasHlo));
  std::vector<Shape> shapes = MakeCanonicalDynamicParameterShapes(*module);
  EXPECT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(shapes)));
  EXPECT_EQ(shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsDynamicArrayCopyBeforePadRealToStatic) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule copy_before_pad_real_to_static

ENTRY main {
  p0 = s64[<=16,32]{1,0} parameter(0)
  d0 = s32[] parameter(1)
  d1 = s32[] parameter(2)
  copy_p0 = s64[<=16,32]{1,0} copy(p0)
  ROOT pad = u32[16,32]{1,0} custom-call(copy_p0, d0, d1),
      custom_call_target="PadRealToStatic"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  EXPECT_THAT(ClearPadRealToStaticParameterMetadataPrefixes(
                  *module, absl::MakeSpan(param_shapes)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("PadRealToStatic"),
                             HasSubstr("entry computation parameter"),
                             HasSubstr("copy_p0"))));
  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsNonParameterDynamicOperand) {
  constexpr absl::string_view kSliceToDynamicHlo = R"hlo(
HloModule slice_to_dynamic_into_pad_real_to_static

ENTRY main {
  p0 = s32[16,32]{1,0} parameter(0)
  d0 = s32[] parameter(1)
  d1 = s32[] parameter(2)
  slice_dyn = s32[<=16,32]{1,0} custom-call(p0, d0, d1),
      custom_call_target="SliceToDynamic"
  ROOT pad = s32[16,32]{1,0} custom-call(slice_dyn, d0, d1),
      custom_call_target="PadRealToStatic"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> slice_module,
                       ParseAndReturnUnverifiedModule(kSliceToDynamicHlo));
  std::vector<Shape> slice_shapes =
      MakeCanonicalDynamicParameterShapes(*slice_module);
  EXPECT_THAT(ClearPadRealToStaticParameterMetadataPrefixes(
                  *slice_module, absl::MakeSpan(slice_shapes)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("PadRealToStatic"),
                             HasSubstr("entry computation parameter"),
                             HasSubstr("slice_dyn"))));

  constexpr absl::string_view kAllGatherHlo = R"hlo(
HloModule all_gather_into_pad_real_to_static

ENTRY main {
  p0 = f32[<=8,32]{1,0} parameter(0)
  d0 = s32[] parameter(1)
  d1 = s32[] parameter(2)
  ag = f32[<=16,32]{1,0} all-gather(p0), dimensions={0}, replica_groups={{0,1}}
  ROOT pad = f32[16,32]{1,0} custom-call(ag, d0, d1),
      custom_call_target="PadRealToStatic"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> ag_module,
                       ParseAndReturnUnverifiedModule(kAllGatherHlo));
  std::vector<Shape> ag_shapes =
      MakeCanonicalDynamicParameterShapes(*ag_module);
  EXPECT_THAT(ClearPadRealToStaticParameterMetadataPrefixes(
                  *ag_module, absl::MakeSpan(ag_shapes)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("PadRealToStatic"),
                             HasSubstr("entry computation parameter"),
                             HasSubstr("ag"))));
  EXPECT_EQ(ag_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsCustomCallWithCalledComputationSharingParameter) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule embedded_custom_call_sharing_parameter

%unused_param_comp (sub_p0: s64[<=16,32]) -> s32[] {
  %sub_p0 = s64[<=16,32]{1,0} parameter(0)
  ROOT %c = s32[] constant(0)
}

ENTRY main {
  p0 = s64[<=16,32]{1,0} parameter(0)
  d0 = s32[] parameter(1)
  d1 = s32[] parameter(2)
  pad = u32[16,32]{1,0} custom-call(p0, d0, d1),
      custom_call_target="PadRealToStatic"
  other_cc = s32[] custom-call(p0),
      custom_call_target="OtherTarget",
      called_computations={%unused_param_comp}
  ROOT out = (u32[16,32]{1,0}, s32[]) tuple(pad, other_cc)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *module, absl::MakeSpan(param_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0"), HasSubstr("other_cc"))));
  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsOffloadedSubcomputationReturningParameter) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule offloaded_subcomputation_escaping_parameter

%leaf_comp (leaf_p0: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> (u32[16,32], s64[<=16,32]) {
  %leaf_p0 = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  %pad = u32[16,32]{1,0} custom-call(%leaf_p0, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
  ROOT %escaped = (u32[16,32]{1,0}, s64[<=16,32]{1,0}) tuple(%pad, %leaf_p0)
}

ENTRY main {
  p0 = s64[<=16,32]{1,0} parameter(0)
  d0 = s32[] parameter(1)
  d1 = s32[] parameter(2)
  ROOT res = (u32[16,32]{1,0}, s64[<=16,32]{1,0}) call(p0, d0, d1),
      to_apply=%leaf_comp
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *module, absl::MakeSpan(param_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0"),
                     HasSubstr("returned as a result of computation"))));
  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsEmbeddedAsyncWrappedSubcomputationCall) {
  // Post-compile SparseCore offloads embed the async wrapper's root call as a
  // `custom-call` with `to_apply`, optionally deduplicating `to_apply` across
  // `async-start` sites or packing operands into a tuple.
  constexpr absl::string_view kHlo = R"hlo(
HloModule embedded_sparsecore_async_pad_real_to_static

%sc_plane_lo (leaf_p0: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> u32[16,32] {
  %leaf_p0 = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p0, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
}

%sc_plane_hi_tupled (leaf_tup: (s64[<=16,32], s32[], s32[])) -> u32[16,32] {
  %leaf_tup = (s64[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  %leaf_p0 = s64[<=16,32]{1,0} get-tuple-element(%leaf_tup), index=0
  %leaf_d0 = s32[] get-tuple-element(%leaf_tup), index=1
  %leaf_d1 = s32[] get-tuple-element(%leaf_tup), index=2
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p0, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
}

%async_wrap_p0_lo (out_p0: s64[<=16,32], out_d0: s32[], out_d1: s32[]) -> u32[16,32] {
  %out_p0 = s64[<=16,32]{1,0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %out_d1 = s32[] parameter(2)
  ROOT %sc_call = u32[16,32]{1,0} custom-call(%out_p0, %out_d0, %out_d1),
      custom_call_target="AsyncOffloadCall", to_apply=%sc_plane_lo
}, execution_thread="sparsecore"

%async_wrap_p0_hi (out_p0: s64[<=16,32], out_d0: s32[], out_d1: s32[]) -> u32[16,32] {
  %out_p0 = s64[<=16,32]{1,0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %out_d1 = s32[] parameter(2)
  %packed = (s64[<=16,32]{1,0}, s32[], s32[]) tuple(%out_p0, %out_d0, %out_d1)
  ROOT %sc_call = u32[16,32]{1,0} custom-call(%packed),
      custom_call_target="AsyncOffloadCall", to_apply=%sc_plane_hi_tupled
}, execution_thread="sparsecore"

%async_wrap_p1_lo (out_p1: s64[<=16,32], out_d0: s32[], out_d1: s32[]) -> u32[16,32] {
  %out_p1 = s64[<=16,32]{1,0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %out_d1 = s32[] parameter(2)
  ROOT %sc_call = u32[16,32]{1,0} custom-call(%out_p1, %out_d0, %out_d1),
      custom_call_target="AsyncOffloadCall", to_apply=%sc_plane_lo
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], p1: s64[<=16,32], p2: f32[<=16,32], d0: s32[], d1: s32[]) -> (u32[16,32], u32[16,32], u32[16,32], f32[16,32]) {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %p1 = s64[<=16,32]{1,0} parameter(1)
  %p2 = f32[<=16,32]{1,0} parameter(2)
  %d0 = s32[] parameter(3)
  %d1 = s32[] parameter(4)

  %start_p0_lo = ((), ()) async-start(),
      async_execution_thread="sparsecore", calls=%async_wrap_p0_lo
  %update_p0_lo = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-update(%start_p0_lo, %p0, %d0, %d1),
      async_execution_thread="sparsecore"
  %done_p0_lo = u32[16,32]{1,0} async-done(%update_p0_lo),
      async_execution_thread="sparsecore"

  %start_p0_hi = ((), ()) async-start(),
      async_execution_thread="sparsecore", calls=%async_wrap_p0_hi
  %update_p0_hi = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-update(%start_p0_hi, %p0, %d0, %d1),
      async_execution_thread="sparsecore"
  %done_p0_hi = u32[16,32]{1,0} async-done(%update_p0_hi),
      async_execution_thread="sparsecore"

  %start_p1_lo = ((), ()) async-start(),
      async_execution_thread="sparsecore", calls=%async_wrap_p1_lo
  %update_p1_lo = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-update(%start_p1_lo, %p1, %d0, %d1),
      async_execution_thread="sparsecore"
  %done_p1_lo = u32[16,32]{1,0} async-done(%update_p1_lo),
      async_execution_thread="sparsecore"

  %pad_std = (f32[16,32]{1,0}, s32[], s32[]) custom-call(%p2),
      custom_call_target="PadToStatic"
  %std_data = f32[16,32]{1,0} get-tuple-element(%pad_std), index=0

  ROOT %out = (u32[16,32]{1,0}, u32[16,32]{1,0}, u32[16,32]{1,0}, f32[16,32]{1,0})
      tuple(%done_p0_lo, %done_p0_hi, %done_p1_lo, %std_data)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[2].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsEmbeddedAsyncWrappedSubcomputationMisuse) {
  constexpr absl::string_view kEscapedInEmbeddedHlo = R"hlo(
HloModule embedded_sparsecore_escaping_parameter

%sc_escape (leaf_p0: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> (u32[16,32], s64[<=16,32]) {
  %leaf_p0 = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  %pad = u32[16,32]{1,0} custom-call(%leaf_p0, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
  ROOT %escaped = (u32[16,32]{1,0}, s64[<=16,32]{1,0}) tuple(%pad, %leaf_p0)
}

%async_wrap (out_p0: s64[<=16,32], out_d0: s32[], out_d1: s32[]) -> (u32[16,32], s64[<=16,32]) {
  %out_p0 = s64[<=16,32]{1,0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %out_d1 = s32[] parameter(2)
  ROOT %sc_call = (u32[16,32]{1,0}, s64[<=16,32]{1,0})
      custom-call(%out_p0, %out_d0, %out_d1),
      custom_call_target="AsyncOffloadCall", to_apply=%sc_escape
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], d0: s32[], d1: s32[]) -> (u32[16,32], s64[<=16,32]) {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %d0 = s32[] parameter(1)
  %d1 = s32[] parameter(2)
  %start = ((s64[<=16,32]{1,0}, s32[], s32[]), (u32[16,32]{1,0}, s64[<=16,32]{1,0}))
      async-start(%p0, %d0, %d1),
      async_execution_thread="sparsecore", calls=%async_wrap
  ROOT %done = (u32[16,32]{1,0}, s64[<=16,32]{1,0}) async-done(%start),
      async_execution_thread="sparsecore"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> escaped_module,
                       ParseAndReturnUnverifiedModule(kEscapedInEmbeddedHlo));
  std::vector<Shape> escaped_shapes =
      MakeCanonicalDynamicParameterShapes(*escaped_module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *escaped_module, absl::MakeSpan(escaped_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0"),
                     HasSubstr("returned as a result of computation"))));
  EXPECT_EQ(escaped_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);

  constexpr absl::string_view kSharedNonPadAsyncHlo = R"hlo(
HloModule embedded_sparsecore_shared_with_non_pad_offload

%sc_pad (leaf_p0: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> u32[16,32] {
  %leaf_p0 = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p0, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
}

%sc_other (other_p0: s64[<=16,32]) -> s32[] {
  %other_p0 = s64[<=16,32]{1,0} parameter(0)
  ROOT %zero = s32[] constant(0)
}

%async_wrap_pad (out_p0: s64[<=16,32], out_d0: s32[], out_d1: s32[]) -> u32[16,32] {
  %out_p0 = s64[<=16,32]{1,0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %out_d1 = s32[] parameter(2)
  ROOT %sc_call = u32[16,32]{1,0} custom-call(%out_p0, %out_d0, %out_d1),
      custom_call_target="AsyncOffloadCall", to_apply=%sc_pad
}, execution_thread="sparsecore"

%async_wrap_other (out_p0: s64[<=16,32]) -> s32[] {
  %out_p0 = s64[<=16,32]{1,0} parameter(0)
  ROOT %sc_other_call = s32[] custom-call(%out_p0),
      custom_call_target="AsyncOffloadCall", to_apply=%sc_other
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], d0: s32[], d1: s32[]) -> (u32[16,32], s32[]) {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %d0 = s32[] parameter(1)
  %d1 = s32[] parameter(2)
  %start_pad = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-start(%p0, %d0, %d1),
      async_execution_thread="sparsecore", calls=%async_wrap_pad
  %done_pad = u32[16,32]{1,0} async-done(%start_pad),
      async_execution_thread="sparsecore"
  %start_other = ((s64[<=16,32]{1,0}), s32[]) async-start(%p0),
      async_execution_thread="sparsecore", calls=%async_wrap_other
  %done_other = s32[] async-done(%start_other),
      async_execution_thread="sparsecore"
  ROOT %out = (u32[16,32]{1,0}, s32[]) tuple(%done_pad, %done_other)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> shared_module,
                       ParseAndReturnUnverifiedModule(kSharedNonPadAsyncHlo));
  std::vector<Shape> shared_shapes =
      MakeCanonicalDynamicParameterShapes(*shared_module);
  EXPECT_THAT(ClearPadRealToStaticParameterMetadataPrefixes(
                  *shared_module, absl::MakeSpan(shared_shapes)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("Entry parameter 0"),
                             HasSubstr("sc_other_call"))));
  EXPECT_EQ(shared_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);

  constexpr absl::string_view kAliasedEmbeddedCallHlo = R"hlo(
HloModule embedded_sparsecore_aliased_operand

%sc_pad (leaf_p0: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> u32[16,32] {
  %leaf_p0 = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p0, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
}

%async_wrap_alias (out_p0: s64[<=16,32], out_d0: s32[], out_d1: s32[]) -> u32[16,32] {
  %out_p0 = s64[<=16,32]{1,0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %out_d1 = s32[] parameter(2)
  ROOT %sc_aliased = u32[16,32]{1,0} custom-call(%out_p0, %out_d0, %out_d1),
      custom_call_target="AsyncOffloadCall",
      output_to_operand_aliasing={{}: (0, {})},
      to_apply=%sc_pad
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], d0: s32[], d1: s32[]) -> u32[16,32] {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %d0 = s32[] parameter(1)
  %d1 = s32[] parameter(2)
  %start = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-start(%p0, %d0, %d1),
      async_execution_thread="sparsecore", calls=%async_wrap_alias
  ROOT %done = u32[16,32]{1,0} async-done(%start),
      async_execution_thread="sparsecore"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> aliased_module,
                       ParseAndReturnUnverifiedModule(kAliasedEmbeddedCallHlo));
  std::vector<Shape> aliased_shapes =
      MakeCanonicalDynamicParameterShapes(*aliased_module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *aliased_module, absl::MakeSpan(aliased_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0"), HasSubstr("sc_aliased"))));
  EXPECT_EQ(aliased_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsPrefixAcrossPrefetchedDeduplicatedAndNestedAsyncTileOffloads) {
  constexpr absl::string_view kFiveLayerDeduplicatedHlo = R"hlo(
HloModule prefetched_deduplicated_nested_async_tile_offload

%inner_tile_callee (leaf_p: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> u32[16,32] {
  %leaf_p = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
}

%inner_tile_async_wrap (tile_p: s64[<=16,32], tile_d0: s32[], tile_d1: s32[]) -> u32[16,32] {
  %tile_p = s64[<=16,32]{1,0} parameter(0)
  %tile_d0 = s32[] parameter(1)
  %tile_d1 = s32[] parameter(2)
  ROOT %inner_call = u32[16,32]{1,0} call(%tile_p, %tile_d0, %tile_d1),
      to_apply=%inner_tile_callee
}, execution_thread="sparsecore"

%reconstructed_scs_comp (scs_tuple_param: (s64[<=16,32], s32[], s32[])) -> u32[16,32] {
  %scs_tuple_param = (s64[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  %scs_p = s64[<=16,32]{1,0} get-tuple-element(%scs_tuple_param), index=0
  %scs_d0 = s32[] get-tuple-element(%scs_tuple_param), index=1
  %scs_d1 = s32[] get-tuple-element(%scs_tuple_param), index=2
  %tile_start = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-start(%scs_p, %scs_d0, %scs_d1),
      async_execution_thread="sparsecore", calls=%inner_tile_async_wrap
  ROOT %tile_done = u32[16,32]{1,0} async-done(%tile_start),
      async_execution_thread="sparsecore"
}

%outer_async_wrap_0 (outer_tuple_0: (s64[<=16,32], s32[], s32[])) -> u32[16,32] {
  %outer_tuple_0 = (s64[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  ROOT %sc_call_0 = u32[16,32]{1,0} custom-call(%outer_tuple_0),
      custom_call_target="AsyncOffloadCall", to_apply=%reconstructed_scs_comp
}, execution_thread="sparsecore"

%outer_async_wrap_1 (outer_tuple_1: (s64[<=16,32], s32[], s32[])) -> u32[16,32] {
  %outer_tuple_1 = (s64[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  ROOT %sc_call_1 = u32[16,32]{1,0} custom-call(%outer_tuple_1),
      custom_call_target="AsyncOffloadCall", to_apply=%reconstructed_scs_comp
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], p1: s64[<=16,32], p2: f32[<=16,32], d0: s32[], d1: s32[]) -> (u32[16,32], u32[16,32], f32[16,32]) {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %p1 = s64[<=16,32]{1,0} parameter(1)
  %p2 = f32[<=16,32]{1,0} parameter(2)
  %d0 = s32[] parameter(3)
  %d1 = s32[] parameter(4)
  %tuple0 = (s64[<=16,32]{1,0}, s32[], s32[]) tuple(%p0, %d0, %d1)
  %start0 = ((), ()) async-start(),
      async_execution_thread="sparsecore", calls=%outer_async_wrap_0
  %update0 = (((s64[<=16,32]{1,0}, s32[], s32[])), u32[16,32]{1,0})
      async-update(%start0, %tuple0), async_execution_thread="sparsecore"
  %done0 = u32[16,32]{1,0} async-done(%update0),
      async_execution_thread="sparsecore"
  %tuple1 = (s64[<=16,32]{1,0}, s32[], s32[]) tuple(%p1, %d0, %d1)
  %start1 = ((), ()) async-start(),
      async_execution_thread="sparsecore", calls=%outer_async_wrap_1
  %update1 = (((s64[<=16,32]{1,0}, s32[], s32[])), u32[16,32]{1,0})
      async-update(%start1, %tuple1), async_execution_thread="sparsecore"
  %done1 = u32[16,32]{1,0} async-done(%update1),
      async_execution_thread="sparsecore"
  %pad_std = (f32[16,32]{1,0}, s32[], s32[]) custom-call(%p2),
      custom_call_target="PadToStatic"
  %pad_std_data = f32[16,32]{1,0} get-tuple-element(%pad_std), index=0
  ROOT %out = (u32[16,32]{1,0}, u32[16,32]{1,0}, f32[16,32]{1,0})
      tuple(%done0, %done1, %pad_std_data)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnUnverifiedModule(kFiveLayerDeduplicatedHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[2].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsNestedAsyncTileSharedReaderAndAliasedAsyncStart) {
  constexpr absl::string_view kInnerTileSharedReaderHlo = R"hlo(
HloModule nested_async_tile_shared_reader

%inner_tile_callee (leaf_p: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> u32[16,32] {
  %leaf_p = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
}

%inner_tile_async_wrap (tile_p: s64[<=16,32], tile_d0: s32[], tile_d1: s32[]) -> u32[16,32] {
  %tile_p = s64[<=16,32]{1,0} parameter(0)
  %tile_d0 = s32[] parameter(1)
  %tile_d1 = s32[] parameter(2)
  ROOT %inner_call = u32[16,32]{1,0} call(%tile_p, %tile_d0, %tile_d1),
      to_apply=%inner_tile_callee
}, execution_thread="sparsecore"

%other_tile_wrap (other_p: s64[<=16,32]) -> s64[<=16,32] {
  %other_p = s64[<=16,32]{1,0} parameter(0)
  ROOT %sum = s64[<=16,32]{1,0} add(%other_p, %other_p)
}, execution_thread="sparsecore"

%reconstructed_scs_comp (scs_tuple_param: (s64[<=16,32], s32[], s32[])) -> (u32[16,32], s64[<=16,32]) {
  %scs_tuple_param = (s64[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  %scs_p = s64[<=16,32]{1,0} get-tuple-element(%scs_tuple_param), index=0
  %scs_d0 = s32[] get-tuple-element(%scs_tuple_param), index=1
  %scs_d1 = s32[] get-tuple-element(%scs_tuple_param), index=2
  %tile_start = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-start(%scs_p, %scs_d0, %scs_d1),
      async_execution_thread="sparsecore", calls=%inner_tile_async_wrap
  %tile_done = u32[16,32]{1,0} async-done(%tile_start),
      async_execution_thread="sparsecore"
  %other_tile_start = ((s64[<=16,32]{1,0}), s64[<=16,32]{1,0})
      async-start(%scs_p),
      async_execution_thread="sparsecore", calls=%other_tile_wrap
  %other_tile_done = s64[<=16,32]{1,0} async-done(%other_tile_start),
      async_execution_thread="sparsecore"
  ROOT %res = (u32[16,32]{1,0}, s64[<=16,32]{1,0})
      tuple(%tile_done, %other_tile_done)
}

%outer_async_wrap (outer_tuple: (s64[<=16,32], s32[], s32[])) -> (u32[16,32], s64[<=16,32]) {
  %outer_tuple = (s64[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  ROOT %sc_call = (u32[16,32]{1,0}, s64[<=16,32]{1,0})
      custom-call(%outer_tuple),
      custom_call_target="AsyncOffloadCall", to_apply=%reconstructed_scs_comp
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], d0: s32[], d1: s32[]) -> (u32[16,32], s64[<=16,32]) {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %d0 = s32[] parameter(1)
  %d1 = s32[] parameter(2)
  %tuple0 = (s64[<=16,32]{1,0}, s32[], s32[]) tuple(%p0, %d0, %d1)
  %start0 = ((), ()) async-start(),
      async_execution_thread="sparsecore", calls=%outer_async_wrap
  %update0 = (((s64[<=16,32]{1,0}, s32[], s32[])), (u32[16,32]{1,0}, s64[<=16,32]{1,0}))
      async-update(%start0, %tuple0), async_execution_thread="sparsecore"
  ROOT %done0 = (u32[16,32]{1,0}, s64[<=16,32]{1,0}) async-done(%update0),
      async_execution_thread="sparsecore"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> shared_tile_module,
      ParseAndReturnUnverifiedModule(kInnerTileSharedReaderHlo));
  std::vector<Shape> shared_tile_shapes =
      MakeCanonicalDynamicParameterShapes(*shared_tile_module);
  EXPECT_THAT(ClearPadRealToStaticParameterMetadataPrefixes(
                  *shared_tile_module, absl::MakeSpan(shared_tile_shapes)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("Entry parameter 0"),
                             HasSubstr("other_tile_start"))));
  EXPECT_EQ(
      shared_tile_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
      1024);

  constexpr absl::string_view kAliasedAsyncStartHlo = R"hlo(
HloModule aliased_async_start_pad_real_to_static

%async_pad (ap0: s32[<=16,32], ad0: s32[], ad1: s32[]) -> s32[16,32] {
  %ap0 = s32[<=16,32]{1,0} parameter(0)
  %ad0 = s32[] parameter(1)
  %ad1 = s32[] parameter(2)
  ROOT %pad = s32[16,32]{1,0} custom-call(%ap0, %ad0, %ad1),
      custom_call_target="PadRealToStatic"
}, execution_thread="sparsecore"

ENTRY %main (p0: s32[<=16,32], d0: s32[], d1: s32[]) -> s32[16,32] {
  %p0 = s32[<=16,32]{1,0} parameter(0)
  %d0 = s32[] parameter(1)
  %d1 = s32[] parameter(2)
  %start = ((s32[<=16,32]{1,0}, s32[], s32[]), s32[16,32]{1,0})
      async-start(%p0, %d0, %d1),
      async_execution_thread="sparsecore",
      output_to_operand_aliasing={{1}: (0, {})},
      calls=%async_pad
  ROOT %done = s32[16,32]{1,0} async-done(%start),
      async_execution_thread="sparsecore"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> aliased_async_module,
                       ParseAndReturnUnverifiedModule(kAliasedAsyncStartHlo));
  std::vector<Shape> aliased_async_shapes =
      MakeCanonicalDynamicParameterShapes(*aliased_async_module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *aliased_async_module, absl::MakeSpan(aliased_async_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0"), HasSubstr("%start"),
                     HasSubstr("output_to_operand_aliasing"))));
  EXPECT_EQ(
      aliased_async_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
      1024);
}

class FakeDynamicPrefixTopology : public PjRtTopologyDescription {
 public:
  explicit FakeDynamicPrefixTopology(int default_memory_space_kind_id)
      : default_memory_space_kind_id_(default_memory_space_kind_id) {}

  PjRtPlatformId platform_id() const override { return TpuId(); }
  absl::string_view platform_name() const override { return "tpu"; }
  absl::string_view platform_version() const override { return "fake"; }
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions()
      const override {
    return {};
  }
  absl::StatusOr<uint64_t> Fingerprint() const override { return 0; }
  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }
  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override {
    return LayoutUtil::GetDefaultLayoutForShape(
        ShapeUtil::MakeShape(element_type, dims));
  }
  absl::StatusOr<Shape> MakeCanonicalShapeForMemorySpace(
      int memory_space_kind_id, Shape shape,
      const Layout* layout) const override {
    if (layout != nullptr) {
      *shape.mutable_layout() = *layout;
    } else {
      LayoutUtil::SetToDefaultLayout(&shape);
    }
    if (shape.IsArray() && shape.is_dynamic()) {
      shape.mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(1024);
    }
    return shape;
  }
  absl::StatusOr<int> GetMemorySpaceKindForShape(
      const Shape& shape) const override {
    return default_memory_space_kind_id_;
  }

 private:
  int default_memory_space_kind_id_;
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_;
};

class InferDispatchInfoMlirPadRealToStaticTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(client_, GetXlaPjrtCpuClient(CpuClientOptions{}));
    ASSERT_FALSE(client_->addressable_devices().empty());
    device_ = client_->addressable_devices().front();
    ASSERT_OK_AND_ASSIGN(PjRtMemorySpace * default_ms,
                         device_->default_memory_space());
    topology_ =
        std::make_unique<FakeDynamicPrefixTopology>(default_ms->kind_id());
    device_assignment_ = std::make_shared<DeviceAssignment>(1, 1);
    (*device_assignment_)(0, 0) = device_->global_device_id().value();
  }

  absl::StatusOr<PjRtLoadedExecutableDispatchInfo> RunMlirInferDispatchInfo(
      absl::string_view mlir_source) {
    mlir::MLIRContext context;
    ABSL_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModuleString(mlir_source, context));
    CompileOptions options;
    std::vector<PjRtLoadedExecutable::LogicalDeviceIds> logical_ids = {
        PjRtLoadedExecutable::LogicalDeviceIds{0, 0}};
    std::vector<PjRtDevice*> addressable_devices = {device_};
    return InferDispatchInfo(topology_.get(), *module, options,
                             device_assignment_, std::move(logical_ids),
                             std::move(addressable_devices),
                             /*tuple_inputs=*/false);
  }

  std::unique_ptr<PjRtClient> client_;
  PjRtDevice* device_ = nullptr;
  std::unique_ptr<FakeDynamicPrefixTopology> topology_;
  std::shared_ptr<DeviceAssignment> device_assignment_;
};

TEST_F(InferDispatchInfoMlirPadRealToStaticTest,
       ClearsPrefixForPadRealToStaticAndPreservesPadToStatic) {
  constexpr absl::string_view kMlir = R"mlir(
module @pad_real_vs_pad_std_mlir {
  sdy.mesh @mesh = <[]>
  func.func @main(
      %arg0: tensor<?x32xi64, #stablehlo.bounds<16, ?>>
          {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
      %arg1: tensor<?x32xf32, #stablehlo.bounds<16, ?>>
          {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
      %arg2: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>},
      %arg3: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>})
      -> (tensor<16x32xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
          tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %0 = stablehlo.custom_call @PadRealToStatic(%arg0, %arg2, %arg3)
        : (tensor<?x32xi64, #stablehlo.bounds<16, ?>>, tensor<i32>, tensor<i32>)
        -> tensor<16x32xi64>
    %1:3 = stablehlo.custom_call @PadToStatic(%arg1)
        : (tensor<?x32xf32, #stablehlo.bounds<16, ?>>)
        -> (tensor<16x32xf32>, tensor<i32>, tensor<i32>)
    return %0, %1#0 : tensor<16x32xi64>, tensor<16x32xf32>
  }
}
)mlir";

  ASSERT_OK_AND_ASSIGN(PjRtLoadedExecutableDispatchInfo info,
                       RunMlirInferDispatchInfo(kMlir));
  ASSERT_EQ(info.parameter_device_shapes.size(), 4);
  EXPECT_TRUE(info.parameter_device_shapes[0].is_bounded_dynamic_dimension(0));
  EXPECT_EQ(info.parameter_device_shapes[0].dimensions(0), 16);
  EXPECT_EQ(info.parameter_device_shapes[0]
                .layout()
                .dynamic_shape_metadata_prefix_bytes(),
            0);
  EXPECT_TRUE(info.parameter_device_shapes[1].is_bounded_dynamic_dimension(0));
  EXPECT_EQ(info.parameter_device_shapes[1].dimensions(0), 16);
  EXPECT_EQ(info.parameter_device_shapes[1]
                .layout()
                .dynamic_shape_metadata_prefix_bytes(),
            1024);
  ASSERT_OK_AND_ASSIGN(
      const std::vector<std::shared_ptr<const PjRtLayout>>& layouts,
      info.extras->parameter_layouts);
  EXPECT_EQ(layouts[0]->xla_layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(layouts[1]->xla_layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST_F(InferDispatchInfoMlirPadRealToStaticTest,
       RejectsSharedDynamicArgumentInMlir) {
  constexpr absl::string_view kSharedOpMlir = R"mlir(
module @shared_op_mlir {
  sdy.mesh @mesh = <[]>
  func.func @main(
      %arg0: tensor<?x32xi64, #stablehlo.bounds<16, ?>>
          {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
      %arg1: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>},
      %arg2: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>})
      -> (tensor<16x32xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
          tensor<?x32xi64, #stablehlo.bounds<16, ?>>
              {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %0 = stablehlo.custom_call @PadRealToStatic(%arg0, %arg1, %arg2)
        : (tensor<?x32xi64, #stablehlo.bounds<16, ?>>, tensor<i32>, tensor<i32>)
        -> tensor<16x32xi64>
    %1 = stablehlo.add %arg0, %arg0
        : tensor<?x32xi64, #stablehlo.bounds<16, ?>>
    return %0, %1 : tensor<16x32xi64>,
                    tensor<?x32xi64, #stablehlo.bounds<16, ?>>
  }
}
)mlir";

  EXPECT_THAT(RunMlirInferDispatchInfo(kSharedOpMlir),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("Entry argument 0"),
                             HasSubstr("PadRealToStatic"),
                             HasSubstr("stablehlo.add"))));

  constexpr absl::string_view kReturnedArgMlir = R"mlir(
module @returned_arg_mlir {
  sdy.mesh @mesh = <[]>
  func.func @main(
      %arg0: tensor<?x32xi64, #stablehlo.bounds<16, ?>>
          {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
      %arg1: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>},
      %arg2: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>})
      -> (tensor<16x32xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
          tensor<?x32xi64, #stablehlo.bounds<16, ?>>
              {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %0 = stablehlo.custom_call @PadRealToStatic(%arg0, %arg1, %arg2)
        : (tensor<?x32xi64, #stablehlo.bounds<16, ?>>, tensor<i32>, tensor<i32>)
        -> tensor<16x32xi64>
    return %0, %arg0 : tensor<16x32xi64>,
                       tensor<?x32xi64, #stablehlo.bounds<16, ?>>
  }
}
)mlir";

  EXPECT_THAT(
      RunMlirInferDispatchInfo(kReturnedArgMlir),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry argument 0"), HasSubstr("func.return"))));
}

TEST_F(InferDispatchInfoMlirPadRealToStaticTest,
       RejectsAliasedDynamicArgumentInMlir) {
  constexpr absl::string_view kAliasedMlir = R"mlir(
module @aliased_arg_mlir {
  sdy.mesh @mesh = <[]>
  func.func @main(
      %arg0: tensor<?x32xi64, #stablehlo.bounds<16, ?>>
          {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>,
           tf.aliasing_output = 0 : i32},
      %arg1: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>},
      %arg2: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>})
      -> (tensor<16x32xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %0 = stablehlo.custom_call @PadRealToStatic(%arg0, %arg1, %arg2)
        : (tensor<?x32xi64, #stablehlo.bounds<16, ?>>, tensor<i32>, tensor<i32>)
        -> tensor<16x32xi64>
    return %0 : tensor<16x32xi64>
  }
}
)mlir";

  EXPECT_THAT(RunMlirInferDispatchInfo(kAliasedMlir),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("Entry argument 0"),
                             HasSubstr("tf.aliasing_output"))));
}

}  // namespace
}  // namespace xla
