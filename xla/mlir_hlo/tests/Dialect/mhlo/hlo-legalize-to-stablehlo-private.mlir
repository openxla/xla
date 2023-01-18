// RUN: mlir-hlo-opt --hlo-legalize-to-stablehlo=allow-private-features --mlir-print-op-generic --split-input-file --verify-diagnostics %s | FileCheck %s

// -----

func.func @op_add_dependency(%arg0: tensor<16xf32>, %arg1: !mhlo.token) -> tensor<16xf32> {
  // CHECK: stablehlo.custom_call
  // CHECK-SAME: call_target_name = "mhlo.add_dependency"
  %0 = "mhlo.add_dependency"(%arg0, %arg1) : (tensor<16xf32>, !mhlo.token) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}

// -----

func.func @op_bitcast(%arg0: tensor<i32>) -> tensor<f32> {
  // CHECK: stablehlo.custom_call
  // CHECK-SAME: call_target_name = "mhlo.bitcast"
  %0 = "mhlo.bitcast"(%arg0) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @method(%arg0 : tensor<f32>) -> tensor<f32> {
  func.return %arg0 : tensor<f32>
}

func.func @op_copy(%arg0: tensor<f32>) -> tensor<f32> {
  // mhlo.copy is immediately folded away at the first opportunity,
  // so it doesn't seem to be possible to capture it in FileCheck tests.
  %0 = "mhlo.copy"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @op_convolution_unknown_dimension_numbers(%arg0: tensor<1x8x8x32x207xf32>, %arg1: tensor<3x3x32x207x16xf32>) -> tensor<32x1x8x8x16xf32> {
  // CHECK: stablehlo.custom_call
  // CHECK-SAME: call_target_name = "mhlo.convolution"
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    window_strides = dense<1> : tensor<2xi64>,
    padding = dense<1> : tensor<2x2xi64>,
    lhs_dilation = dense<1> : tensor<2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_reversal = dense<false> : tensor<2xi1>,
    dimension_numbers = #mhlo.conv<[b, 0, 1, ?, f]x[0, 1, ?, i, o]->[?, b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<1x8x8x32x207xf32>, tensor<3x3x32x207x16xf32>) -> tensor<32x1x8x8x16xf32>
  func.return %0 : tensor<32x1x8x8x16xf32>
}

// -----

func.func @op_custom_call_custom_call_schedule(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: stablehlo.custom_call
  // CHECK-SAME: call_target_name = "mhlo.custom_call"
  %0 = "mhlo.custom_call"(%arg0) {
    call_target_name = "foo",
    custom_call_schedule = #mhlo<custom_call_schedule EARLIEST>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @op_domain(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: stablehlo.custom_call
  // CHECK-SAME: call_target_name = "mhlo.domain"
  %0 = "mhlo.domain"(%arg0) {
    kind = #mhlo<kind sharding>,
    entry_metadata = "\08\01\1A\01\01\22\01\01",
    exit_metadata = "\08\02"
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @op_stochastic_convert(%arg0: tensor<f32>, %arg1: tensor<ui32>) -> tensor<i8> {
  // CHECK: stablehlo.custom_call
  // CHECK-SAME: call_target_name = "mhlo.stochastic_convert"
  %0 = "mhlo.stochastic_convert"(%arg0, %arg1) : (tensor<f32>, tensor<ui32>) -> tensor<i8>
  return %0 : tensor<i8>
}

// -----

func.func @op_xla_rng_get_and_update_state() -> tensor<2xui64> {
  // CHECK: stablehlo.custom_call
  // CHECK-SAME: call_target_name = "mhlo.xla.rng_get_and_update_state"
  %0 = "mhlo.xla.rng_get_and_update_state"() {
    delta = 1: i64
  } : () -> tensor<2xui64>
  func.return %0 : tensor<2xui64>
}

// ============ NEGATIVE TESTS ============
// Ops that have MHLO types or attributes that cannot be represented in
// StableHLO or regions are currently not supported for encoding.
// MHLO ops that don't exist in StableHLO with regions are currently not
// supported.

// -----

func.func @async_computation(%arg0: tensor<16xf32>) -> tensor<16xf32>
  attributes {execution_thread = "main"} {
  return %arg0 : tensor<16xf32>
}

func.func @op_async_done(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  // expected-error@+1 {{failed to legalize operation 'mhlo.async_start' that was explicitly marked illegal}}
  %0 = "mhlo.async_start"(%arg0) {
    called_computation = @async_computation,
    execution_thread = "main"
  } : (tensor<16xf32>) -> !mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>>
  %1 = "mhlo.async_done"(%0) {
    called_computation = @async_computation,
    execution_thread = "main"
  } : (!mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>>) -> tensor<16xf32>
  func.return %1 : tensor<16xf32>
}

// -----

func.func @async_computation_start(%arg0: tensor<16xf32>) -> tensor<16xf32>
  attributes {execution_thread = "main"} {
  return %arg0 : tensor<16xf32>
}

// expected-error@+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @op_async_start(%arg0: tensor<16xf32>) -> !mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>> {
  %0 = "mhlo.async_start"(%arg0) {
    called_computation = @async_computation_start,
    execution_thread = "main"
  } : (tensor<16xf32>) -> !mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>>
  func.return %0 : !mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>>
}

// -----

func.func @async_computation_update(%arg0: tensor<16xf32>) -> tensor<16xf32>
  attributes {execution_thread = "main"} {
  return %arg0 : tensor<16xf32>
}

// expected-error@+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @op_async_update(%arg0: !mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>>) -> !mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>> {
  %0 = "mhlo.async_update"(%arg0) {
    called_computation = @async_computation_update,
    execution_thread = "main"
  } : (!mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>>) -> !mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>>
  func.return %0 : !mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>>
}

// -----

func.func @op_fusion(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{failed to legalize operation 'mhlo.fusion' that was explicitly marked illegal}}
  %0 = "mhlo.fusion"(%arg0) ({
    ^bb0(%arg1: tensor<f32>):
      "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  }) {
    fusion_kind = #mhlo<fusion_kind kCustom>,
    output_operand_aliases = [
      #mhlo.output_operand_alias<output_tuple_indices = [],
                                 operand_index = 0,
                                 operand_tuple_indices = []>
    ]
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

