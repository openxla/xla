// RUN: sdy_opt %s -xla-sdy-round-trip-shard-map-import -split-input-file -verify-diagnostics

func.func @using_same_body_func(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0) : (tensor<8x8xf32>) -> (tensor<2x8xf32>)
  %1 = call @xla.sdy.manual_computation_body(%0) {mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}, {\\\22b\\\22}]>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\\\22a\\\22, \\\22b\\\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}, {}], replicated={\\\22b\\\22}>]>"}} : (tensor<2x8xf32>) -> (tensor<2x8xf32>)
  %2 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%1) : (tensor<2x8xf32>) -> (tensor<8x8xf32>)
  %3 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%2) : (tensor<8x8xf32>) -> (tensor<2x8xf32>)
  // expected-error @+2 {{'func.call' op expected a unique FuncOp per @xla.sdy.manual_computation_body call}}
  // expected-error @+1 {{failed to legalize operation 'func.call'}}
  %4 = call @xla.sdy.manual_computation_body(%3) {mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}, {\\\22b\\\22}]>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\\\22a\\\22, \\\22b\\\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}, {}], replicated={\\\22b\\\22}>]>"}} : (tensor<2x8xf32>) -> (tensor<2x8xf32>)
  %5 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%4) : (tensor<2x8xf32>) -> (tensor<8x8xf32>)
  return %5 : tensor<8x8xf32>
}

func.func @xla.sdy.manual_computation_body(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  return %arg0 : tensor<2x8xf32>
}

// -----

func.func @manual_computation_missing_global_to_local_shape(%arg0: tensor<0x16xf32>) -> (tensor<0x16xf32>) {
  %c = stablehlo.constant dense<0.000000e+00> : tensor<0x8xf32>
  // expected-error @+2 {{'func.call' op expected at least one operand of @xla.sdy.manual_computation_body to be produced by a xla.sdy.GlobalToLocalShape CustomCallOp}}
  // expected-error @+1 {{failed to legalize operation 'func.call'}}
  %0 = call @xla.sdy.manual_computation_body(%c) : (tensor<0x8xf32>) -> tensor<0x8xf32>
  %1 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%0) {mhlo.frontend_attributes = {xla.sdy.manual_axes = "#sdy<manual_axes{\22b\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh, [{}, {\22b\22}]>]>"}} : (tensor<0x8xf32>) -> (tensor<0x16xf32>)
  return %1 : tensor<0x16xf32>
}

func.func @xla.sdy.manual_computation_body(%arg0: tensor<0x8xf32>) -> tensor<0x8xf32> {
  return %arg0 : tensor<0x8xf32>
}

// -----

func.func @manual_computation_missing_local_to_global_shape(%arg0: tensor<0x16xf32>) -> (tensor<0x16xf32>) {
  %c = stablehlo.constant dense<0.000000e+00> : tensor<0x16xf32>
  %0 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0) {mhlo.frontend_attributes = {xla.sdy.manual_axes = "#sdy<manual_axes{\22b\22}>", xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh, [{}, {\22b\22}]>]>"}} : (tensor<0x16xf32>) -> tensor<0x8xf32>
  // expected-error @+2 {{'func.call' op expected the first use of @xla.sdy.manual_computation_body to be by a xla.sdy.LocalToGlobalShape CustomCallOp}}
  // expected-error @+1 {{failed to legalize operation 'func.call'}}
  %1 = call @xla.sdy.manual_computation_body(%0) : (tensor<0x8xf32>) -> tensor<0x8xf32>
  return %c : tensor<0x16xf32>
}

func.func @xla.sdy.manual_computation_body(%arg0: tensor<0x8xf32>) -> tensor<0x8xf32> {
  return %arg0 : tensor<0x8xf32>
}

