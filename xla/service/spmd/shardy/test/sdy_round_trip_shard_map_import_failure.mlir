// RUN: sdy_opt %s -xla-sdy-round-trip-shard-map-import -split-input-file -verify-diagnostics

sdy.mesh @mesh = <["a"=2]>

func.func @using_same_body_func(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0) : (tensor<8x8xf32>) -> (tensor<2x8xf32>)
  %1 = call @xla.sdy.manual_computation_body(%0) {mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\\\22a\\\22}, {\\\22b\\\22}]>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\\\22a\\\22, \\\22b\\\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\\\22a\\\22}, {}], replicated={\\\22b\\\22}>]>"}} : (tensor<2x8xf32>) -> (tensor<2x8xf32>)
  %2 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%1) : (tensor<2x8xf32>) -> (tensor<8x8xf32>)
  %3 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%2) : (tensor<8x8xf32>) -> (tensor<2x8xf32>)
  // expected-error @+2 {{'func.call' op expected a unique FuncOp per @xla.sdy.manual_computation_body call}}
  // expected-error @+1 {{failed to legalize operation 'func.call'}}
  %4 = call @xla.sdy.manual_computation_body(%3) {mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\\\22a\\\22}, {\\\22b\\\22}]>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\\\22a\\\22, \\\22b\\\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\\\22a\\\22}, {}], replicated={\\\22b\\\22}>]>"}} : (tensor<2x8xf32>) -> (tensor<2x8xf32>)
  %5 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%4) : (tensor<2x8xf32>) -> (tensor<8x8xf32>)
  return %5 : tensor<8x8xf32>
}

func.func @xla.sdy.manual_computation_body(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  return %arg0 : tensor<2x8xf32>
}
