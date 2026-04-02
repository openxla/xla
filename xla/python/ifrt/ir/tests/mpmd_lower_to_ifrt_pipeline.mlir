// RUN: ifrt-opt %s -ifrt-mpmd-lower-to-ifrt-pipeline -verify-diagnostics -split-input-file 2>&1 | FileCheck %s

!arg_0_tensor = !mpmd.mesh_tensor<"mesh1", tensor<3x5xf32>>
!arg_1_tensor = !mpmd.mesh_tensor<"mesh2", tensor<5x7xf32>>
!arg_2_tensor = !mpmd.mesh_tensor<"mesh1", tensor<10x3xf32>, sharding=<@mesh, [{"x"}, {}]>>
!tmp_tensor_mesh1 = !mpmd.mesh_tensor<"mesh1", tensor<10x5xf32>, sharding=<@mesh, [{"x"}, {}]>>
!tmp_tensor_mesh2 = !mpmd.mesh_tensor<"mesh2", tensor<10x5xf32>, sharding=<@mesh, [{"x"}, {}]>>
!res_tensor = !mpmd.mesh_tensor<"mesh2", tensor<10x7xf32>, sharding=<@mesh, [{"x"}, {}]>>

// CHECK: #sp = #ifrt.sharding_param<1x1 to [0] on 2>
// CHECK: #sp1 = #ifrt.sharding_param<2x1 to [0] on 2>
// CHECK-LABEL: module

// CHECK-NOT: sdy.mesh
sdy.mesh @mesh = <["x"=2]>

// CHECK: func.func public @main
// CHECK-SAME:      %arg0: !ifrt.array<tensor<3x5xf32>, #sp, [0, 1]>,
// CHECK-SAME:      %arg1: !ifrt.array<tensor<5x7xf32>, #sp, [2, 3]>,
// CHECK-SAME:      %arg2: !ifrt.array<tensor<10x3xf32>, #sp1, [0, 1]>)
// CHECK-SAME:      -> !ifrt.array<tensor<10x7xf32>, #sp1, [2, 3]>
// CHECK-SAME:      xla_tpu_user_reserved_hbm_bytes = 256 : i64
func.func public @main(%arg0: !arg_0_tensor,
                       %arg1: !arg_1_tensor,
                       %arg2: !arg_2_tensor)
  -> (!res_tensor) attributes {
    topology = #mpmd.topology<<"mesh1" : <["x"=2]>>, <"mesh2" : <["x"=2]>>>,
    xla_tpu_user_reserved_hbm_bytes = 256 : i64} {
      // CHECK-NEXT: %[[OUTPUTS_0:.*]], %[[CONTROL_OUTPUT_0:.*]] = ifrt.Call @stage1::@main(%arg0, %arg2)
      // CHECK-NEXT: %[[OUTPUTS_1:.*]], %[[CONTROL_OUTPUT_1:.*]] = ifrt.CopyArrays(%[[OUTPUTS_0]])
      // CHECK-NEXT: %[[OUTPUTS_2:.*]], %[[CONTROL_OUTPUT_2:.*]] = ifrt.Call @stage2::@main(%arg1, %[[OUTPUTS_1]])
      // CHECK-NEXT: return %[[OUTPUTS_2]]
      %0 = mpmd.fragment_call<mesh="mesh1", origin=[]> @stage1(%arg0, %arg2) {mpmd.is_gspmd_partitioned} : (!arg_0_tensor, !arg_2_tensor) -> !tmp_tensor_mesh1
      %1 = mpmd.transfer %0 : (!tmp_tensor_mesh1) -> !tmp_tensor_mesh2
      %2 = mpmd.fragment_call<mesh="mesh2", origin=[]> @stage2(%arg1, %1) : (!arg_1_tensor, !tmp_tensor_mesh2) -> !res_tensor
      return %2 : !res_tensor
}
// CHECK: module @stage1 attributes {sym_visibility = "private"} {
// CHECK-NEXT: func.func @main(%arg0: tensor<3x5xf32>, %arg1: tensor<10x3xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2]>, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<10x5xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2]>, [{"x"}, {}]>}) {
func.func @stage1(%arg0: tensor<3x5xf32>, %arg1: tensor<10x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
  -> (tensor<10x5xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) attributes {mesh_shape = #sdy.mesh<["x"=2]>} {
    %0 = "stablehlo.dot"(%arg1, %arg0) : (tensor<10x3xf32>, tensor<3x5xf32>) -> tensor<10x5xf32>
    return %0 : tensor<10x5xf32>
}
// CHECK: module @stage2 attributes {sym_visibility = "private"} {
// CHECK-NEXT: func.func @main(%arg0: tensor<5x7xf32>, %arg1: tensor<10x5xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2]>, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<10x7xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2]>, [{"x"}, {}]>}) {
func.func @stage2(%arg0: tensor<5x7xf32>, %arg1: tensor<10x5xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
  -> (tensor<10x7xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) attributes {mesh_shape = #sdy.mesh<["x"=2]>} {
    %0 = "stablehlo.dot"(%arg1, %arg0) : (tensor<10x5xf32>, tensor<5x7xf32>) -> tensor<10x7xf32>
    return %0 : tensor<10x7xf32>
}