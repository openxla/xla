// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo %s | FileCheck %s

module attributes {
  mhlo.use_auto_spmd_partitioning = true,
  mhlo.is_dynamic = true,
  mhlo.dynamic_parameter_bindings = [
    #mhlo.dynamic_parameter_binding<
      dynamic_param_num = 0,
      dynamic_param_indices = [],
      target = kParam,
      target_num = 1,
      target_indices = [],
      target_dim_num = 0>] } {
  func.func @main(%a : tensor<i32>, %b : tensor<?xf32, #mhlo.type_extensions<bounds = [2]>>) -> () {
    func.return
  }
}

// CHECK-LABEL: hlo_module       {
// CHECK: dynamic_parameter_binding {
// CHECK-NEXT: entries {
// CHECK-NEXT:    target_num: 1
// CHECK-NEXT:    target: KPARAM
// CHECK-NEXT:  }
// CHECK: is_dynamic: true
// CHECK: use_auto_spmd_partitioning: true

// -----

module attributes {
  mhlo.use_auto_spmd_partitioning = true,
  mhlo.is_dynamic = true,
  mhlo.dynamic_parameter_bindings = [
    #mhlo.dynamic_parameter_binding<
      dynamic_param_num = 0,
      dynamic_param_indices = [],
      target = kOutput,
      target_num = 0,
      target_indices = [],
      target_dim_num = 0>] } {
  func.func @main(%a : tensor<i32>, %b : tensor<?xf32, #mhlo.type_extensions<bounds = [2]>>) -> (tensor<?xf32, #mhlo.type_extensions<bounds = [2]>>) {
    func.return %b : tensor<?xf32, #mhlo.type_extensions<bounds = [2]>>
  }
}

// CHECK-LABEL: hlo_module       {
// CHECK: dynamic_parameter_binding {
// CHECK: entries {
// CHECK-NEXT:   target: KOUTPUT
// CHECK-NEXT: }
// CHECK: is_dynamic: true
// CHECK: use_auto_spmd_partitioning: true

// -----

module attributes {
  mhlo.use_auto_spmd_partitioning = true,
  mhlo.is_dynamic = true,
  mhlo.dynamic_parameter_bindings = [
    #mhlo.dynamic_parameter_binding<
      dynamic_param_num = 0,
      dynamic_param_indices = [],
      target = kOutput,
      target_num = 1,
      target_indices = [],
      target_dim_num = 0>] } {
  func.func @main(%a : tensor<i32>, %b : tensor<?xf32, #mhlo.type_extensions<bounds = [2]>>) -> (tensor<i32>, tensor<?xf32, #mhlo.type_extensions<bounds = [2]>>) {
    func.return %a, %b : tensor<i32>, tensor<?xf32, #mhlo.type_extensions<bounds = [2]>>
  }
}

// CHECK-LABEL: hlo_module       {
// CHECK: dynamic_parameter_binding {
// CHECK: entries {
// CHECK-NEXT:   target_index: 1
// CHECK-NEXT:   target: KOUTPUT
// CHECK-NEXT: }
// CHECK: is_dynamic: true
// CHECK: use_auto_spmd_partitioning: true
