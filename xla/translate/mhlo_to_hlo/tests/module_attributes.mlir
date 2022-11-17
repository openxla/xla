// RUN: xla-translate -verify-diagnostics -split-input-file -mlir-hlo-to-hlo %s | FileCheck %s
// RUN: xla-translate -verify-diagnostics -split-input-file -mlir-hlo-to-hlo --via-builder=true %s | FileCheck %s

module attributes { mhlo.cross_program_prefetches = [ #mhlo.cross_program_prefetch<parameter = 1, indices = [0]> ], mhlo.device_assignment = #mhlo.device_assignment<replicaCount = 3, computationCount = 1, computationDevices = 0,1,2> } {
  func.func @copy(%arg0 : tuple<tensor<2x3xi32>, tensor<i32>>) -> tuple<tensor<2x3xi32>, tensor<i32>> attributes {execution_thread = "main"} {
    %0 = "mhlo.copy"(%arg0) {is_cross_program_prefetch} : (tuple<tensor<2x3xi32>, tensor<i32>>) -> (tuple<tensor<2x3xi32>, tensor<i32>>)
    return %0 : tuple<tensor<2x3xi32>, tensor<i32>>
  }
  func.func @main(%arg0 : tensor<i32>, %arg1 : tuple<tensor<2x3xi32>, tensor<i32>>) -> tuple<tensor<2x3xi32>, tensor<i32>> attributes { mhlo.computation_ids = dense<1>:tensor<1xi64> } {
    %1 = "mhlo.async_start"(%arg1) {called_computation=@copy, execution_thread="main"} : (tuple<tensor<2x3xi32>, tensor<i32>>) -> (!mhlo.async_bundle<tuple<tuple<tensor<2x3xi32>, tensor<i32>>>, tuple<tuple<tensor<2x3xi32>, tensor<i32>>>>)
    %2 = "mhlo.async_done"(%1) {called_computation=@copy, execution_thread="main"} : (!mhlo.async_bundle<tuple<tuple<tensor<2x3xi32>, tensor<i32>>>, tuple<tuple<tensor<2x3xi32>, tensor<i32>>>>) -> (tuple<tensor<2x3xi32>, tensor<i32>>)
    return %2 : tuple<tensor<2x3xi32>, tensor<i32>>
  }
}
// CHECK-LABEL: hlo_module       {
// CHECK: cross_program_prefetches {
// CHECK-NEXT:    parameter: 1
// CHECK-NEXT:    index: 0
// CHECK-NEXT:  }

// CHECK: device_assignment {
// CHECK-NEXT:    replica_count: 3
// CHECK-NEXT:    computation_count: 1
// CHECK-NEXT:    computation_devices {
// CHECK-NEXT:      replica_device_ids: 0
// CHECK-NEXT:      replica_device_ids: 1
// CHECK-NEXT:      replica_device_ids: 2
// CHECK-NEXT:    }

// -----

// expected-error@+1 {{device_assignment: size of device assignment must equal replicaCount * computationCount}}
module attributes { mhlo.cross_program_prefetches = [ #mhlo.cross_program_prefetch<parameter = 1, indices = [0]> ], mhlo.device_assignment = #mhlo.device_assignment<replicaCount = 3, computationCount = 1, computationDevices = 0,1,2,3,4,5> } {
  func.func @copy(%arg0 : tuple<tensor<2x3xi32>, tensor<i32>>) -> tuple<tensor<2x3xi32>, tensor<i32>> attributes {execution_thread = "main"} {
    %0 = "mhlo.copy"(%arg0) {is_cross_program_prefetch} : (tuple<tensor<2x3xi32>, tensor<i32>>) -> (tuple<tensor<2x3xi32>, tensor<i32>>)
    return %0 : tuple<tensor<2x3xi32>, tensor<i32>>
  }
  func.func @main(%arg0 : tensor<i32>, %arg1 : tuple<tensor<2x3xi32>, tensor<i32>>) -> tuple<tensor<2x3xi32>, tensor<i32>> {
    %1 = "mhlo.async_start"(%arg1) {called_computation=@copy, execution_thread="main"} : (tuple<tensor<2x3xi32>, tensor<i32>>) -> (!mhlo.async_bundle<tuple<tuple<tensor<2x3xi32>, tensor<i32>>>, tuple<tuple<tensor<2x3xi32>, tensor<i32>>>>)
    %2 = "mhlo.async_done"(%1) {called_computation=@copy, execution_thread="main"} : (!mhlo.async_bundle<tuple<tuple<tensor<2x3xi32>, tensor<i32>>>, tuple<tuple<tensor<2x3xi32>, tensor<i32>>>>) -> (tuple<tensor<2x3xi32>, tensor<i32>>)
    return %2 : tuple<tensor<2x3xi32>, tensor<i32>>
  }
}

// -----


// expected-error@+1 {{cross_program_prefetch: parameter 2 out of range. main has only 2 arguments}}
module attributes { mhlo.cross_program_prefetches = [ #mhlo.cross_program_prefetch<parameter = 2, indices = [0]> ] } {
  func.func @copy(%arg0 : tuple<tensor<2x3xi32>, tensor<i32>>) -> tuple<tensor<2x3xi32>, tensor<i32>> attributes {execution_thread = "main"} {
    %0 = "mhlo.copy"(%arg0) {is_cross_program_prefetch} : (tuple<tensor<2x3xi32>, tensor<i32>>) -> (tuple<tensor<2x3xi32>, tensor<i32>>)
    return %0 : tuple<tensor<2x3xi32>, tensor<i32>>
  }
  func.func @main(%arg0 : tensor<i32>, %arg1 : tuple<tensor<2x3xi32>, tensor<i32>>) -> tuple<tensor<2x3xi32>, tensor<i32>> {
    %1 = "mhlo.async_start"(%arg1) {called_computation=@copy, execution_thread="main"} : (tuple<tensor<2x3xi32>, tensor<i32>>) -> (!mhlo.async_bundle<tuple<tuple<tensor<2x3xi32>, tensor<i32>>>, tuple<tuple<tensor<2x3xi32>, tensor<i32>>>>)
    %2 = "mhlo.async_done"(%1) {called_computation=@copy, execution_thread="main"} : (!mhlo.async_bundle<tuple<tuple<tensor<2x3xi32>, tensor<i32>>>, tuple<tuple<tensor<2x3xi32>, tensor<i32>>>>) -> (tuple<tensor<2x3xi32>, tensor<i32>>)
    return %2 : tuple<tensor<2x3xi32>, tensor<i32>>
  }
}

// -----

// expected-error@+1 {{cross_program_prefetch: no subshape at given index: 0, 1}}
module attributes { mhlo.cross_program_prefetches = [ #mhlo.cross_program_prefetch<parameter = 1, indices = [0,1]> ] } {
  func.func @copy(%arg0 : tuple<tensor<2x3xi32>, tensor<i32>>) -> tuple<tensor<2x3xi32>, tensor<i32>> attributes {execution_thread = "main"} {
    %0 = "mhlo.copy"(%arg0) {is_cross_program_prefetch} : (tuple<tensor<2x3xi32>, tensor<i32>>) -> (tuple<tensor<2x3xi32>, tensor<i32>>)
    return %0 : tuple<tensor<2x3xi32>, tensor<i32>>
  }
  func.func @main(%arg0 : tensor<i32>, %arg1 : tuple<tensor<2x3xi32>, tensor<i32>>) -> tuple<tensor<2x3xi32>, tensor<i32>> {
    %1 = "mhlo.async_start"(%arg1) {called_computation=@copy, execution_thread="main"} : (tuple<tensor<2x3xi32>, tensor<i32>>) -> (!mhlo.async_bundle<tuple<tuple<tensor<2x3xi32>, tensor<i32>>>, tuple<tuple<tensor<2x3xi32>, tensor<i32>>>>)
    %2 = "mhlo.async_done"(%1) {called_computation=@copy, execution_thread="main"} : (!mhlo.async_bundle<tuple<tuple<tensor<2x3xi32>, tensor<i32>>>, tuple<tuple<tensor<2x3xi32>, tensor<i32>>>>) -> (tuple<tensor<2x3xi32>, tensor<i32>>)
    return %2 : tuple<tensor<2x3xi32>, tensor<i32>>
  }
}


// -----

module attributes {
  mhlo.use_auto_spmd_partitioning = true,
  mhlo.is_dynamic = true,
  mhlo.dynamic_parameter_bindings = [
    #mhlo.dynamic_parameter_binding<
      dynamic_param_num = 0,
      dynamic_param_indices = [],
      target_param_num = 1,
      target_param_indices = [],
      target_param_dim_num = 0>] } {
  func.func @main(%a : tensor<i32>, %b : tensor<?xf32, #mhlo.type_extensions<bounds = [2]>>) -> () {
    func.return
  }
}

// CHECK-LABEL: hlo_module       {
// CHECK: dynamic_parameter_binding {
// CHECK-NEXT: entries {
// CHECK-NEXT:    target_param_num: 1
// CHECK-NEXT:  }
// CHECK: is_dynamic: true
// CHECK: use_auto_spmd_partitioning: true

// -----

// expected-error@+1 {{dynamic_parameter_binding: parameters 5 and 3 out of range. main has only 2 arguments}}
module attributes {
 mhlo.dynamic_parameter_bindings = [
   #mhlo.dynamic_parameter_binding<
     dynamic_param_num = 5,
     dynamic_param_indices = [],
     target_param_num = 3,
     target_param_indices = [],
     target_param_dim_num = 0>] } {
 func.func @main(%a : tensor<i32>, %b : tensor<?xf32, #mhlo.type_extensions<bounds = [2]>>) -> () {
   func.return
 }
}

// -----

// expected-error@+1 {{dynamic_parameter_binding: no ranked tensor type at dynamic_param_indices: 8}}
module attributes {
 mhlo.dynamic_parameter_bindings = [
   #mhlo.dynamic_parameter_binding<
     dynamic_param_num = 0,
     dynamic_param_indices = [8],
     target_param_num = 1,
     target_param_indices = [],
     target_param_dim_num = 0>] } {
 func.func @main(%a : tensor<i32>, %b : tensor<?xf32, #mhlo.type_extensions<bounds = [2]>>) -> () {
   func.return
 }
}

// -----

// expected-error@+1 {{dynamic_parameter_binding: no dimension number 1 in target subshape}}
module attributes {
 mhlo.dynamic_parameter_bindings = [
   #mhlo.dynamic_parameter_binding<
     dynamic_param_num = 0,
     dynamic_param_indices = [],
     target_param_num = 1,
     target_param_indices = [],
     target_param_dim_num = 1>] } {
 func.func @main(%a : tensor<i32>, %b : tensor<?xf32, #mhlo.type_extensions<bounds = [2]>>) -> () {
   func.return
 }
}

// -----

// expected-error@+1 {{dynamic_parameter_binding: dynamic size must be tensor<i32>}}
module attributes {
 mhlo.dynamic_parameter_bindings = [
   #mhlo.dynamic_parameter_binding<
     dynamic_param_num = 0,
     dynamic_param_indices = [],
     target_param_num = 1,
     target_param_indices = [],
     target_param_dim_num = 0>] } {
 func.func @main(%a : tensor<f32>, %b : tensor<?xf32, #mhlo.type_extensions<bounds = [2]>>) -> () {
   func.return
 }
}


// -----

// expected-error@+1 {{dynamic_parameter_binding: dimension number 0 in target subshape 'tensor<3xf32>' is not dynamic}}
module attributes {
 mhlo.dynamic_parameter_bindings = [
   #mhlo.dynamic_parameter_binding<
     dynamic_param_num = 0,
     dynamic_param_indices = [],
     target_param_num = 1,
     target_param_indices = [],
     target_param_dim_num = 0>] } {
 func.func @main(%a : tensor<i32>, %b : tensor<3xf32>) -> () {
   func.return
 }
}

// -----

module attributes { mhlo.spmd_output_sharding = "\08\03\1A\02\01\02\22\02\00\01", mhlo.spmd_parameters_shardings = ["\08\03\1A\02\01\02\22\02\00\01"]} {
  func.func @main(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = "mhlo.custom_call"(%arg0) {backend_config = "", call_target_name = "Sharding", mhlo.sharding = "\08\03\1A\02\01\02\22\02\00\01"} : (tensor<16x16xf32>) -> tensor<16x16xf32>
    func.return %0 : tensor<16x16xf32>
  }
}

// CHECK: spmd_output_sharding {
// CHECK:   type: OTHER
// CHECK:   tile_assignment_dimensions: 1
// CHECK:   tile_assignment_dimensions: 2
// CHECK:   tile_assignment_devices: 0
// CHECK:   tile_assignment_devices: 1
// CHECK: }
// CHECK: spmd_parameters_shardings {
// CHECK:   type: OTHER
// CHECK:   tile_assignment_dimensions: 1
// CHECK:   tile_assignment_dimensions: 2
// CHECK:   tile_assignment_devices: 0
// CHECK:   tile_assignment_devices: 1
// CHECK: }
// -----

module attributes { mhlo.device_assignment = #mhlo.device_assignment<replicaCount = 2, computationCount = 3, computationDevices = 0,1,0,1,2,3 > } {
  func.func @main(%arg0: tensor<i1>) -> tensor<i1> attributes { mhlo.computation_ids = dense<0> : tensor<1xi64> } {
    %0 = "mhlo.while"(%arg0) ({
      ^bb0(%arg1: tensor<i1>):
        "mhlo.return"(%arg1) : (tensor<i1>) -> ()
      }, {
      ^bb0(%arg1: tensor<i1>):
        "mhlo.return"(%arg1) : (tensor<i1>) -> ()
    }) { mhlo.computation_ids = dense<[1, 2]> : tensor<2xi64> } : (tensor<i1>) -> tensor<i1>
    func.return %0: tensor<i1>
  }
}

// CHECK: device_assignment {
// CHECK-NEXT:    replica_count: 2
// CHECK-NEXT:    computation_count: 3
// CHECK-NEXT:    computation_devices {
// CHECK-NEXT:      replica_device_ids: 0
// CHECK-NEXT:      replica_device_ids: 1
// CHECK-NEXT:    }
// CHECK-NEXT:    computation_devices {
// CHECK-NEXT:      replica_device_ids: 0
// CHECK-NEXT:      replica_device_ids: 1
// CHECK-NEXT:    }
// CHECK-NEXT:    computation_devices {
// CHECK-NEXT:      replica_device_ids: 2
// CHECK-NEXT:      replica_device_ids: 3
// CHECK-NEXT:    }

// -----

// expected-error@+1 {{device_assignment: computationCount 2 does not match the number of computations 3 in the module}}
module attributes { mhlo.device_assignment = #mhlo.device_assignment<replicaCount = 2, computationCount = 2, computationDevices = 0,0,1,3 > } {
  func.func @main(%arg0: tensor<i1>) -> tensor<i1> attributes { mhlo.computation_ids = dense<0> : tensor<1xi64> } {
    %0 = "mhlo.while"(%arg0) ({
      ^bb0(%arg1: tensor<i1>):
        "mhlo.return"(%arg1) : (tensor<i1>) -> ()
      }, {
      ^bb0(%arg1: tensor<i1>):
        "mhlo.return"(%arg1) : (tensor<i1>) -> ()
    }) { mhlo.computation_ids = dense<[1, 2]> : tensor<2xi64> } : (tensor<i1>) -> tensor<i1>
    func.return %0: tensor<i1>
  }
}

// -----

// expected-error@+1 {{device_assignment: repeated computation IDs}}
module attributes { mhlo.device_assignment = #mhlo.device_assignment<replicaCount = 2, computationCount = 2, computationDevices = 0,0,1,3 > } {
  func.func @main(%arg0: tensor<i1>) -> tensor<i1> attributes { mhlo.computation_ids = dense<0> : tensor<1xi64> } {
    %0 = "mhlo.while"(%arg0) ({
      ^bb0(%arg1: tensor<i1>):
        "mhlo.return"(%arg1) : (tensor<i1>) -> ()
      }, {
      ^bb0(%arg1: tensor<i1>):
        "mhlo.return"(%arg1) : (tensor<i1>) -> ()
    }) { mhlo.computation_ids = dense<[2, 2]> : tensor<2xi64> } : (tensor<i1>) -> tensor<i1>
    func.return %0: tensor<i1>
  }
}
