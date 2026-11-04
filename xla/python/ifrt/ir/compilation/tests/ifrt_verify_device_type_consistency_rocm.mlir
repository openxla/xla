// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: ifrt-opt %s -ifrt-verify-device-type-consistency='platform_names=rocm:2,cpu' -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @good_call_rocm
#sharding = #ifrt.sharding_param<2 to [0] on 2>
module @good_call_rocm {
  func.func @main() -> !ifrt.array<tensor<2xi32>, #sharding, [0,1]>
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @const_hlo() on devices [0,1]
        : () -> !ifrt.array<tensor<2xi32>, #sharding, [0,1]>
    %1, %ctrl_1 = ifrt.Call @identity(%0) on devices [0,1]
        : (!ifrt.array<tensor<2xi32>, #sharding, [0,1]>)
        -> !ifrt.array<tensor<2xi32>, #sharding, [0,1]>
    return %1 : !ifrt.array<tensor<2xi32>, #sharding, [0,1]>
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }

  func.func private @const_hlo() -> tensor<2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2xi32>
    return %0 : tensor<2xi32>
  }
}

// -----

module @rocm_and_cpu {
  func.func @main() -> !ifrt.array<tensor<2x2xi32>,
                                   #ifrt.sharding_param<2x1 to [0] on 2>,
                                   [0, 2]>
      attributes {ifrt.function} {
  // expected-error @+1 {{'ifrt.Call' op requires a single platform type. Expected platform: rocm. Actual platform of logical device 2: cpu}}
    %0, %ctrl_0 = ifrt.Call @hlo() on devices [0,2]
        : () -> !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [0,2]>
    return %0 : !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [0,2]>
  }

  func.func private @hlo() -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}
