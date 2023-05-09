// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

func.func @scalar_tensor_sharding(
    %arg0: !ifrt.array<tensor<i32>, to [0] on 2, [0,1]>) {
  %0 = "ifrt.Reshard"(%arg0)
      : (!ifrt.array<tensor<i32>, to [0] on 2, [0,1]>)
      -> !ifrt.array<tensor<i32>, to [0] on 4, [0,1,2,3]>
  return
}
