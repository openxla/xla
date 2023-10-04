# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from absl.testing import absltest
from jax.lib import xla_client as xc
import jax.numpy as jnp
from mlir.dialects import func
import mlir.ir as mlir

from xla.python.ifrt.ir.python import ifrt_ir


class IfrtIrTest(absltest.TestCase):

  def test_reshard(self):
    with mlir.Context() as ctx, mlir.Location.unknown() as loc:
      ifrt_ir.load_ifrt_ir_dialect(ctx)
      module = mlir.Module.create()
      f32_tensor_type = mlir.RankedTensorType.get((4,), mlir.F32Type.get())
      input_array_type = ifrt_ir.ArrayType.get(
          f32_tensor_type,
          ifrt_ir.ShardingParam.from_hlo_sharding(
              xc.HloSharding.replicate(),
              xc.Shape.array_shape(xc.dtype_to_etype(jnp.float32), (4,)),
              [2],
          ),
          ifrt_ir.DevicesAttr.get([0, 1]),
          ctx,
      )
      output_array_type = ifrt_ir.ArrayType.get(
          f32_tensor_type,
          ifrt_ir.ShardingParam.from_hlo_sharding(
              xc.HloSharding.replicate(),
              xc.Shape.array_shape(xc.dtype_to_etype(jnp.float32), (4,)),
              [2],
          ),
          ifrt_ir.DevicesAttr.get([2, 3]),
          ctx,
      )
      with mlir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(input_array_type, results=[output_array_type])
        def f(tensor):  # pylint: disable=unused-variable
          resharded = ifrt_ir.ReshardOp(
              output_array_type, tensor, control_inputs=[], loc=loc
          )
          func.ReturnOp([resharded.result])

      self.assertRegex(
          str(module),
          r"ifrt.Reshard.*ifrt\.array<tensor<4xf32>.*\[0, 1\]>"
          r".*ifrt\.array<tensor<4xf32>.*\[2, 3\]>",
      )

  def test_call(self):
    with mlir.Context() as ctx, mlir.Location.unknown() as loc:
      ifrt_ir.load_ifrt_ir_dialect(ctx)
      module = mlir.Module.create()
      f32_tensor_type = mlir.RankedTensorType.get((4,), mlir.F32Type.get())
      array_type = ifrt_ir.ArrayType.get(
          f32_tensor_type,
          ifrt_ir.ShardingParam.from_hlo_sharding(
              xc.HloSharding.replicate(),
              xc.Shape.array_shape(xc.dtype_to_etype(jnp.float32), (4,)),
              [2],
          ),
          ifrt_ir.DevicesAttr.get([0, 1]),
          ctx,
      )
      with mlir.InsertionPoint(module.body):
        @func.FuncOp.from_py_func(array_type)
        def unary_return(a):  # pylint: disable=unused-variable
          return a

        @func.FuncOp.from_py_func(array_type, results=[array_type])
        def f(tensor):  # pylint: disable=unused-variable
          called = ifrt_ir.CallOp(
              [array_type],
              ifrt_ir.ControlType.get(),
              [tensor],
              [],
              mlir.FlatSymbolRefAttr.get("unary_return"),
              ifrt_ir.DevicesAttr.get([2, 3]),
              loc=loc,
          )
          func.ReturnOp(called.outputs)

      self.assertRegex(
          str(module),
          r"ifrt.Call.*callee = @unary_return.*ifrt<devices\[2, 3\]>.*->"
          r" \(!ifrt.array<tensor<4xf32>.*\[0, 1\]>",
      )


if __name__ == "__main__":
  absltest.main()
