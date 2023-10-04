/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir-c/IR.h"  // from @llvm-project
#include "mlir/Bindings/Python/PybindAdaptors.h"  // from @llvm-project
#include "mlir/CAPI/IR.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/support/sharding_conversions.h"
#include "xla/python/status_casters.h"

namespace xla::ifrt {
namespace {

absl::Status LoadIfrtIrDialect(MlirContext py_context) {
  mlir::MLIRContext* context = unwrap(py_context);
  context->loadDialect<IfrtDialect>();
  return absl::OkStatus();
}

PYBIND11_MODULE(ifrt_ir_support, m) {
  m.def("load_ifrt_ir_dialect", xla::ThrowIfErrorWrapper(LoadIfrtIrDialect));

  py::class_<ShardingParam::MinorToMajor> minor_to_major(m, "MinorToMajor");
  minor_to_major
      .def(py::init([](const std::vector<int>& permutation,
                       const std::vector<int>& axis_sizes) {
        auto minor_to_major = std::make_unique<ShardingParam::MinorToMajor>();
        minor_to_major->permutation =
            llvm::SmallVector<int, 4>(permutation.size());
        minor_to_major->permutation.assign(permutation.begin(),
                                           permutation.end());
        minor_to_major->axis_sizes =
            llvm::SmallVector<int, 4>(axis_sizes.size());
        minor_to_major->axis_sizes.assign(axis_sizes.begin(), axis_sizes.end());
        return minor_to_major;
      }))
      .def("__eq__",
           [](const ShardingParam::MinorToMajor& a,
              const ShardingParam::MinorToMajor& b) { return a == b; });

  py::class_<ShardingParam> sharding_param(m, "ShardingParam");
  sharding_param
      .def(py::init([](const std::vector<int64_t>& dim_shards,
                       const ShardingParam::MinorToMajor& minor_to_major) {
        return std::make_unique<ShardingParam>(dim_shards, minor_to_major);
      }))
      .def_property_readonly(
          "dim_shards",
          [](const ShardingParam& sharding_param) {
            return std::vector<int64_t>(sharding_param.dim_shards());
          })
      .def_property_readonly("minor_to_major", &ShardingParam::minor_to_major)
      .def_static("from_hlo_sharding",
                  xla::ValueOrThrowWrapper(support::ToShardingParam))
      .def("__eq__", [](const ShardingParam& a,
                        const ShardingParam& b) { return a == b; })
      .def("__ne__", [](const ShardingParam& a,
                        const ShardingParam& b) { return a != b; })
      .def("__hash__", [](const ShardingParam& sharding_param) {
        return static_cast<size_t>(sharding_param.hash_value());
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "DevicesAttr",
      [](MlirAttribute attr) {
        return unwrap(attr).isa<xla::ifrt::IfrtDevicesAttr>();
      })
      .def_classmethod(
          "get",
          [](const py::object& cls, const std::vector<int>& ids,
             MlirContext ctx) {
            return cls(wrap(xla::ifrt::IfrtDevicesAttr::get(unwrap(ctx), ids)));
          },
          py::arg("cls"), py::arg("ids"), py::arg("context") = py::none(),
          "Creates a 'DevicesAttr' type.");

  mlir::python::adaptors::mlir_type_subclass(
      m, "ControlType",
      [](MlirType type) {
        return unwrap(type).isa<xla::ifrt::IfrtControlType>();
      })
      .def_classmethod(
          "get",
          [](const py::object& cls, MlirContext ctx) {
            return cls(wrap(xla::ifrt::IfrtControlType::get(unwrap(ctx))));
          },
          py::arg("cls"), py::arg("context") = py::none(),
          "Creates a 'ControlType' type.");

  mlir::python::adaptors::mlir_type_subclass(
      m, "ArrayType",
      [](MlirType type) {
        return unwrap(type).isa<xla::ifrt::IfrtArrayType>();
      })
      .def_classmethod(
          "get",
          [](const py::object& cls, MlirType shape,
             const ShardingParam& sharding, MlirAttribute devices,
             MlirContext ctx) {
            std::vector<int> ids;
            auto ranked_type = unwrap(shape).cast<mlir::RankedTensorType>();
            auto devices_attr = unwrap(devices).cast<IfrtDevicesAttr>();
            return cls(wrap(xla::ifrt::IfrtArrayType::get(
                unwrap(ctx), ranked_type, sharding, devices_attr)));
          },
          py::arg("cls"), py::arg("shape"), py::arg("sharding"),
          py::arg("devices"), py::arg("context") = py::none(),
          "Creates an 'ArrayType' type.");
}

}  // namespace
}  // namespace xla::ifrt
