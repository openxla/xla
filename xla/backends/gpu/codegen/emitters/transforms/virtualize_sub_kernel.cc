/* Copyright 2026 The OpenXLA Authors.

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
#include <algorithm>
#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"  // IWYU pragma: keep
#include "xla/util.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_VIRTUALIZESUBKERNELPASS
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h.inc"

namespace {

constexpr llvm::StringRef kGlobalSmemName = "global_smem";
constexpr int64_t kSharedMemoryAlignment = 16;
constexpr unsigned kSharedMemoryAddrSpace = 3;

void EnsureExternalGlobalSmem(mlir::ModuleOp module) {
  if (auto existing =
          module.lookupSymbol<mlir::LLVM::GlobalOp>(kGlobalSmemName)) {
    existing.setAlignment(kSharedMemoryAlignment);
    return;
  }
  mlir::ImplicitLocOpBuilder b(module.getLoc(), module.getContext());
  b.setInsertionPointToStart(module.getBody());
  auto array_ty = mlir::LLVM::LLVMArrayType::get(b.getI8Type(), 0);
  mlir::LLVM::GlobalOp::create(b, array_ty, /*isConstant=*/false,
                               mlir::LLVM::Linkage::External,
                               std::string(kGlobalSmemName), mlir::Attribute{},
                               /*alignment=*/kSharedMemoryAlignment,
                               /*addrSpace=*/kSharedMemoryAddrSpace);
}

int64_t PackSubKernelSharedMemory(mlir::ModuleOp sub_module) {
  llvm::SmallVector<mlir::LLVM::GlobalOp> shmem_globals;
  for (auto global : sub_module.getOps<mlir::LLVM::GlobalOp>()) {
    if (global.getAddrSpace() == kSharedMemoryAddrSpace &&
        global.getLinkage() != mlir::LLVM::Linkage::External &&
        global.getSymName() != kGlobalSmemName) {
      shmem_globals.push_back(global);
    }
  }
  if (shmem_globals.empty()) {
    return 0;
  }

  EnsureExternalGlobalSmem(sub_module);

  mlir::DataLayout data_layout(sub_module);

  absl::flat_hash_map<std::string, int64_t> global_offsets;
  int64_t total_bytes = 0;
  for (mlir::LLVM::GlobalOp global : shmem_globals) {
    total_bytes = RoundUpTo<int64_t>(total_bytes, kSharedMemoryAlignment);
    global_offsets[global.getSymName().str()] = total_bytes;
    int64_t size_bytes = static_cast<int64_t>(
        data_layout.getTypeSize(global.getGlobalType()).getFixedValue());
    total_bytes += size_bytes;
  }
  total_bytes = RoundUpTo<int64_t>(total_bytes, kSharedMemoryAlignment);

  llvm::SmallVector<mlir::LLVM::AddressOfOp> addr_ops;
  sub_module.walk([&](mlir::LLVM::AddressOfOp addr_op) {
    if (global_offsets.contains(addr_op.getGlobalName().str())) {
      addr_ops.push_back(addr_op);
    }
  });

  for (mlir::LLVM::AddressOfOp addr_op : addr_ops) {
    int64_t byte_offset = global_offsets.at(addr_op.getGlobalName().str());
    mlir::ImplicitLocOpBuilder b(addr_op.getLoc(), addr_op);
    mlir::Type ptr_3_ty = addr_op.getRes().getType();
    auto base = mlir::LLVM::AddressOfOp::create(b, ptr_3_ty, kGlobalSmemName);
    mlir::Value replacement = base.getResult();
    if (byte_offset > 0) {
      auto gep =
          mlir::LLVM::GEPOp::create(b, ptr_3_ty, b.getI8Type(), base,
                                    llvm::SmallVector<mlir::LLVM::GEPArg>{
                                        static_cast<int32_t>(byte_offset)});
      replacement = gep.getResult();
    }
    addr_op.replaceAllUsesWith(replacement);
    addr_op.erase();
  }

  for (mlir::LLVM::GlobalOp global : shmem_globals) {
    global.erase();
  }

  return total_bytes;
}

// Removes leftover Triton metadata from the functions of `module`: `ttg.*`
// function attributes (e.g. the scratch memory sizes the AMD backend attaches
// to the kernel) and `tt.*` argument attributes (e.g. `tt.divisibility`).
// Neither dialect has an LLVM translation interface, so leaving them behind
// makes translating the launcher module to LLVM IR fail.
void StripTritonAttrs(mlir::ModuleOp module) {
  module.walk([](mlir::LLVM::LLVMFuncOp func) {
    llvm::SmallVector<llvm::StringRef> names_to_erase;
    for (mlir::NamedAttribute attr : func->getDiscardableAttrs()) {
      if (attr.getName().getValue().starts_with("ttg.")) {
        names_to_erase.push_back(attr.getName().getValue());
      }
    }
    for (llvm::StringRef name : names_to_erase) {
      func->removeAttr(name);
    }
    for (unsigned idx = 0; idx < func.getNumArguments(); ++idx) {
      llvm::SmallVector<llvm::StringRef> arg_names_to_erase;
      if (mlir::DictionaryAttr arg_attrs = func.getArgAttrDict(idx)) {
        for (mlir::NamedAttribute attr : arg_attrs) {
          if (attr.getName().getValue().starts_with("tt.")) {
            arg_names_to_erase.push_back(attr.getName().getValue());
          }
        }
      }
      for (llvm::StringRef name : arg_names_to_erase) {
        func.removeArgAttr(idx, name);
      }
    }
  });
}

void ReplaceAllAndErase(mlir::ImplicitLocOpBuilder& b,
                        llvm::ArrayRef<mlir::Operation*> ops, mlir::Value val) {
  for (mlir::Operation* op : ops) {
    if (b.getInsertionPoint() == op->getIterator()) {
      b.setInsertionPointAfter(op);
    }
    op->replaceAllUsesWith(mlir::ValueRange{val});
    op->erase();
  }
}

mlir::Value CreateI32Const(mlir::ImplicitLocOpBuilder& b, int64_t val) {
  return mlir::LLVM::ConstantOp::create(
      b, b.getI32Type(), b.getI32IntegerAttr(static_cast<int32_t>(val)));
}

// Collects the `OpX`/`OpY`/`OpZ` ops of `func` into `x_ops`/`y_ops`/`z_ops`,
// skipping `exclude`. Matching ops in other functions of the module are an
// error, except for already-virtualized 1D thread ID reads in nested
// sub-kernels when `allow_nested_thread_id_x` is set.
template <typename OpX, typename OpY, typename OpZ>
mlir::LogicalResult CollectDimensionalOps(
    mlir::LLVM::LLVMFuncOp func, mlir::Operation* exclude,
    bool allow_nested_thread_id_x,
    llvm::SmallVectorImpl<mlir::Operation*>& x_ops,
    llvm::SmallVectorImpl<mlir::Operation*>& y_ops,
    llvm::SmallVectorImpl<mlir::Operation*>& z_ops) {
  auto module = func->getParentOfType<mlir::ModuleOp>();
  mlir::WalkResult walk_res = module.walk([&](mlir::Operation* op) {
    if (op == exclude) {
      return mlir::WalkResult::advance();
    }
    if (!mlir::isa<OpX, OpY, OpZ>(op)) {
      return mlir::WalkResult::advance();
    }
    if (op->getParentOfType<mlir::LLVM::LLVMFuncOp>() != func) {
      // Nested sub-kernels (e.g. phases of a `PhasedBlockClusterFusion`) have
      // already been virtualized and may contain 1D `nvvm.read.ptx.sreg.tid.x`
      // (`rocdl.workitem.id.x`) reads, which remain valid when the outer
      // sub-kernel is also 1D in threads.
      if (mlir::isa<mlir::NVVM::ThreadIdXOp, mlir::ROCDL::ThreadIdXOp>(op) &&
          allow_nested_thread_id_x) {
        return mlir::WalkResult::advance();
      }
      op->emitError(
          "Hardware ID and dimension ops in non-entry functions are not "
          "supported in horizontal fusion.");
      return mlir::WalkResult::interrupt();
    }
    if (mlir::isa<OpX>(op)) {
      x_ops.push_back(op);
    } else if (mlir::isa<OpY>(op)) {
      y_ops.push_back(op);
    } else {
      z_ops.push_back(op);
    }
    return mlir::WalkResult::advance();
  });
  return mlir::failure(walk_res.wasInterrupted());
}

// The static grid or thread-block dimensions of a sub-kernel, together with
// their materialized `i32` constants. The constants are shared between
// dimension-read replacement and ID delinearization.
struct SubKernelDims {
  SubKernelDims(mlir::ImplicitLocOpBuilder& b, int64_t x, int64_t y, int64_t z)
      : x(x),
        y(y),
        z(z),
        x_const(CreateI32Const(b, x)),
        y_const(CreateI32Const(b, y)),
        z_const(CreateI32Const(b, z)) {}

  // True if the sub-kernel is 1D in this set of dimensions.
  bool is_1d() const { return y == 1 && z == 1; }

  int64_t x;
  int64_t y;
  int64_t z;
  mlir::Value x_const;
  mlir::Value y_const;
  mlir::Value z_const;
};

// Replaces the grid/block dimension reads of `func` with `dims`. The ROCDL
// dialect has no dimension-read ops (`gpu.grid_dim`/`gpu.block_dim` lower to
// `__ockl_*` calls), so this only applies to NVVM.
template <typename DimXOp, typename DimYOp, typename DimZOp>
mlir::LogicalResult VirtualizeDims(mlir::LLVM::LLVMFuncOp func,
                                   mlir::ImplicitLocOpBuilder& b,
                                   const SubKernelDims& dims) {
  llvm::SmallVector<mlir::Operation*> dim_x_ops;
  llvm::SmallVector<mlir::Operation*> dim_y_ops;
  llvm::SmallVector<mlir::Operation*> dim_z_ops;
  if (mlir::failed(CollectDimensionalOps<DimXOp, DimYOp, DimZOp>(
          func, /*exclude=*/nullptr,
          /*allow_nested_thread_id_x=*/dims.is_1d(), dim_x_ops, dim_y_ops,
          dim_z_ops))) {
    return mlir::failure();
  }
  ReplaceAllAndErase(b, dim_x_ops, dims.x_const);
  ReplaceAllAndErase(b, dim_y_ops, dims.y_const);
  ReplaceAllAndErase(b, dim_z_ops, dims.z_const);
  return mlir::success();
}

// Replaces the hardware ID reads of `func` with the coordinates obtained by
// delinearizing `linear_id` over `dims`.
template <typename IdXOp, typename IdYOp, typename IdZOp>
mlir::LogicalResult VirtualizeIds(mlir::LLVM::LLVMFuncOp func,
                                  mlir::ImplicitLocOpBuilder& b,
                                  const SubKernelDims& dims,
                                  mlir::Value linear_id) {
  llvm::SmallVector<mlir::Operation*> x_ops;
  llvm::SmallVector<mlir::Operation*> y_ops;
  llvm::SmallVector<mlir::Operation*> z_ops;
  if (mlir::failed(CollectDimensionalOps<IdXOp, IdYOp, IdZOp>(
          func, /*exclude=*/linear_id.getDefiningOp(),
          /*allow_nested_thread_id_x=*/dims.is_1d(), x_ops, y_ops, z_ops))) {
    return mlir::failure();
  }

  if (x_ops.empty() && y_ops.empty() && z_ops.empty()) {
    return mlir::success();
  }

  mlir::Value zero = CreateI32Const(b, 0);
  mlir::Value id_x = linear_id;
  mlir::Value id_y = zero;
  mlir::Value id_z = zero;
  if (dims.y > 1 || dims.z > 1) {
    id_x = mlir::LLVM::URemOp::create(b, linear_id, dims.x_const);
    mlir::Value id_yz = mlir::LLVM::UDivOp::create(b, linear_id, dims.x_const);
    id_y = id_yz;
    if (dims.z > 1) {
      id_y = mlir::LLVM::URemOp::create(b, id_yz, dims.y_const);
      id_z = mlir::LLVM::UDivOp::create(b, id_yz, dims.y_const);
    }
  }
  ReplaceAllAndErase(b, x_ops, id_x);
  ReplaceAllAndErase(b, y_ops, id_y);
  ReplaceAllAndErase(b, z_ops, id_z);
  return mlir::success();
}

mlir::LogicalResult RewritePartialCtaBarriers(mlir::ModuleOp module,
                                              mlir::ImplicitLocOpBuilder& b,
                                              int64_t sub_threads,
                                              int64_t max_threads,
                                              int64_t warp_size) {
  if (sub_threads >= max_threads) {
    return mlir::success();
  }

  llvm::SmallVector<mlir::NVVM::BarrierOp> barrier_ops;
  module.walk([&](mlir::NVVM::BarrierOp bar) {
    if (!bar.getNumberOfThreads()) {
      barrier_ops.push_back(bar);
    }
  });
  if (barrier_ops.empty()) {
    return mlir::success();
  }

  if (warp_size > 0 && sub_threads % warp_size != 0) {
    return module.emitError(absl::StrCat(
        "Sub-kernel uses CTA barriers with ", sub_threads,
        " threads (< max_threads=", max_threads,
        "), which is not a multiple of warp size (", warp_size, ")."));
  }

  mlir::OpBuilder::InsertionGuard guard(b);
  for (mlir::NVVM::BarrierOp bar : barrier_ops) {
    b.setInsertionPoint(bar);
    mlir::Value bar_id = bar.getBarrierId();
    if (!bar_id) {
      bar_id = CreateI32Const(b, 0);
    }
    mlir::Value num_threads_val = CreateI32Const(b, sub_threads);
    mlir::NVVM::BarrierOp::create(b, bar.getLoc(), bar_id, num_threads_val,
                                  b.getBoolAttr(false));
    bar.erase();
  }
  return mlir::success();
}

mlir::LogicalResult StripTrailingScratchArgs(mlir::LLVM::LLVMFuncOp func,
                                             mlir::ImplicitLocOpBuilder& b,
                                             int64_t expected_num_tensor_args) {
  if (expected_num_tensor_args < 0) {
    return mlir::success();
  }
  if (static_cast<int64_t>(func.getNumArguments()) < expected_num_tensor_args) {
    return func.emitError(
        absl::StrCat("Expected at least ", expected_num_tensor_args,
                     " arguments on sub-kernel ", func.getSymName().str(),
                     ", got ", func.getNumArguments()));
  }
  auto num_tensor_args = static_cast<unsigned>(expected_num_tensor_args);
  for (unsigned i = 0; i < num_tensor_args; ++i) {
    if (func.getArgAttr(i, "tt.nv_tma_desc") != nullptr) {
      return func.emitError(
          "Triton sub-kernels with TMA descriptor arguments are not "
          "supported in horizontal fusion.");
    }
  }
  if (func.getNumArguments() <= num_tensor_args) {
    return mlir::success();
  }
  llvm::BitVector args_to_erase(func.getNumArguments(), false);
  for (unsigned idx = num_tensor_args; idx < func.getNumArguments(); ++idx) {
    mlir::BlockArgument scratch_arg = func.getArgument(idx);
    if (!scratch_arg.use_empty()) {
      auto null_ptr = mlir::LLVM::ZeroOp::create(b, scratch_arg.getType());
      scratch_arg.replaceAllUsesWith(null_ptr.getResult());
    }
    args_to_erase.set(idx);
  }
  if (mlir::failed(func.eraseArguments(args_to_erase))) {
    return func.emitError("Failed to erase trailing scratch arguments.");
  }
  return mlir::success();
}

class VirtualizeSubKernelPass
    : public impl::VirtualizeSubKernelPassBase<VirtualizeSubKernelPass> {
 public:
  using impl::VirtualizeSubKernelPassBase<
      VirtualizeSubKernelPass>::VirtualizeSubKernelPassBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    if (auto scratch_attr = module->getAttrOfType<mlir::IntegerAttr>(
            "ttg.global_scratch_memory_size");
        scratch_attr && scratch_attr.getInt() > 0) {
      module.emitError(
          "Triton sub-kernels requiring global scratch memory are not "
          "supported in horizontal fusion.");
      signalPassFailure();
      return;
    }

    auto func =
        module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(entry_function_name_);
    if (func == nullptr || func.isExternal()) {
      module.emitError(absl::StrCat("Sub-kernel entry function '",
                                    entry_function_name_, "' not found."));
      signalPassFailure();
      return;
    }

    mlir::Block& entry_block = func.getBody().front();
    mlir::ImplicitLocOpBuilder b(func.getLoc(), &getContext());
    b.setInsertionPointToStart(&entry_block);

    // 1. Strip trailing scratch pointer arguments when requested.
    if (mlir::failed(
            StripTrailingScratchArgs(func, b, expected_num_tensor_args_))) {
      signalPassFailure();
      return;
    }

    // 2. Append `%rel_bid_i32 : i32` parameter to `func`.
    mlir::Value rel_bid_i32 =
        entry_block.addArgument(b.getI32Type(), b.getLoc());
    llvm::SmallVector<mlir::Type> new_param_types(
        func.getArgumentTypes().begin(), func.getArgumentTypes().end());
    new_param_types.push_back(b.getI32Type());
    func.setFunctionType(mlir::LLVM::LLVMFunctionType::get(
        func.getFunctionType().getReturnType(), new_param_types,
        /*isVarArg=*/false));
    if (mlir::ArrayAttr arg_attrs = func.getArgAttrsAttr()) {
      llvm::SmallVector<mlir::Attribute> new_arg_attrs(arg_attrs.begin(),
                                                       arg_attrs.end());
      new_arg_attrs.push_back(b.getDictionaryAttr({}));
      func.setArgAttrsAttr(b.getArrayAttr(new_arg_attrs));
    }

    // 3. Virtualize block IDs and grid dims. ROCDL has no grid dimension ops.
    SubKernelDims grid_dims(b, block_dim_x_, block_dim_y_, block_dim_z_);
    if (!is_amd_gpu_ &&
        mlir::failed(
            VirtualizeDims<mlir::NVVM::GridDimXOp, mlir::NVVM::GridDimYOp,
                           mlir::NVVM::GridDimZOp>(func, b, grid_dims))) {
      signalPassFailure();
      return;
    }
    if (mlir::failed(
            is_amd_gpu_
                ? VirtualizeIds<mlir::ROCDL::BlockIdXOp,
                                mlir::ROCDL::BlockIdYOp,
                                mlir::ROCDL::BlockIdZOp>(func, b, grid_dims,
                                                         rel_bid_i32)
                : VirtualizeIds<mlir::NVVM::BlockIdXOp, mlir::NVVM::BlockIdYOp,
                                mlir::NVVM::BlockIdZOp>(func, b, grid_dims,
                                                        rel_bid_i32))) {
      signalPassFailure();
      return;
    }

    // 4. Virtualize thread IDs and block dims. ROCDL has no block dimension
    // ops.
    mlir::Value raw_tid;
    if (is_amd_gpu_) {
      raw_tid = mlir::ROCDL::ThreadIdXOp::create(b, b.getI32Type());
    } else {
      raw_tid = mlir::NVVM::ThreadIdXOp::create(b, b.getI32Type());
    }
    SubKernelDims thread_dims(b, thread_dim_x_, thread_dim_y_, thread_dim_z_);
    if (!is_amd_gpu_ &&
        mlir::failed(
            VirtualizeDims<mlir::NVVM::BlockDimXOp, mlir::NVVM::BlockDimYOp,
                           mlir::NVVM::BlockDimZOp>(func, b, thread_dims))) {
      signalPassFailure();
      return;
    }
    if (mlir::failed(is_amd_gpu_ ? VirtualizeIds<mlir::ROCDL::ThreadIdXOp,
                                                 mlir::ROCDL::ThreadIdYOp,
                                                 mlir::ROCDL::ThreadIdZOp>(
                                       func, b, thread_dims, raw_tid)
                                 : VirtualizeIds<mlir::NVVM::ThreadIdXOp,
                                                 mlir::NVVM::ThreadIdYOp,
                                                 mlir::NVVM::ThreadIdZOp>(
                                       func, b, thread_dims, raw_tid))) {
      signalPassFailure();
      return;
    }

    // 5. Rewrite partial-CTA barriers when `sub_threads < max_threads_`.
    int64_t sub_threads = thread_dim_x_ * thread_dim_y_ * thread_dim_z_;
    if (mlir::failed(RewritePartialCtaBarriers(module, b, sub_threads,
                                               max_threads_, warp_size_))) {
      signalPassFailure();
      return;
    }

    // 6. Pack shared memory globals and record `"xla.shared_memory_bytes"`.
    int64_t packed_bytes = PackSubKernelSharedMemory(module);
    int64_t ttg_shared_bytes = 0;
    if (auto ttg_shared =
            module->getAttrOfType<mlir::IntegerAttr>("ttg.shared")) {
      ttg_shared_bytes = ttg_shared.getInt();
    }
    int64_t existing_shmem_bytes = 0;
    if (auto existing_shmem = module->getAttrOfType<mlir::IntegerAttr>(
            "xla.shared_memory_bytes")) {
      existing_shmem_bytes = existing_shmem.getInt();
    }
    int64_t shmem_bytes =
        std::max({packed_bytes, ttg_shared_bytes, existing_shmem_bytes});
    module->setAttr("xla.shared_memory_bytes",
                    b.getI64IntegerAttr(shmem_bytes));

    func->removeAttr("xla.entry");
    func->removeAttr("nvvm.kernel");
    func->removeAttr("nvvm.maxntid");
    func->removeAttr("nvvm.reqntid");
    func->removeAttr("nvvm.cluster_dim");
    func->removeAttr("rocdl.kernel");
    func->removeAttr("rocdl.flat_work_group_size");

    // Drop leftover Triton metadata, which is meaningless once the sub-kernel
    // is spliced into the non-Triton launcher module.
    StripTritonAttrs(module);
  }
};

}  // namespace
}  // namespace gpu
}  // namespace xla
