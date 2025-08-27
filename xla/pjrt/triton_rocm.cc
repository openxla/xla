/* Copyright 2025 The OpenXLA Authors.

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

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SHA256.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "xla/debug_options_flags.h"
#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"
#include "xla/pjrt/triton.h"
#include "xla/service/gpu/llvm_gpu_backend/amdgpu_backend.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/random.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace xla::triton {

namespace {

absl::Status TritonToLLVM(
    mlir::ModuleOp module, absl::string_view arch_name, int num_warps,
    int num_ctas, int num_stages,
    mlir::triton::nvidia_gpu::ClusterInfo* out_cluster_info) {
  mlir::PassManager pm(module.getContext());
  pm.enableVerifier();
  pm.addPass(mlir::createLowerAffinePass());
  TF_RETURN_IF_ERROR(
      xla::gpu::CreateTritonPipeline(&pm, std::string(arch_name), num_warps,
                                     num_ctas, num_stages, *out_cluster_info));
  pm.addPass(mlir::createStripDebugInfoPass());
  return pm.run(module).succeeded()
             ? absl::OkStatus()
             : absl::InternalError("Failed to compile Triton to LLVM");
}

using HashType = std::array<uint8_t, 32>;
const std::string tempdir_default = "/tmp";

struct HsacoCache {
 private:
  struct Hash64 {
    size_t operator()(const HashType& s) const noexcept {
      return *reinterpret_cast<const size_t*>(s.data());
    }
  };
  absl::Mutex mutex_;
  absl::flat_hash_map<HashType, std::string, Hash64> hsaco_cache_
      ABSL_GUARDED_BY(mutex_);
  bool has_uncached_file_ ABSL_GUARDED_BY(mutex_) = false;
  std::string hsaco_cache_dir_;

 public:
  HsacoCache() {
    auto* env = tsl::Env::Default();
    std::vector<std::string> tempdirs;
    env->GetLocalTempDirectories(&tempdirs);
    if (tempdirs.empty()) {
      tempdirs.push_back(tempdir_default);
    }
    std::string tempdir = tempdirs.front();
    std::string random64 = std::to_string(tsl::random::New64());
    hsaco_cache_dir_ = tsl::io::JoinPath(tempdir, random64);
    if (env->IsDirectory(hsaco_cache_dir_).ok()) {
      hsaco_cache_dir_ = tempdir_default;
      LOG(WARNING) << "HSACO cache directory " << hsaco_cache_dir_
                   << " is in use!";
    } else {
      if (!env->RecursivelyCreateDir(hsaco_cache_dir_).ok()) {
        hsaco_cache_dir_ = tempdir_default;
        LOG(WARNING) << "Unable to create HSACO cache directory "
                     << hsaco_cache_dir_;
      }
    }
  }
  ~HsacoCache() {
    auto* env = tsl::Env::Default();
    int64_t undeleted_files = 0, undeleted_dirs = 0;
    if (has_uncached_file_) {
      for (const auto& [hash, path] : hsaco_cache_) {
        if (!env->DeleteFile(path).ok()) {
          LOG(WARNING) << "Unable to delete HSACO file " << path;
        }
      }
    }
    if (hsaco_cache_dir_ != tempdir_default) {
      if (!env->DeleteRecursively(hsaco_cache_dir_, &undeleted_files,
                                  &undeleted_dirs)
               .ok()) {
        LOG(WARNING) << "Unable to delete HSACO cache directory "
                     << hsaco_cache_dir_;
      }
    }
  }
  std::string find(const HashType& hash);
  std::string emplace(const HashType& hash, std::string& hsaco_path);
  static HsacoCache& i() {
    static HsacoCache obj_;
    return obj_;
  }
};

std::string HsacoCache::find(const HashType& hash) {
  absl::MutexLock lock(&mutex_);
  if (auto it = hsaco_cache_.find(hash); it != hsaco_cache_.end()) {
    return it->second;
  }
  return std::string();
}

std::string HsacoCache::emplace(const HashType& hash, std::string& hsaco_path) {
  auto* env = tsl::Env::Default();
  absl::string_view hex_string(reinterpret_cast<const char*>(hash.data()),
                               hash.size());
  std::string hsaco_name = absl::BytesToHexString(hex_string);
  std::string new_hsaco_path = tsl::io::JoinPath(hsaco_cache_dir_, hsaco_name);
  if (env->FileExists(new_hsaco_path).ok()) {
    new_hsaco_path = hsaco_path;
    LOG(WARNING) << "HSACO file " << new_hsaco_path << "already exists!";
  } else {
    if (!env->RenameFile(hsaco_path, new_hsaco_path).ok()) {
      if (!env->CopyFile(hsaco_path, new_hsaco_path).ok()) {
        new_hsaco_path = hsaco_path;
        LOG(WARNING) << "Unable to copy HSACO file " << hsaco_path << " to "
                     << new_hsaco_path;
      } else {
        if (!env->DeleteFile(hsaco_path).ok()) {
          LOG(WARNING) << "Unable to delete HSACO file " << hsaco_path;
        }
      }
    }
  }
  absl::MutexLock lock(&mutex_);
  hsaco_cache_.emplace(hash, new_hsaco_path);
  if (new_hsaco_path == hsaco_path) {
    has_uncached_file_ = true;
  }
  return new_hsaco_path;
}

class raw_sha256_ostream : public llvm::raw_ostream {
 private:
  llvm::SHA256 state_;
  uint64_t total_bytes_ = 0;

  void write_impl(const char* ptr, size_t size) override {
    state_.update(llvm::ArrayRef(reinterpret_cast<const uint8_t*>(ptr), size));
    total_bytes_ += size;
  }

 public:
  HashType sha256() {
    flush();
    return state_.final();
  }
  size_t current_pos() const override { return total_bytes_; }
};

absl::StatusOr<std::string> LLVMToHSACO(mlir::ModuleOp module,
                                        absl::string_view arch_name,
                                        int num_warps) {
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);
  module.getContext()->appendDialectRegistry(registry);

  auto rocm_cc = stream_executor::RocmComputeCapability(std::string(arch_name));
  int threads_per_block = num_warps * (rocm_cc.gfx9_mi100_or_later() ? 64 : 32);

  llvm::LLVMContext llvm_context;
  std::unique_ptr<llvm::Module> llvm_module =
      mlir::translateModuleToLLVMIR(module, llvm_context);
  if (!llvm_module) {
    return absl::InternalError("Failed to emit LLVM IR");
  }
  llvm_module->setTargetTriple(llvm::Triple(xla::gpu::amdgpu::TargetTriple()));
  llvm_module->setDataLayout(xla::gpu::amdgpu::DataLayout());
  for (llvm::Function& func : *llvm_module) {
    if (!func.isDeclaration() && func.hasExternalLinkage()) {
      func.setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
      func.addFnAttr("uniform-work-group-size", "true");
      func.addFnAttr(
          "amdgpu-flat-work-group-size",
          absl::StrJoin({threads_per_block, threads_per_block}, ","));
    }
  }

  auto debug_opts = xla::DefaultDebugOptionsIgnoringFlags();
  auto llvm_opts = xla::gpu::amdgpu::GetAMDGPUBackendOptions(debug_opts);
  llvm_ir::LLVMCommandLineOptionsLock llvm_lock(llvm_opts);

  raw_sha256_ostream os;
  llvm::WriteBitcodeToFile(*llvm_module, os);
  os << arch_name;
  auto opt_level = debug_opts.xla_backend_optimization_level();
  os.write(reinterpret_cast<const char*>(&opt_level), sizeof(opt_level));
  HashType hash = os.sha256();

  auto& cache = HsacoCache::i();
  std::string hsaco_path = cache.find(hash);
  if (!hsaco_path.empty()) {
    return hsaco_path;
  }
  TF_ASSIGN_OR_RETURN(
      hsaco_path, xla::gpu::amdgpu::CompileToHsaco(llvm_module.get(), rocm_cc,
                                                   debug_opts, false));
  return cache.emplace(hash, hsaco_path);
}

}  // namespace

absl::StatusOr<CompilationResult> Compile(absl::string_view module,
                                          absl::string_view arch_name,
                                          int num_warps, int num_ctas,
                                          int num_stages) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::triton::TritonDialect,
                      mlir::triton::gpu::TritonGPUDialect,
                      mlir::arith::ArithDialect, mlir::affine::AffineDialect,
                      mlir::LLVM::LLVMDialect, mlir::func::FuncDialect,
                      mlir::tensor::TensorDialect>();
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  context.appendDialectRegistry(registry);

  mlir::OwningOpRef<mlir::ModuleOp> module_op =
      mlir::parseSourceString<mlir::ModuleOp>(module, &context);
  if (!module_op) {
    return absl::InvalidArgumentError("Failed to parse Triton module");
  }

  mlir::PassManager pm((*module_op)->getContext());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (!pm.run(*module_op).succeeded()) {
    return absl::InvalidArgumentError("Failed to canonicalize Triton module");
  }

  mlir::triton::nvidia_gpu::ClusterInfo cluster_info;
  TF_RETURN_IF_ERROR(TritonToLLVM(*module_op, arch_name, num_warps, num_ctas,
                                  num_stages, &cluster_info));

  auto shared_mem_bytes =
      (*module_op)->getAttrOfType<mlir::IntegerAttr>("ttg.shared").getInt();

  TF_ASSIGN_OR_RETURN(auto hsaco_path,
                      LLVMToHSACO(*module_op, arch_name, num_warps));

  return CompilationResult{
      hsaco_path,
      shared_mem_bytes,
      cluster_info.clusterDimX,
      cluster_info.clusterDimY,
      cluster_info.clusterDimZ,
  };
}

}  // namespace xla::triton
