/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/llvm_gpu_backend/amdgpu_backend.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <mutex>  // NOLINT
#include <optional>
#include <string>
#include <system_error>  // NOLINT
#include <utility>
#include <variant>
#include <vector>
#include <list>

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/PassRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Scalar.h"
#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "xla/service/gpu/llvm_gpu_backend/load_ir_module.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/rocm_rocdl_path.h"
#include "xla/tsl/util/env_var.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "tsl/platform/random.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/base64.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {
namespace {

// Inline threshold value to use in LLVM AMDGPU backend.
const int kAMDGPUInlineThreshold = 0x100000;

// Gets the ROCm-Device-Libs filenames for a particular AMDGPU version.
std::vector<std::string> GetROCDLPaths(std::string gcn_arch_name,
                                       const std::string& rocdl_dir_path) {
  // AMDGPU version-neutral bitcodes.
  static std::vector<std::string>* rocdl_filenames =
      new std::vector<std::string>(
          {"opencl.bc", "ocml.bc", "ockl.bc", "oclc_finite_only_off.bc",
           "oclc_daz_opt_off.bc", "oclc_correctly_rounded_sqrt_on.bc",
           "oclc_unsafe_math_off.bc", "oclc_wavefrontsize64_on.bc",
           "oclc_abi_version_500.bc"});

  // Construct full path to ROCDL bitcode libraries.
  std::vector<std::string> result;
  result.reserve(rocdl_filenames->size() + 1);
  for (auto& filename : *rocdl_filenames) {
    result.push_back(tsl::io::JoinPath(rocdl_dir_path, filename));
  }

  // Add AMDGPU version-specific bitcodes.
  std::vector<std::string> tokens = absl::StrSplit(gcn_arch_name, ':');
  std::string amdgpu_version = gcn_arch_name;
  if (!tokens.empty() && tokens[0].size() >= 3) {
    amdgpu_version = tokens[0].substr(3);
  }
  result.push_back(tsl::io::JoinPath(
      rocdl_dir_path,
      absl::StrCat("oclc_isa_version_", amdgpu_version, ".bc")));
  return result;
}

struct HsacoCacheEntry {
  std::string hash_str;
  std::string ir;
  std::vector<uint8_t> hsaco;
};

struct HsacoCache {
 protected:
  std::list<HsacoCacheEntry> hsaco_cache_;
  std::mutex mutex_;
  int request_count_ = 0;
  int hit_count_ = 0;
  std::string hsaco_cache_dir_;

  HsacoCache() {
    auto env = tsl::Env::Default();
    (void)tsl::ReadStringFromEnvVar("TF_XLA_HSACO_CACHE_DIR", "/tmp",
                                     &hsaco_cache_dir_);
    if (hsaco_cache_dir_.empty()) {
      LOG(INFO) << "Will not cache XLA HSACOs. ";
    } else {
      if (!env->IsDirectory(hsaco_cache_dir_).ok()) {
        if(!env->CreateDir(hsaco_cache_dir_).ok()) {
          LOG(FATAL) << "Unable to create hsaco cache dir: " << hsaco_cache_dir_;
        }
      }
      LOG(INFO) << "Cache XLA HSACOs in " << hsaco_cache_dir_;
      if(hsaco_cache_dir_.back() != '/') hsaco_cache_dir_ += '/';
    }
  }

 public:
  static HsacoCache& i() {
    static HsacoCache obj;
    return obj;
  }

bool Find(const std::string& ir, const std::string& gfx, 
        std::string *hash_str, std::vector<uint8_t> *hsaco, std::string *hsaco_path) {
  std::lock_guard<std::mutex> lg(mutex_);

  llvm::SHA256 sha256;
  sha256.update(llvm::StringRef(ir));
  std::array<uint8_t, 32> lhash = sha256.final();
  // C++ strict aliasing rules allow reinterpret casting to (const) char*.
  absl::string_view hash_view(reinterpret_cast<const char*>(lhash.data()),
                              lhash.size());
  (void)tsl::Base64Encode(hash_view, hash_str);
  // VLOG(0) << "Got hashview:" << hash_str;
  *hash_str += "." + gfx;

  bool hit = false;
  for (const auto& x : hsaco_cache_) {
    if (!(x.hash_str == *hash_str && x.ir == ir)) continue;
    *hsaco = x.hsaco;
    hit = true;
    break;
  }

  *hsaco_path = hsaco_cache_dir_ + *hash_str + ".hsaco";
  if (!hit && tsl::Env::Default()->FileExists(*hsaco_path).ok()) {
      VLOG(1) << "Hsaco cache hit in file " << *hsaco_path;
      std::ifstream hsaco_file(*hsaco_path, std::ios::binary | std::ios::ate);
      std::ifstream::pos_type hsaco_file_size = hsaco_file.tellg();
      *hsaco = std::vector<uint8_t>(hsaco_file_size);
      hsaco_file.seekg(0, std::ios::beg);
      hsaco_file.read(reinterpret_cast<char*>(hsaco->data()), hsaco_file_size);
      hsaco_cache_.emplace_back(HsacoCacheEntry{*hash_str, ir, *hsaco});
      hit = true;
  }
  request_count_++;
  if (hit) hit_count_++;
  VLOG(1) << "HSACO cache: " << request_count_ << " requests, "
            << hit_count_ << " hits";
  return hit;
}

void Add(const std::string& ir, const std::string& hash_str,
                     const std::vector<uint8_t>& hsaco) {
  std::lock_guard<std::mutex> lg(mutex_);
  hsaco_cache_.emplace_back(HsacoCacheEntry{hash_str, ir, hsaco});
}

}; // HsacoCache

struct JaxPluginPaths {
  std::string bitcode_path;
  std::string lld_path;
};

JaxPluginPaths getJaxPluginPaths() {
  JaxPluginPaths paths;

  paths.bitcode_path = std::getenv("JAX_ROCM_PLUGIN_INTERNAL_BITCODE_PATH") ?: "";
  paths.lld_path = std::getenv("JAX_ROCM_PLUGIN_INTERNAL_LLD_PATH") ?: "";

  return paths;
}

// Emits the given module to HSA Code Object. target_machine is an initialized
// TargetMachine for the AMDGPU target.
absl::StatusOr<std::vector<uint8_t>> EmitModuleToHsaco(
    llvm::Module* module, llvm::TargetMachine* target_machine,
    const std::string& hsaco_path) {
  auto* env = tsl::Env::Default();
  std::vector<std::string> tempdir_vector;
  env->GetLocalTempDirectories(&tempdir_vector);
  if (tempdir_vector.empty()) {
    return xla::Internal(
        "Unable to locate a temporary directory for compile-time artifacts.");
  }
  std::string tempdir_name = tempdir_vector.front();
  VLOG(1) << "Compile-time artifacts located at: " << tempdir_name;

  bool keep_tempfiles = false;
  TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_ROCM_KEEP_XLA_TEMPFILES",
                                      /*default_val=*/false, &keep_tempfiles));
  // Prepare filenames for all stages of compilation:
  // IR, binary ISA, and HSACO.
  std::string random_number = std::to_string(tsl::random::New64());
  std::string ir_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + ".ll");
  std::string ir_path = tsl::io::JoinPath(tempdir_name, ir_filename);

  std::string ir_opt_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + "_opt.ll");
  std::string ir_opt_path = tsl::io::JoinPath(tempdir_name, ir_opt_filename);

  std::string isabin_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + ".o");
  std::string isabin_path = tsl::io::JoinPath(tempdir_name, isabin_filename);

  std::error_code ec;

  // Dump LLVM IR.
  std::unique_ptr<llvm::raw_fd_ostream> ir_fs(
      new llvm::raw_fd_ostream(ir_path, ec, llvm::sys::fs::OF_None));
  module->print(*ir_fs, nullptr);
  ir_fs->flush();

  // Emit GCN ISA binary.
  llvm::legacy::PassManager pm;
  pm.add(new llvm::TargetLibraryInfoWrapperPass(
      llvm::Triple(module->getTargetTriple())));
  llvm::SmallVector<char, 0> stream;
  llvm::raw_svector_ostream pstream(stream);
  std::unique_ptr<llvm::raw_fd_ostream> isabin_fs(
      new llvm::raw_fd_ostream(isabin_path, ec, llvm::sys::fs::OF_Text));
  module->setDataLayout(target_machine->createDataLayout());
  target_machine->addPassesToEmitFile(pm, *isabin_fs, nullptr,
                                      llvm::CodeGenFileType::ObjectFile);
  pm.run(*module);
  isabin_fs->flush();

  if (keep_tempfiles) {
    std::unique_ptr<llvm::raw_fd_ostream> ir_fs(
        new llvm::raw_fd_ostream(ir_opt_path, ec, llvm::sys::fs::OF_None));
    module->print(*ir_fs, nullptr);
    ir_fs->flush();
  }
  // Locate lld.
  std::string lld_path;
  if (std::getenv("LLVM_PATH")) {
    lld_path = tsl::io::JoinPath(std::getenv("LLVM_PATH"), "bin");
  } else {
    lld_path = tsl::io::JoinPath(tsl::RocmRoot(), "llvm/bin");
  }
  auto lld_program = llvm::sys::findProgramByName("ld.lld", {lld_path});
  if (!lld_program) {
    return xla::Internal("unable to find ld.lld in PATH: %s",
                         lld_program.getError().message());
  }
  std::vector<llvm::StringRef> lld_args{
      llvm_ir::AsStringRef("ld.lld"),    llvm_ir::AsStringRef("-flavor"),
      llvm_ir::AsStringRef("gnu"),       llvm_ir::AsStringRef("-shared"),
      llvm_ir::AsStringRef(isabin_path), llvm_ir::AsStringRef("-o"),
      llvm_ir::AsStringRef(hsaco_path),
  };

  std::string error_message;
  int lld_result =
      llvm::sys::ExecuteAndWait(*lld_program, llvm_ir::AsArrayRef(lld_args),
                                std::nullopt, {}, 0, 0, &error_message);
  if (lld_result) {
    return xla::Internal("ld.lld execute fail: %s, error code %d",
                         error_message, lld_result);
  }

  // Read HSACO.
  std::ifstream hsaco_file(hsaco_path, std::ios::binary | std::ios::ate);
  std::ifstream::pos_type hsaco_file_size = hsaco_file.tellg();

  std::vector<uint8_t> hsaco(hsaco_file_size);
  hsaco_file.seekg(0, std::ios::beg);
  hsaco_file.read(reinterpret_cast<char*>(hsaco.data()), hsaco_file_size);
  hsaco_file.close();
  if (!keep_tempfiles) {
    remove(ir_path.c_str());
    remove(isabin_path.c_str());
  }
  VLOG(1) << "Written: " << hsaco_path << " size: " << hsaco_file_size;
  return hsaco;
}

// Links ROCm-Device-Libs into the given module if the module needs it.
absl::Status LinkROCDLIfNecessary(llvm::Module* module,
                                  std::string gcn_arch_name,
                                  const std::string& rocdl_dir_path) {
  if (!CouldNeedDeviceBitcode(*module)) {
    return absl::OkStatus();
  }

  return LinkWithBitcodeVector(module,
                               GetROCDLPaths(gcn_arch_name, rocdl_dir_path));
}

absl::Status AMDGPUTargetModuleLinker(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    const std::string& device_bitcode_dir_path) {
  // Link the input module with ROCDL.

  auto compute_capability =
      std::get_if<se::RocmComputeCapability>(&gpu_version);
  if (!compute_capability) {
    return xla::Internal("Incompatible compute capability was specified.");
  }

  std::string gcn_arch_name = compute_capability->gcn_arch_name();
  TF_RETURN_IF_ERROR(
      LinkROCDLIfNecessary(module, gcn_arch_name, device_bitcode_dir_path));

  // If ftz is enabled, set it as an attribute on every function in the module.
  if (debug_options.xla_gpu_ftz()) {
    for (llvm::Function& fn : *module) {
      fn.addFnAttr("denormal-fp-math-f32", "preserve-sign");
    }
  }
  const int32_t kAbiVersion = 500;
  module->addModuleFlag(llvm::Module::Error, "amdhsa_code_object_version",
                        kAbiVersion);

  return absl::OkStatus();
}

// The following routine maps a feature token extracted from the
// hipDeviceProp_t::gcnArchName string, and maps it to a valid feature_str
// to be used for creating the AMDGPUTarget.
// This mapping is currently in a state of flux because TF XLA uses its
// own copy of LLVM, which is different from the LLVM version used by
// hipcc/runtime in the ROCm install. Ordinarily this is not a problem,
// but right now, the LLVM version used by hipcc/runtime has "targetID"
// related changes which have not yet been upstreamed (to the LLVM repo)
// When that upstreaming happens (and TF LLVM pointer moves past the
// upstream commit), the following mapping will need to change
std::string MapGCNArchNameTokenToFeatureStr(const std::string& token,
                                            const std::string& gfx) {
  if (token == "sramecc+") {
    return "+sramecc";
  } else if (token == "sramecc-") {
    if (gfx == "gfx90a" || gfx == "gfx942") return "";
    return "-sramecc";
  } else if (token == "xnack+") {
    return "+xnack";
  } else if (token == "xnack-") {
    return "-xnack";
  }
  return "";
}

std::pair<std::string, std::string> GetFeatureStrFromGCNArchName(
    const std::string& gcn_arch_name) {
  std::string feature_str;

  std::string gfx = gcn_arch_name;
  // For ROCm versions 4.0 and greater, we need to specify the correct
  // feature str, based on the underlying GPU HW to get max performance.
  std::vector<std::string> tokens = absl::StrSplit(gcn_arch_name, ':');
  std::vector<std::string> mapped_tokens;
  if (!tokens.empty()) gfx = tokens[0];
  for (auto it = tokens.begin(); it != tokens.end(); it++) {
    // Skip the first token, that is the gfxNNN str
    // The rest of the tokens are the feature/targetid strings
    if (it != tokens.begin()) {
      std::string token(*it);
      std::string mapped_token = MapGCNArchNameTokenToFeatureStr(token, gfx);
      mapped_tokens.push_back(mapped_token);
    }
  }
  feature_str = absl::StrJoin(mapped_tokens, ",");

  return std::make_pair(gfx, feature_str);
}

std::unique_ptr<llvm::TargetMachine> AMDGPUGetTargetMachine(
    llvm::Triple target_triple, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options) {
  auto compute_capability =
      std::get_if<se::RocmComputeCapability>(&gpu_version);

  std::string gcn_arch_name = compute_capability->gcn_arch_name();
  auto arch = GetFeatureStrFromGCNArchName(gcn_arch_name);
  return GetTargetMachine(std::move(target_triple), arch.first, debug_options,
                          arch.second);
}

// Returns the directory containing ROCm-Device-Libs files.
std::string GetROCDLDir(const DebugOptions& debug_options) {
  std::vector<std::string> potential_rocdl_dirs;
  const std::string& datadir = debug_options.xla_gpu_cuda_data_dir();
  if (!datadir.empty()) {
    potential_rocdl_dirs.push_back(datadir);
  }
  potential_rocdl_dirs.push_back(tsl::RocdlRoot());

  // Tries all potential ROCDL directories in the order they are inserted.
  // Returns the first directory that exists in the file system.
  for (const std::string& potential_rocdl_dir : potential_rocdl_dirs) {
    if (tsl::Env::Default()->IsDirectory(potential_rocdl_dir).ok()) {
      VLOG(2) << "Found ROCm-Device-Libs dir " << potential_rocdl_dir;
      return potential_rocdl_dir;
    }
    VLOG(2) << "Unable to find potential ROCm-Device-Libs dir "
            << potential_rocdl_dir;
  }

  // Last resort: maybe in the current folder.
  return ".";
}

void AMDGPUBackendInit(const DebugOptions& debug_options,
                       std::string& rocdl_dir_path) {
  // Initialize the AMDGPU target; it's the only target we link with, so call
  // its specific initialization functions instead of the catch-all
  // InitializeAll*.
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();

  rocdl_dir_path = GetROCDLDir(debug_options);
  llvm::PassRegistry* registry = llvm::PassRegistry::getPassRegistry();
  gpu::InitializePasses(registry);
}

}  // namespace

namespace amdgpu {

std::vector<std::string> GetAMDGPUBackendOptions(
    const DebugOptions& debug_options) {
  std::vector<std::string> backend_llvm_opts;

  // Extra backend options must go after regular backend options in order to be
  // able for the later to override the former.
  auto backend_extra_llvm_opts = llvm_ir::ExtractXlaBackendExtraOptions(
      debug_options.xla_backend_extra_options());
  backend_llvm_opts.insert(backend_llvm_opts.end(),
                           backend_extra_llvm_opts.cbegin(),
                           backend_extra_llvm_opts.cend());

  return backend_llvm_opts;
}

std::string LibDevicePath(std::string gcn_arch_name,
                          const std::string& rocdl_dir_path) {
  auto libdevice_dir_paths = GetROCDLPaths(gcn_arch_name, rocdl_dir_path);
  for (auto libdevice_dir_path : libdevice_dir_paths) {
    if (libdevice_dir_path.find("ocml.bc")) {
      return libdevice_dir_path;
    }
  }
  return "";
}

absl::StatusOr<std::vector<uint8_t>> CompileToHsaco(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    const std::string& module_config_cache_key) {
  static absl::once_flag backend_init_flag;
  // TODO(rocm) Ideally this would be refreshed if xla_gpu_cuda_data_dir
  // changes.
  static std::string rocdl_dir_path;  // NOLINT: static/global vars forbidden
  absl::call_once(backend_init_flag, AMDGPUBackendInit, debug_options,
                  rocdl_dir_path);
  auto llvm_opts = GetAMDGPUBackendOptions(debug_options);
  llvm_ir::LLVMCommandLineOptionsLock llvm_lock(llvm_opts);

  std::vector<uint8_t> hsaco;
  std::unique_ptr<llvm::TargetMachine> target_machine;
  std::string str;
  llvm::raw_string_ostream stream(str);
  stream << *module;
  // Delete the first two lines, since they usually vary even when the rest of
  // the code is the same (but verify that they are what we expect).
  if (str.size() >= 13 && str.substr(0, 13) == "; ModuleID = ") {
    auto pos = str.find('\n');
    if (pos != std::string::npos) str = str.substr(pos + 1);
  }
  if (str.size() >= 18 && str.substr(0, 18) == "source_filename = ") {
    auto pos = str.find('\n');
    if (pos != std::string::npos) str = str.substr(pos + 1);
  }
  str += module_config_cache_key;
  {
    tsl::profiler::TraceMe activity(
        [&] { return absl::StrCat("Compiling IR", module->getName().str()); },
        tsl::profiler::TraceMeLevel::kInfo);
    XLA_SCOPED_LOGGING_TIMER("Compile module " + module->getName().str());

    auto compute_capability =
        std::get_if<se::RocmComputeCapability>(&gpu_version);
    if (!compute_capability) {
      return xla::Internal("Incompatible compute capability was specified.");
    }

    std::string hash_str, hsaco_path, 
        gfx = compute_capability->gfx_version();
    if (HsacoCache::i().Find(str, gfx, &hash_str, &hsaco, &hsaco_path)) {
      VLOG(1) << "HSACO cache hit";
      return hsaco;
    }

    VLOG(1) << "HSACO cache miss";
    bool dump_lls = false;
    if (dump_lls) {
      static int hsaco_count = 0;
      std::string name = "/tmp/" + std::to_string(hsaco_count) + ".ll";
      hsaco_count++;
      std::ofstream ofs(name);
      ofs << str;
      ofs.close();
    }

    llvm::Triple default_target_triple("amdgcn--amdhsa-amdgiz");
    // Construct LLVM TargetMachine for AMDGPU.
    std::unique_ptr<llvm::TargetMachine> target_machine =
        AMDGPUGetTargetMachine(default_target_triple, gpu_version,
                               debug_options);

    // Link with ROCm-Device-Libs, and optimize the LLVM module.
    TF_RETURN_IF_ERROR(gpu::LinkAndOptimizeModule(
        module, gpu_version, debug_options, rocdl_dir_path,
        AMDGPUTargetModuleLinker, default_target_triple, target_machine.get(),
        kAMDGPUInlineThreshold));

    // Lower optimized LLVM module to HSA code object.
    TF_ASSIGN_OR_RETURN(hsaco, EmitModuleToHsaco(module, target_machine.get(),
                       hsaco_path));
    HsacoCache::i().Add(str, hash_str, hsaco);
  }
  return hsaco;
}

}  // namespace amdgpu
}  // namespace gpu
}  // namespace xla
