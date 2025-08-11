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
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/Driver.h"
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
#include "tsl/profiler/lib/traceme.h"

LLD_HAS_DRIVER(elf)

namespace xla {
namespace gpu {
namespace {

// Inline threshold value to use in LLVM AMDGPU backend.
const int kAMDGPUInlineThreshold = 0x100000;
const int32_t kAMDGPUAbiVersion = 500;

// Gets the ROCm-Device-Libs filenames for a particular AMDGPU version.
std::vector<std::string> GetROCDLPaths(const std::string& rocdl_dir_path) {
  // Construct full path to ROCDL bitcode libraries.
  std::vector<std::string> result;
  result.reserve(2);
  for (absl::string_view filename : {"ocml.bc", "ockl.bc"}) {
    result.emplace_back(tsl::io::JoinPath(rocdl_dir_path, filename));
  }

  return result;
}

struct HsacoCache {
  
  using HashType = std::array<uint8_t, 32>;
private:
  struct Hash64 {
    size_t operator()(const HashType& s) const noexcept {
      return *reinterpret_cast< const size_t * >(s.data());
    }
  };

  absl::Mutex mutex_;
  absl::flat_hash_map<HashType, std::vector<uint8_t>, Hash64> 
                hsaco_cache_ ABSL_GUARDED_BY(mutex_);
  std::atomic_int request_count_, hit_count_;
  std::string hsaco_cache_dir_;
  int64_t bitcode_size_threshold_;
  bool keep_temp_files_;

  HsacoCache() {
    auto *env = tsl::Env::Default();
    TF_CHECK_OK(tsl::ReadStringFromEnvVar("TF_XLA_HSACO_CACHE_DIR", "/tmp",
                                     &hsaco_cache_dir_));
    // minimal size of llvm Module bitcode to use file cache
    TF_CHECK_OK(tsl::ReadInt64FromEnvVar("TF_XLA_HSACO_BITCODE_SIZE_THRESHOLD", 
                  /*default_val=*/65536, &bitcode_size_threshold_));

    TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_ROCM_KEEP_XLA_TEMPFILES",
                          /*default_val=*/false, &keep_temp_files_));

    if (hsaco_cache_dir_.empty()) {
      hsaco_cache_dir_ = "/tmp";
      LOG(WARNING) << 
       "TF_XLA_HSACO_CACHE_DIR is empty: using default location for HSACO cache!";
    }
    if (!env->IsDirectory(hsaco_cache_dir_).ok()) {
      if(!env->CreateDir(hsaco_cache_dir_).ok()) {
        LOG(FATAL) << "Unable to create hsaco cache dir: " << hsaco_cache_dir_;
      }
    }
    LOG(INFO) << "HSACO file cache in '" << hsaco_cache_dir_ 
              << "' is enabled for LLVM modules with bitcode size >= " 
              << bitcode_size_threshold_ << " bytes";

    if(hsaco_cache_dir_.back() != '/') hsaco_cache_dir_ += '/';
  }
 public:
  static HsacoCache& i() {
    static HsacoCache obj;
    return obj;
  }

  bool keep_temp_files() const { return keep_temp_files_; }

  std::string hsaco_file_path(const std::string& hash_str) const {
    return hsaco_cache_dir_ + hash_str + ".hsaco";
  }

  bool find(const HashType& hash_val, int64_t bitcode_size, 
          std::string *hash_str, std::vector<uint8_t> *hsaco);

  // attempts to read an hsaco binary file, adds it to in-memory cache, and
  // (if enabled) moves/copies the binary file to the cached location
  bool read_from_file(const HashType& hash_val, int64_t bitcode_size, 
      const std::string& hash_str, std::optional< std::string > hsaco_src_path,
      std::vector<uint8_t> *hsaco);
}; // HsacoCache

bool HsacoCache::find(const HashType& hash_val, int64_t bitcode_size, 
      std::string *hash_str, std::vector<uint8_t> *hsaco) {
  
  bool hit = false;
  request_count_++;
  {
    absl::MutexLock lock(&mutex_);
    if (auto it = hsaco_cache_.find(hash_val); it != hsaco_cache_.end()) {
      hit = true, *hsaco = it->second;
    }
  }
  absl::string_view hview(reinterpret_cast<const char*>(hash_val.data()),
                              hash_val.size());
  *hash_str = absl::BytesToHexString(hview);

  if (!hit && bitcode_size >= bitcode_size_threshold_) {
    if (read_from_file(hash_val, bitcode_size,  *hash_str, std::nullopt, hsaco)) {
      hit = true;
      VLOG(1) << "HSACO file cache hit";
    }
  }
  if (hit) hit_count_++;
  VLOG(1) << "HSACO cache: " << request_count_ << " requests, "
            << hit_count_ << " hits";
  return hit;
}

bool HsacoCache::read_from_file(const HashType& hash_val, 
    int64_t bitcode_size, const std::string& hash_str, 
    std::optional< std::string > hsaco_src_path, std::vector<uint8_t> *hsaco) {
  
  size_t fsize = 0;
  auto save_path = hsaco_file_path(hash_str);
  if (!hsaco_src_path) hsaco_src_path = save_path;
  {
    std::ifstream ifs(*hsaco_src_path, std::ios::binary | std::ios::ate); 
    if (!ifs.is_open()) return false;
    fsize = ifs.tellg();
    if (fsize == 0) return false;
    *hsaco = std::vector<uint8_t>(fsize);
    ifs.seekg(0, std::ios::beg);
    ifs.read(reinterpret_cast<char*>(hsaco->data()), fsize);
  }

  absl::MutexLock lock(&mutex_);
  hsaco_cache_.emplace(hash_val, *hsaco);

  if (*hsaco_src_path != save_path && bitcode_size >= bitcode_size_threshold_) {
    // write hsaco file to the new location if simple rename fails
    if (!tsl::Env::Default()->RenameFile(*hsaco_src_path, save_path).ok()) {
      std::ofstream ofs(save_path, std::ios::binary);
      ofs.write(reinterpret_cast< const char *>(hsaco->data()), fsize);
      std::remove(hsaco_src_path->c_str()); // remove temporary file
      if (ofs.fail()) {
        LOG(FATAL) << "Unable to write hsaco file cache: " << save_path;
      }
    }
  }
  return true;
}

const auto& getJaxPluginPaths() {
  static const struct {
    std::string bitcode_path;
    std::string lld_path;
  } paths = {
    std::getenv("JAX_ROCM_PLUGIN_INTERNAL_BITCODE_PATH") ?: "",
    std::getenv("JAX_ROCM_PLUGIN_INTERNAL_LLD_PATH") ?: "",
  };
  return paths;
}

// Emits the given module to HSA Code Object. target_machine is an initialized
// TargetMachine for the AMDGPU target.
absl::StatusOr< std::string> EmitModuleToHsaco(
    llvm::Module* module, llvm::TargetMachine* target_machine) {

  auto* env = tsl::Env::Default();
  std::vector<std::string> tempdir_vector;
  env->GetLocalTempDirectories(&tempdir_vector);
  if (tempdir_vector.empty()) {
    return xla::Internal(
        "Unable to locate a temporary directory for compile-time artifacts.");
  }
  std::string tempdir_name = tempdir_vector.front();
  VLOG(1) << "Compile-time artifacts located at: " << tempdir_name;

  // Prepare filenames for all stages of compilation:
  // IR, binary ISA, and HSACO.
  std::string random_number = std::to_string(tsl::random::New64());
  auto gen_path = [module, &random_number, &tempdir_name](absl::string_view ext) {
    auto name =
      absl::StrCat(module->getModuleIdentifier(), random_number, ext);
    return tsl::io::JoinPath(tempdir_name, name);
  };

  std::string ir_path = gen_path(".ll"),
              ir_opt_path = gen_path("_opt.ll"),
              isabin_path = gen_path(".o"),
              hsaco_path = gen_path(".hsaco");

  std::error_code ec;
  { // Dump LLVM IR.
    llvm::raw_fd_ostream ir_fs(ir_path, ec, llvm::sys::fs::OF_None);
    module->print(ir_fs, nullptr);
  }

  { // Emit GCN ISA binary.
    llvm::legacy::PassManager pm;
    pm.add(new llvm::TargetLibraryInfoWrapperPass(
      llvm::Triple(module->getTargetTriple())));

    llvm::raw_fd_ostream isabin_fs(isabin_path, ec, llvm::sys::fs::OF_Text);
    module->setDataLayout(target_machine->createDataLayout());
    target_machine->addPassesToEmitFile(pm, isabin_fs, nullptr,
                                      llvm::CodeGenFileType::ObjectFile);
    pm.run(*module);
  }

  if (HsacoCache::i().keep_temp_files()) {
    llvm::raw_fd_ostream ir_fs(ir_opt_path, ec, llvm::sys::fs::OF_None);
    module->print(ir_fs, nullptr);
  }
  
  static bool use_inprocess_lld = []() {
    bool inprocess_lld = false;
    TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_ROCM_INPROCESS_LLD",
                                        /*default_val=*/true, &inprocess_lld));
    return inprocess_lld;
  }();

  if (use_inprocess_lld) {
    static absl::Mutex lld_mu(absl::kConstInit);

    std::initializer_list<const char*> args{
        "ld.lld", "--threads=1", "-shared",
        "--no-undefined", isabin_path.c_str(), "-o",
        hsaco_path.c_str(),
    };

    std::string error_message;
    llvm::raw_string_ostream os(error_message);
    lld::Result result;
    {
      absl::MutexLock lock(&lld_mu);
      result =
          lld::lldMain(args, llvm::nulls(), os, {{lld::Gnu, &lld::elf::link}});
    }
    CHECK(result.canRunAgain)
        << "ld.lld (in-process) failed with fatal error " << error_message;
    if (result.retCode) {
      return xla::Internal(
          "ld.lld (in-process) execute fail: %s, error code %d", error_message,
          result.retCode);
    }
  } else {
    // Locate lld.
    llvm::SmallVector<std::string, 3> lld_paths;

    if (const char* llvm_path = std::getenv("LLVM_PATH")) {
      lld_paths.push_back(tsl::io::JoinPath(llvm_path, "bin"));
    }
    lld_paths.push_back(tsl::io::JoinPath(tsl::RocmRoot(), "llvm/bin"));

    // push LLD path from JAX plugin if set
    if (const auto& jpaths = getJaxPluginPaths(); !jpaths.lld_path.empty()) {
      lld_paths.push_back(jpaths.lld_path);
    }

    auto lld_program = llvm::sys::findProgramByName(
      "ld.lld", llvm::to_vector_of<llvm::StringRef>(lld_paths));
    if (!lld_program) {
      return xla::Internal("unable to find ld.lld in PATH: %s",
                         lld_program.getError().message());
    }
    std::initializer_list<llvm::StringRef> lld_args{
        "ld.lld", "-flavor", "gnu", "-shared",
        "--no-undefined", isabin_path, "-o", hsaco_path,
    };

    std::string error_message;
    int lld_result =
        llvm::sys::ExecuteAndWait(*lld_program, lld_args,
                                  std::nullopt, {}, 0, 0, &error_message);
    if (lld_result) {
      return xla::Internal("ld.lld execute fail: %s, error code %d",
                           error_message, lld_result);
    }
  } // use_inprocess_lld

  if (!HsacoCache::i().keep_temp_files()) {
    std::remove(ir_path.c_str());
    std::remove(isabin_path.c_str());
    std::remove(ir_opt_path.c_str());
  }
  return hsaco_path;
}

// Links ROCm-Device-Libs into the given module if the module needs it.
absl::Status LinkROCDLIfNecessary(llvm::Module* module,
                                  const std::string& gfx_version,
                                  const DebugOptions& debug_options,
                                  const std::string& rocdl_dir_path) {
  if (!CouldNeedDeviceBitcode(*module)) {
    return absl::OkStatus();
  }

  auto addControlVariable = [&](llvm::StringRef name, uint32_t value,
                                uint32_t bitwidth = 8) {
    if (module->getNamedGlobal(name)) return;
    llvm::IntegerType* type =
        llvm::IntegerType::getIntNTy(module->getContext(), bitwidth);
    llvm::GlobalVariable* control_variable = new llvm::GlobalVariable(
        *module, type, /*isConstant=*/true,
        llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
        llvm::ConstantInt::get(type, value), name, /*before=*/nullptr,
        /*threadLocalMode=*/llvm::GlobalValue::ThreadLocalMode::NotThreadLocal,
        /*addressSpace=*/4);
    control_variable->setVisibility(
        llvm::GlobalValue::VisibilityTypes::ProtectedVisibility);
    control_variable->setAlignment(llvm::MaybeAlign(bitwidth / 8));
    control_variable->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);
    VLOG(2) << "addControlVariable " << name.data() << " " << value;
  };

  addControlVariable("__oclc_finite_only_opt", false);
  // TODO(rocm): Maybe check ftz for this one
  addControlVariable("__oclc_daz_opt", false);
  addControlVariable("__oclc_correctly_rounded_sqrt32", true);
  addControlVariable("__oclc_unsafe_math_opt", false);

  // TODO(rocm): Move this into device_description.h or use llvm infra
  CHECK((gfx_version[3] == '9' && gfx_version.size() == 6) ||
        (gfx_version[3] == '1' && gfx_version.size() == 7));

  uint32_t major, stepping, minor;

  if (gfx_version[3] == '9') {
    major = 9;
    CHECK(absl::SimpleAtoi({&gfx_version[4], 1}, &stepping));
    CHECK(absl::SimpleHexAtoi({&gfx_version[5], 1}, &minor));
  } else {
    CHECK(absl::SimpleAtoi({&gfx_version[3], 2}, &major));
    CHECK(absl::SimpleAtoi({&gfx_version[5], 1}, &stepping));
    CHECK(absl::SimpleAtoi({&gfx_version[6], 1}, &minor));
  }

  // TODO(rocm): Not great, not terrible
  addControlVariable("__oclc_wavefrontsize64", major == 9);
  addControlVariable("__oclc_ISA_version",
                     1000 * major + 100 * stepping + minor, 32);
  addControlVariable("__oclc_ABI_version", kAMDGPUAbiVersion, 32);


  static bool use_embedded_device_lib = []() {
    bool embedded_device_lib = false;
    TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_ROCM_EMBEDDED_DEVICE_LIB",
                                  /*default_val=*/true, &embedded_device_lib));
    return embedded_device_lib;
  }();

  if (use_embedded_device_lib) {
    static const char device_lib_data[] = {
#include "amdgpu_device_lib_data.inc"
    };

    llvm::Linker linker(*module);
    auto device_lib = llvm::getLazyBitcodeModule(
        {llvm::StringRef{device_lib_data, sizeof(device_lib_data)},
         "device_lib"},
        module->getContext());
    if (!device_lib) {
      return xla::Internal("Error loading embeded device lib.");
    }
    if (linker.linkInModule(
            std::move(*device_lib), llvm::Linker::Flags::LinkOnlyNeeded,
            [](llvm::Module& M, const llvm::StringSet<>& GVS) {
              internalizeModule(M, [&GVS](const llvm::GlobalValue& GV) {
                return !GV.hasName() || (GVS.count(GV.getName()) == 0);
              });
            })) {
      return xla::Internal("Error linking embeded device lib.");
    }
    return absl::OkStatus();
  }

  TF_RETURN_IF_ERROR(
      LinkWithBitcodeVector(module, GetROCDLPaths(rocdl_dir_path)));

  // Sanitize stray metadata from the bitcode files
  if (auto* opencl_version = module->getNamedMetadata("opencl.ocl.version"))
    module->eraseNamedMetadata(opencl_version);

  if (auto* ident = module->getNamedMetadata("llvm.ident"))
    module->eraseNamedMetadata(ident);

  return absl::OkStatus();
}

absl::Status AMDGPUTargetModuleLinker(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    const std::string& device_bitcode_dir_path) {
  // Link the input module with ROCDL.

  auto comp_c =
      std::get_if<se::RocmComputeCapability>(&gpu_version);
  if (!comp_c) {
    return xla::Internal("Incompatible compute capability was specified.");
  }

  TF_RETURN_IF_ERROR(
      LinkROCDLIfNecessary(module, comp_c->gfx_version(),
                           debug_options, device_bitcode_dir_path));

  // If ftz is enabled, set it as an attribute on every function in the module.
  if (debug_options.xla_gpu_ftz()) {
    for (llvm::Function& fn : *module) {
      fn.addFnAttr("denormal-fp-math-f32", "preserve-sign");
    }
  }
  module->addModuleFlag(llvm::Module::Error, "amdhsa_code_object_version",
                        kAMDGPUAbiVersion);

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
  return std::pair{gfx, absl::StrJoin(mapped_tokens, ",")};
}

std::unique_ptr<llvm::TargetMachine> AMDGPUGetTargetMachine(
    llvm::Triple target_triple, const std::string& gcn_arch_name,
    const DebugOptions& debug_options) {

  auto [gfx, feature_str] = GetFeatureStrFromGCNArchName(gcn_arch_name);
  return GetTargetMachine(std::move(target_triple), gfx, debug_options,
                          feature_str);
}

// Returns the directory containing ROCm-Device-Libs files.
std::string GetROCDLDir(const DebugOptions& debug_options) {
  std::vector<std::string> potential_rocdl_dirs;
  const std::string& datadir = debug_options.xla_gpu_cuda_data_dir();
  if (!datadir.empty()) {
    potential_rocdl_dirs.push_back(datadir);
  }
  potential_rocdl_dirs.push_back(tsl::RocdlRoot());
  potential_rocdl_dirs.push_back(getJaxPluginPaths().bitcode_path);

  // Tries all potential ROCDL directories in the order they are inserted.
  // Returns the first directory that contains opencompute math libs bitcode file (ocml.bc)
  for (const std::string& potential_rocdl_dir : potential_rocdl_dirs) {
    if (tsl::Env::Default()->FileExists(tsl::io::JoinPath(potential_rocdl_dir, "ocml.bc")).ok()) {
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

class sha256_ostream : public llvm::raw_ostream {
 
  llvm::SHA256& obj_;
  uint64_t pos_ = 0;

  void write_impl(const char *ptr, size_t size) override {
    obj_.update(llvm::StringRef(ptr, size));
    pos_ += size;
  }
 
  /// Return the current position within the stream.
  uint64_t current_pos() const override { return pos_; }

  void anchor() override {}

  size_t preferred_buffer_size() const override {
    return llvm::raw_ostream::preferred_buffer_size(); // TODO ?
  }
 
public:
  explicit sha256_ostream(llvm::SHA256& sha256)
      : llvm::raw_ostream(/* unbuffered */false), obj_(sha256) {
    //SetUnbuffered(); // copied from raw_svector_ostream
  }

  uint64_t bitcode_size() const {
    return pos_;
  }
  ~sha256_ostream() override {
    llvm::raw_ostream::flush();
  }
};

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

  auto comp_c = std::get_if<se::RocmComputeCapability>(&gpu_version);
  if (!comp_c) {
    return xla::Internal("Incompatible compute capability was specified.");
  }

  std::vector<uint8_t> hsaco;
  std::string hash_str;

  tsl::profiler::TraceMe activity(
        [&] { return absl::StrCat("Compiling IR", module->getName().str()); },
        tsl::profiler::TraceMeLevel::kInfo);
  XLA_SCOPED_LOGGING_TIMER("Compile module " + module->getName().str());

  llvm::SHA256 sha256;
  sha256_ostream os(sha256);
  llvm::WriteBitcodeToFile(*module, os);
  os.flush();
  auto bitcode_size = os.bitcode_size();

  sha256.update(comp_c->gcn_arch_name());
  for(const auto& s : llvm_opts) sha256.update(s);
  auto opt_level = debug_options.xla_backend_optimization_level();
  sha256.update(llvm::ArrayRef(reinterpret_cast< const uint8_t *>(&opt_level), 
           sizeof(opt_level)));
  HsacoCache::HashType binary_hash = sha256.final();
  
  auto& cache = HsacoCache::i();
  if (cache.find(binary_hash, bitcode_size, &hash_str, &hsaco)) {
    VLOG(1) << "HSACO cache hit";
    return hsaco;
  }

  VLOG(1) << "HSACO cache miss";
  llvm::Triple default_target_triple("amdgcn--amdhsa-amdgiz");
  // Construct LLVM TargetMachine for AMDGPU.
  std::unique_ptr<llvm::TargetMachine> target_machine =
      AMDGPUGetTargetMachine(default_target_triple, 
          comp_c->gcn_arch_name(), debug_options);

  // Link with ROCm-Device-Libs, and optimize the LLVM module.
  TF_RETURN_IF_ERROR(gpu::LinkAndOptimizeModule(
        module, gpu_version, debug_options, rocdl_dir_path,
        AMDGPUTargetModuleLinker, default_target_triple, target_machine.get(),
        kAMDGPUInlineThreshold));

  // Lower optimized LLVM module to HSA code object.
  TF_ASSIGN_OR_RETURN(auto hsaco_output_path, 
        EmitModuleToHsaco(module, target_machine.get()));

  // if file cache is enabled, this will move hsaco file to the file cache dir
  bool ok = cache.read_from_file(binary_hash, bitcode_size, hash_str,
                          hsaco_output_path, &hsaco);
  if (!cache.keep_temp_files()) {
    std::remove(hsaco_output_path.c_str());
  }
  if (!ok) return absl::InternalError("Unable to read hsaco output file!");
  return hsaco;
}

}  // namespace amdgpu
}  // namespace gpu
}  // namespace xla
