/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/target_machine_features.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/primitive_util.h"
#include "xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

struct CachedSubtargetInfo {
  int32_t default_vector_register_byte_size = 0;
  std::string enabled_target_features;
};

// Caches subtarget vector register width and expanded processor features by
// (triple, cpu, feature_string) so constructing a TargetMachineFeatures per
// compilation avoids allocating a temporary LLVMContext and Module.
const CachedSubtargetInfo& GetCachedSubtargetInfo(
    llvm::TargetMachine& target_machine) {
  using CacheKey = std::tuple<std::string, std::string, std::string>;
  static absl::NoDestructor<absl::Mutex> mu;
  static absl::NoDestructor<absl::node_hash_map<CacheKey, CachedSubtargetInfo>>
      cache;

  CacheKey key(target_machine.getTargetTriple().str(),
               target_machine.getTargetCPU().str(),
               target_machine.getTargetFeatureString().str());
  {
    absl::MutexLock lock(*mu);
    if (const auto it = cache->find(key); it != cache->end()) {
      return it->second;
    }
  }

  CachedSubtargetInfo info;
  info.enabled_target_features = absl::StrJoin(
      target_machine.getMCSubtargetInfo().getEnabledProcessorFeatures(), ",",
      [](std::string* out, const llvm::SubtargetFeatureKV* feature) {
        absl::StrAppend(out, "+", llvm_ir::AsStringView(feature->key()));
      });

  llvm::LLVMContext context;
  llvm::Module module("target_features_probe", context);
  llvm::Function* probe_fn = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(context),
                              /*isVarArg=*/false),
      llvm::GlobalValue::ExternalLinkage, "probe", &module);
  llvm::TargetTransformInfo tti =
      target_machine.getTargetTransformInfo(*probe_fn);
  info.default_vector_register_byte_size =
      tti.getRegisterBitWidth(llvm::TargetTransformInfo::RGK_FixedWidthVector)
          .getFixedValue() /
      8;

  absl::MutexLock lock(*mu);
  return cache->emplace(std::move(key), std::move(info)).first->second;
}

}  // namespace

TargetMachineFeatures::TargetMachineFeatures(
    llvm::TargetMachine* target_machine)
    : target_machine_(target_machine) {
  // Calling virtual methods in the constructor is discouraged, so we don't
  // call `get_target_feature_string` here.
  if (target_machine_) {
    const absl::string_view feature_string =
        llvm_ir::AsStringView(target_machine_->getTargetFeatureString());
    has_avx512bf16_ = absl::StrContains(feature_string, "+avx512bf16");
    has_avx512fp16_ = absl::StrContains(feature_string, "+avx512fp16");
    has_amx_fp16_ = absl::StrContains(feature_string, "+amx-fp16");
    has_amx_fp8_ = absl::StrContains(feature_string, "+amx-fp8");

    const CachedSubtargetInfo& info = GetCachedSubtargetInfo(*target_machine_);
    enabled_target_features_ = info.enabled_target_features;
    default_vector_register_byte_size_ = info.default_vector_register_byte_size;
  }
}

const llvm::TargetMachine* TargetMachineFeatures::target_machine() const {
  return target_machine_;
}

int32_t TargetMachineFeatures::vectorization_factor_in_bytes() const {
  // Ideally this should be a function of the cache line size (which we can
  // get from llvm::TargetTransformInfo::getCacheLineSize) of the target
  // machine.  Guess a value of 128 bytes for now.
  return 128;
}

int32_t TargetMachineFeatures::vector_register_byte_size() const {
  return default_vector_register_byte_size_;
}

int32_t TargetMachineFeatures::vector_register_byte_size(
    const llvm::Function& fn) const {
  llvm::TargetTransformInfo* tti = GetTargetTransformInfoFor(fn);
  return tti->getRegisterBitWidth(
             llvm::TargetTransformInfo::RGK_FixedWidthVector) /
         8;
}

int32_t TargetMachineFeatures::vector_register_num_elements(
    const llvm::Function& fn, PrimitiveType type) const {
  return vector_register_byte_size(fn) / (primitive_util::BitWidth(type) / 8);
}

int32_t TargetMachineFeatures::vector_register_count(
    const llvm::Function& fn) const {
  llvm::TargetTransformInfo* tti = GetTargetTransformInfoFor(fn);
  return static_cast<int32_t>(
      tti->getNumberOfRegisters(tti->getRegisterClassForType(/*Vector=*/true)));
}

llvm::TargetTransformInfo* TargetMachineFeatures::GetTargetTransformInfoFor(
    const llvm::Function& fn) const {
  auto it = target_transform_info_.find(&fn);
  if (it == target_transform_info_.end()) {
    auto emplace_result = target_transform_info_.emplace(
        &fn, target_machine_->getTargetTransformInfo(fn));
    CHECK(emplace_result.second);
    it = emplace_result.first;
  }

  return &it->second;
}

int64_t TargetMachineFeatures::minimum_alignment_for_allocation(
    int64_t size_bytes) const {
  // Assume that all pointers are aligned to at least
  // xla::cpu_function_runtime::kMinAlign.
  if (size_bytes == 0) {
    // No need to align empty buffers.
    return 1;
  }

  // Allow small buffers to be underaligned, there is no vectorization benefit
  // anyways.
  return std::min<int64_t>(llvm::PowerOf2Ceil(size_bytes), MinAlign());
}

std::string TargetMachineFeatures::get_target_feature_string() const {
  return target_machine_ == nullptr
             ? ""
             : target_machine_->getTargetFeatureString().str();
}

std::string TargetMachineFeatures::get_enabled_target_features() const {
  return enabled_target_features_;
}

}  // namespace xla::cpu
