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

#include "xla/backends/cpu/codegen/target_machine_features.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_test_base.h"
#include "xla/codegen/device_spec.h"

namespace xla::cpu {
namespace {

using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;

struct Avx512Bf16Fp16TestSpec {
  std::string cpu_name;
  std::string features;
  bool has_avx512bf16;
  bool has_avx512fp16;
};

class Avx512Bf16Fp16Test
    : public TargetMachineTestBase,
      public ::testing::WithParamInterface<Avx512Bf16Fp16TestSpec> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<Avx512Bf16Fp16TestSpec>& info) {
    return info.param.cpu_name;
  }
};

TEST_P(Avx512Bf16Fp16Test, CheckAvailability) {
  Avx512Bf16Fp16TestSpec spec = GetParam();
  const char* triple_string = "x86_64-unknown-linux-gnu";
  std::unique_ptr<TargetMachineFeatures> features =
      CreateTargetMachineFeatures(triple_string, spec.cpu_name, spec.features);
  EXPECT_EQ(features->has_avx512bf16(), spec.has_avx512bf16);
  EXPECT_EQ(features->has_avx512fp16(), spec.has_avx512fp16);
}

std::vector<Avx512Bf16Fp16TestSpec> GetAvx512Bf16Fp16TestSpecs() {
  return std::vector<Avx512Bf16Fp16TestSpec>{
      Avx512Bf16Fp16TestSpec{"znver3", "+avx,+avx2", false, false},
      Avx512Bf16Fp16TestSpec{"sapphirerapids",
                             "+avx512vnni,+avx512bf16,+avx512fp16,+amx-bf16,"
                             "+amx-int8,+amx-tile,+amx-transpose",
                             true, true}};
}

INSTANTIATE_TEST_SUITE_P(Avx512Bf16Fp16Suite, Avx512Bf16Fp16Test,
                         ::testing::ValuesIn(GetAvx512Bf16Fp16TestSpecs()),
                         Avx512Bf16Fp16Test::Name);

struct AmxFp16Fp8TestSpec {
  std::string cpu_name;
  std::string features;
  bool has_amx_fp16;
  bool has_amx_fp8;
};

class AmxFp16Fp8Test
    : public TargetMachineTestBase,
      public ::testing::WithParamInterface<AmxFp16Fp8TestSpec> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<AmxFp16Fp8TestSpec>& info) {
    return info.param.cpu_name;
  }
};

TEST_P(AmxFp16Fp8Test, CheckAvailability) {
  AmxFp16Fp8TestSpec spec = GetParam();
  const char* triple_string = "x86_64-unknown-linux-gnu";
  std::unique_ptr<TargetMachineFeatures> features =
      CreateTargetMachineFeatures(triple_string, spec.cpu_name, spec.features);
  EXPECT_EQ(features->has_amx_fp16(), spec.has_amx_fp16);
  EXPECT_EQ(features->has_amx_fp8(), spec.has_amx_fp8);
}

std::vector<AmxFp16Fp8TestSpec> GetAmxFp16Fp8TestSpecs() {
  return std::vector<AmxFp16Fp8TestSpec>{
      // SPR-class: neither AMX_FP16 nor AMX_FP8.
      AmxFp16Fp8TestSpec{"sapphirerapids", "+amx-bf16,+amx-int8,+amx-tile",
                         false, false},
      // GNR-class: AMX_FP16 only -> emulation path for FP8.
      AmxFp16Fp8TestSpec{"graniterapids",
                         "+amx-bf16,+amx-int8,+amx-tile,+amx-fp16", true,
                         false},
      // DMR-class: native AMX_FP8 (and AMX_FP16).
      AmxFp16Fp8TestSpec{"diamondrapids",
                         "+amx-bf16,+amx-int8,+amx-tile,+amx-fp16,+amx-fp8",
                         true, true},
  };
}

INSTANTIATE_TEST_SUITE_P(AmxFp16Fp8Suite, AmxFp16Fp8Test,
                         ::testing::ValuesIn(GetAmxFp16Fp8TestSpecs()),
                         AmxFp16Fp8Test::Name);

using SubtargetFeaturesTest = TargetMachineTestBase;

TEST_F(SubtargetFeaturesTest, NullTargetMachineReturnsEmptyDefaults) {
  TargetMachineFeatures features(/*target_machine=*/nullptr);
  EXPECT_EQ(features.vector_register_byte_size(), 0);
  EXPECT_THAT(features.get_enabled_target_features(), IsEmpty());
  EXPECT_THAT(features.get_target_feature_string(), IsEmpty());
}

TEST_F(SubtargetFeaturesTest, DefaultVectorRegisterByteSizeMatchesSubtarget) {
  constexpr absl::string_view kX86Triple = "x86_64-unknown-linux-gnu";
  std::unique_ptr<TargetMachineFeatures> sse =
      CreateTargetMachineFeatures(kX86Triple, "x86-64", "");
  std::unique_ptr<TargetMachineFeatures> avx2 =
      CreateTargetMachineFeatures(kX86Triple, "znver3", "");
  std::unique_ptr<TargetMachineFeatures> avx512 =
      CreateTargetMachineFeatures(kX86Triple, "znver4", "");
  std::unique_ptr<TargetMachineFeatures> avx512_disabled =
      CreateTargetMachineFeatures(kX86Triple, "znver4", "-avx512f");
  std::unique_ptr<TargetMachineFeatures> spr_prefer_256 =
      CreateTargetMachineFeatures(kX86Triple, "sapphirerapids", "");
  std::unique_ptr<TargetMachineFeatures> spr_full_512 =
      CreateTargetMachineFeatures(kX86Triple, "sapphirerapids",
                                  "-prefer-256-bit");

  EXPECT_EQ(sse->vector_register_byte_size(), 16);
  EXPECT_EQ(avx2->vector_register_byte_size(), 32);
  EXPECT_EQ(avx512->vector_register_byte_size(), 64);
  EXPECT_EQ(avx512_disabled->vector_register_byte_size(), 32);
  EXPECT_EQ(spr_prefer_256->vector_register_byte_size(), 32);
  EXPECT_EQ(spr_full_512->vector_register_byte_size(), 64);
}

TEST_F(SubtargetFeaturesTest, EnabledTargetFeaturesIncludeCpuImpliedFeatures) {
  constexpr absl::string_view kX86Triple = "x86_64-unknown-linux-gnu";
  // Empty explicit feature string still exposes features implied by the CPU.
  std::unique_ptr<TargetMachineFeatures> znver3 =
      CreateTargetMachineFeatures(kX86Triple, "znver3", "");
  EXPECT_THAT(znver3->get_target_feature_string(), IsEmpty());
  EXPECT_THAT(znver3->get_enabled_target_features(), HasSubstr("+avx2"));
  EXPECT_THAT(znver3->get_enabled_target_features(), HasSubstr("+fma"));
  EXPECT_THAT(znver3->get_enabled_target_features(),
              Not(HasSubstr("+avx512f")));

  std::unique_ptr<TargetMachineFeatures> spr =
      CreateTargetMachineFeatures(kX86Triple, "sapphirerapids", "");
  EXPECT_THAT(spr->get_enabled_target_features(), HasSubstr("+avx512f"));
  EXPECT_THAT(spr->get_enabled_target_features(), HasSubstr("+avx512bf16"));

  std::unique_ptr<TargetMachineFeatures> spr_no_avx512 =
      CreateTargetMachineFeatures(kX86Triple, "sapphirerapids", "-avx512f");
  EXPECT_THAT(spr_no_avx512->get_enabled_target_features(),
              Not(HasSubstr("+avx512f")));
}

TEST_F(SubtargetFeaturesTest, CachedSubtargetInfoPreservesResultsOnReuse) {
  constexpr absl::string_view kX86Triple = "x86_64-unknown-linux-gnu";
  std::unique_ptr<TargetMachineFeatures> first =
      CreateTargetMachineFeatures(kX86Triple, "znver3", "+avx2");
  std::unique_ptr<TargetMachineFeatures> second =
      CreateTargetMachineFeatures(kX86Triple, "znver3", "+avx2");

  EXPECT_EQ(first->vector_register_byte_size(), 32);
  EXPECT_EQ(second->vector_register_byte_size(),
            first->vector_register_byte_size());
  EXPECT_EQ(second->get_enabled_target_features(),
            first->get_enabled_target_features());
  EXPECT_EQ(second->cpu_device_spec().has_atomic_cmpxchg_128,
            first->cpu_device_spec().has_atomic_cmpxchg_128);
  EXPECT_EQ(second->cpu_device_spec().has_atomic_load_store_128,
            first->cpu_device_spec().has_atomic_load_store_128);
}

TEST(CpuAtomic128NullTest, DefaultAndNullTargetMachineCapabilities) {
  CpuDeviceSpec default_spec;
  EXPECT_FALSE(default_spec.has_atomic_cmpxchg_128);
  EXPECT_FALSE(default_spec.has_atomic_load_store_128);
  EXPECT_TRUE(default_spec.has_atomic_rmw_64);
  EXPECT_TRUE(default_spec.has_atomic_load_store_64);

  CpuDeviceSpec null_spec = CpuDeviceSpecFromTargetMachine(nullptr);
  EXPECT_FALSE(null_spec.has_atomic_cmpxchg_128);
  EXPECT_FALSE(null_spec.has_atomic_load_store_128);
  EXPECT_FALSE(null_spec.has_atomic_rmw_64);
  EXPECT_FALSE(null_spec.has_atomic_load_store_64);

  TargetMachineFeatures null_features(/*target_machine=*/nullptr);
  EXPECT_FALSE(null_features.cpu_device_spec().has_atomic_cmpxchg_128);
  EXPECT_FALSE(null_features.cpu_device_spec().has_atomic_load_store_128);
  EXPECT_FALSE(null_features.cpu_device_spec().has_atomic_rmw_64);
  EXPECT_FALSE(null_features.cpu_device_spec().has_atomic_load_store_64);
}

struct CpuAtomic128TestSpec {
  std::string name;
  std::string triple;
  std::string cpu_name;
  std::string features;
  bool has_cmpxchg_128;
  bool has_load_store_128;
  bool has_rmw_64 = true;
  bool has_load_store_64 = true;
};

class CpuAtomic128SupportTest
    : public TargetMachineTestBase,
      public ::testing::WithParamInterface<CpuAtomic128TestSpec> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<CpuAtomic128TestSpec>& info) {
    return info.param.name;
  }
};

TEST_P(CpuAtomic128SupportTest, DetectsNativeAtomicCapabilities) {
  const CpuAtomic128TestSpec& spec = GetParam();
  std::unique_ptr<TargetMachineFeatures> features =
      CreateTargetMachineFeatures(spec.triple, spec.cpu_name, spec.features);
  ASSERT_NE(features->target_machine(), nullptr);
  CpuDeviceSpec cpu_spec = features->cpu_device_spec();
  EXPECT_EQ(cpu_spec.has_atomic_cmpxchg_128, spec.has_cmpxchg_128);
  EXPECT_EQ(cpu_spec.has_atomic_load_store_128, spec.has_load_store_128);
  EXPECT_EQ(cpu_spec.has_atomic_rmw_64, spec.has_rmw_64);
  EXPECT_EQ(cpu_spec.has_atomic_load_store_64, spec.has_load_store_64);
}

INSTANTIATE_TEST_SUITE_P(
    CpuAtomic128SupportSuite, CpuAtomic128SupportTest,
    ::testing::Values(
        CpuAtomic128TestSpec{"X86_64Base", "x86_64-unknown-linux-gnu", "x86-64",
                             "", false, false},
        CpuAtomic128TestSpec{"X86_64NoCx16", "x86_64-unknown-linux-gnu",
                             "x86-64", "-cx16", false, false},
        CpuAtomic128TestSpec{"X86_64AvxWithoutCx16", "x86_64-unknown-linux-gnu",
                             "x86-64", "+avx,-cx16", false, false},
        CpuAtomic128TestSpec{"X86_64Cx16Only", "x86_64-unknown-linux-gnu",
                             "x86-64", "+cx16", true, false},
        CpuAtomic128TestSpec{"X86_64Cx16Avx", "x86_64-unknown-linux-gnu",
                             "x86-64", "+cx16,+avx", true, true},
        CpuAtomic128TestSpec{"X86_64Cx16AvxSoftFloat",
                             "x86_64-unknown-linux-gnu", "x86-64",
                             "+cx16,+avx,+soft-float", true, false},
        CpuAtomic128TestSpec{"X86_64V2ImpliedCx16", "x86_64-unknown-linux-gnu",
                             "x86-64-v2", "", true, false},
        CpuAtomic128TestSpec{"X86_64HaswellImpliedCx16Avx",
                             "x86_64-unknown-linux-gnu", "haswell", "", true,
                             true},
        CpuAtomic128TestSpec{"X86_64HaswellNoAvx", "x86_64-unknown-linux-gnu",
                             "haswell", "-avx", true, false},
        CpuAtomic128TestSpec{"X86_64HaswellNoCx16", "x86_64-unknown-linux-gnu",
                             "haswell", "-cx16", false, false},
        CpuAtomic128TestSpec{"X86_64X32Ilp32", "x86_64-unknown-linux-gnux32",
                             "haswell", "+cx16,+avx", false, false, false,
                             false},
        CpuAtomic128TestSpec{"X86_32WithCx16Avx", "i686-unknown-linux-gnu",
                             "haswell", "+cx16,+avx", false, false, false,
                             false},
        CpuAtomic128TestSpec{"AArch64Base", "aarch64-unknown-linux-gnu",
                             "generic", "", true, false},
        CpuAtomic128TestSpec{"AArch64Lse", "aarch64-unknown-linux-gnu",
                             "generic", "+lse", true, false},
        CpuAtomic128TestSpec{"AArch64LseLse2", "aarch64-unknown-linux-gnu",
                             "generic", "+lse,+lse2", true, true},
        CpuAtomic128TestSpec{"AArch64OutlineAtomics",
                             "aarch64-unknown-linux-gnu", "generic",
                             "+outline-atomics", false, false, false, true},
        CpuAtomic128TestSpec{"AArch64LseWithOutlineAtomics",
                             "aarch64-unknown-linux-gnu", "generic",
                             "+lse,+outline-atomics", true, false},
        CpuAtomic128TestSpec{"AArch64LseLse2WithOutlineAtomics",
                             "aarch64-unknown-linux-gnu", "generic",
                             "+lse,+lse2,+outline-atomics", true, true},
        CpuAtomic128TestSpec{
            "AArch64Lse2WithOutlineAtomicsNoLse", "aarch64-unknown-linux-gnu",
            "generic", "+lse2,+outline-atomics", false, true, false, true},
        CpuAtomic128TestSpec{"AArch64NeoverseN1ImpliedLse",
                             "aarch64-unknown-linux-gnu", "neoverse-n1", "",
                             true, false},
        CpuAtomic128TestSpec{
            "AArch64NeoverseN1NoLseOutlineAtomics", "aarch64-unknown-linux-gnu",
            "neoverse-n1", "-lse,+outline-atomics", false, false, false, true},
        CpuAtomic128TestSpec{"AArch64NeoverseV2ImpliedLseLse2",
                             "aarch64-unknown-linux-gnu", "neoverse-v2", "",
                             true, true},
        CpuAtomic128TestSpec{"AArch64AppleM1ImpliedLseLse2",
                             "arm64-apple-darwin", "apple-m1", "", true, true},
        CpuAtomic128TestSpec{"AArch64BigEndianDisabled",
                             "aarch64_be-unknown-linux-gnu", "neoverse-v2", "",
                             false, false, false, false},
        CpuAtomic128TestSpec{"AArch64Ilp32Disabled",
                             "aarch64-unknown-linux-gnu_ilp32", "neoverse-v2",
                             "", false, false, false, false},
        CpuAtomic128TestSpec{"AArch64_32Disabled", "aarch64_32-apple-watchos",
                             "generic", "+lse,+lse2", false, false, false,
                             false},
        CpuAtomic128TestSpec{"Arm32Disabled", "armv7-unknown-linux-gnueabihf",
                             "generic", "", false, false, false, false}),
    CpuAtomic128SupportTest::Name);

}  // namespace
}  // namespace xla::cpu
