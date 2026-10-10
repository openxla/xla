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

#include "xla/pjrt/pjrt_build_stamp.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Optional;

TEST(PjRtBuildStampTest, BasicAccessorsAndDefaults) {
  const absl::Time now = absl::FromUnixSeconds(1700000000);
  PjRtBuildStamp::Options options;
  options.build_timestamp = now;
  options.vcs_revision = "git_rev_123";
  options.build_label = "target_foo";
  options.build_cl = 123456;
  options.base_source_hash = "hash_abcdef01";

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(options)));

  EXPECT_EQ(stamp.build_timestamp(), now);
  EXPECT_EQ(stamp.build_vcs_revision(), "git_rev_123");
  EXPECT_EQ(stamp.build_label(), "target_foo");
  EXPECT_EQ(stamp.build_cl(), 123456);
  EXPECT_EQ(stamp.base_source_hash(), "hash_abcdef01");
  EXPECT_EQ(stamp.source_hash(), "hash_abcdef01");
  EXPECT_EQ(stamp.mosaic_version(), std::nullopt);
  EXPECT_TRUE(stamp.components().empty());
  EXPECT_TRUE(stamp.attributes().empty());
  EXPECT_EQ(stamp.CacheKeyFingerprint(), "hash_abcdef01");
}

TEST(PjRtBuildStampTest, ValidationFailures) {
  // Empty identity fields must fail validation.
  EXPECT_THAT(PjRtBuildStamp::Create(PjRtBuildStamp::Options{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("at least one build identity field")));

  // Component with empty name must fail.
  PjRtBuildStamp::Options opt_empty_name;
  opt_empty_name.base_source_hash = "hash";
  opt_empty_name.components.push_back(
      PjRtComponentBuildStamp(/*kind_in=*/"", /*name_in=*/"",
                              /*owner_in=*/"", /*fingerprint_in=*/"fp"));
  EXPECT_THAT(PjRtBuildStamp::Create(std::move(opt_empty_name)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("name and fingerprint must not be empty")));

  // Component with empty fingerprint must fail.
  PjRtBuildStamp::Options opt_empty_fp;
  opt_empty_fp.base_source_hash = "hash";
  opt_empty_fp.components.push_back(
      PjRtComponentBuildStamp(/*kind_in=*/"", /*name_in=*/"comp",
                              /*owner_in=*/"", /*fingerprint_in=*/""));
  EXPECT_THAT(PjRtBuildStamp::Create(std::move(opt_empty_fp)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("name and fingerprint must not be empty")));
}

TEST(PjRtBuildStampTest, OptionsFromBuildData) {
  PjRtBuildStamp::Options options = PjRtBuildStamp::Options::FromBuildData();
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(options)));
  EXPECT_NE(stamp.build_timestamp(), absl::InfinitePast());
}

TEST(PjRtBuildStampTest, ToOptionsAndRecreation) {
  PjRtBuildStamp::Options initial_options;
  initial_options.vcs_revision = "rev1";
  initial_options.build_cl = 123;
  initial_options.base_source_hash = "hash1";
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp original,
                       PjRtBuildStamp::Create(std::move(initial_options)));

  PjRtBuildStamp::Options options = original.ToOptions();
  EXPECT_EQ(options.vcs_revision, "rev1");
  EXPECT_EQ(options.build_cl, 123);
  EXPECT_EQ(options.base_source_hash, "hash1");

  options.components.push_back(PjRtComponentBuildStamp(
      /*kind_in=*/"custom_call", /*name_in=*/"op_a", /*owner_in=*/"",
      /*fingerprint_in=*/"fp_a"));

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp modified,
                       PjRtBuildStamp::Create(std::move(options)));
  EXPECT_EQ(modified.components().size(), 1);
  EXPECT_EQ(modified.components()[0].name, "op_a");
  EXPECT_NE(original.CacheKeyFingerprint(), modified.CacheKeyFingerprint());
}

TEST(PjRtBuildStampTest, AttributesHandling) {
  PjRtBuildStamp::Options options;
  options.base_source_hash = "hash1";
  options.SetMosaicVersion(15);
  options.attributes["vendor.candidate"] = std::string("xla_tpu_candidate_1");
  options.attributes["is_experimental"] = true;

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(options)));

  EXPECT_THAT(stamp.mosaic_version(), Optional(int64_t{15}));
  EXPECT_THAT(stamp.GetAttribute<int64_t>(kBuildStampAttrMosaicVersion),
              Optional(int64_t{15}));
  EXPECT_THAT(stamp.GetStringAttribute("vendor.candidate"),
              Optional(absl::string_view("xla_tpu_candidate_1")));
  EXPECT_THAT(stamp.GetAttribute<std::string>("vendor.candidate"),
              Optional(std::string("xla_tpu_candidate_1")));
  EXPECT_THAT(stamp.GetAttribute<bool>("is_experimental"), Optional(true));
  EXPECT_EQ(stamp.GetAttribute<int64_t>("non_existent"), std::nullopt);
  EXPECT_EQ(stamp.GetStringAttribute("non_existent"), std::nullopt);
  EXPECT_EQ(stamp.GetAttribute<std::string>(kBuildStampAttrMosaicVersion),
            std::nullopt);  // Type mismatch
  EXPECT_EQ(stamp.GetStringAttribute(kBuildStampAttrMosaicVersion),
            std::nullopt);  // Type mismatch
}

TEST(PjRtBuildStampTest, CanonicalOrderIndependentComponentFingerprint) {
  PjRtComponentBuildStamp c1("xla.gpu.native_custom_call", "alpha", "pkg/alpha",
                             "1111");
  PjRtComponentBuildStamp c2("xla.ffi.handler", "beta", "pkg/beta", "2222");

  PjRtBuildStamp::Options opt1;
  opt1.base_source_hash = "base_hash";
  opt1.components = {c1, c2};
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp1,
                       PjRtBuildStamp::Create(std::move(opt1)));

  PjRtBuildStamp::Options opt2;
  opt2.base_source_hash = "base_hash";
  opt2.components = {c2, c1};
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::Create(std::move(opt2)));

  // Changing order of components MUST produce identical CacheKeyFingerprint.
  EXPECT_EQ(stamp1.CacheKeyFingerprint(), stamp2.CacheKeyFingerprint());

  // Changing a component fingerprint MUST alter CacheKeyFingerprint.
  PjRtComponentBuildStamp c2_modified = c2;
  c2_modified.fingerprint = "3333";
  PjRtBuildStamp::Options opt3;
  opt3.base_source_hash = "base_hash";
  opt3.components = {c1, c2_modified};
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp3,
                       PjRtBuildStamp::Create(std::move(opt3)));
  EXPECT_NE(stamp1.CacheKeyFingerprint(), stamp3.CacheKeyFingerprint());
}

TEST(PjRtBuildStampTest, CacheKeyFingerprintForComponentsFiltering) {
  PjRtComponentBuildStamp c_matmul("xla.gpu.native_custom_call", "fused_matmul",
                                   "pkg/matmul", "fp_matmul");
  PjRtComponentBuildStamp c_topk("xla.gpu.native_custom_call", "fused_topk",
                                 "pkg/topk", "fp_topk");

  PjRtBuildStamp::Options opt;
  opt.base_source_hash = "base_hash";
  opt.components = {c_matmul, c_topk};
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(opt)));

  // When HLO only calls "fused_matmul", the key excludes "fused_topk".
  absl::flat_hash_set<absl::string_view> active_matmul = {"fused_matmul"};
  std::string fp_matmul = stamp.CacheKeyFingerprintForComponents(active_matmul);

  absl::flat_hash_set<absl::string_view> active_topk = {"fused_topk"};
  std::string fp_topk = stamp.CacheKeyFingerprintForComponents(active_topk);

  EXPECT_NE(fp_matmul, fp_topk);

  // If active components is empty, returns base_source_hash.
  absl::flat_hash_set<absl::string_view> none = {};
  EXPECT_EQ(stamp.CacheKeyFingerprintForComponents(none), "base_hash");
}

TEST(PjRtBuildStampTest,
     FallbackIdentityComponentsFilteringWithoutDoubleAppend) {
  // Tests that when base_source_hash is empty and fallback identity (e.g. CL)
  // is used, CacheKeyFingerprintForComponents doesn't double-append components.
  PjRtComponentBuildStamp c1("", "comp1", "", "fp1");

  PjRtBuildStamp::Options opt;
  opt.build_cl = 12345;
  opt.components = {c1};
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(opt)));

  absl::flat_hash_set<absl::string_view> none;
  EXPECT_EQ(stamp.CacheKeyFingerprintForComponents(none), "cl:12345");

  absl::flat_hash_set<absl::string_view> active = {"comp1"};
  EXPECT_EQ(stamp.CacheKeyFingerprintForComponents(active),
            stamp.CacheKeyFingerprint());
}

TEST(PjRtBuildStampRegistryTest, RegisterAndLookupCompilerSuccessful) {
  PjRtBuildStampRegistry registry;
  int factory_calls = 0;
  auto factory =
      [&]() -> absl::StatusOr<std::shared_ptr<const PjRtBuildStamp>> {
    factory_calls++;
    PjRtBuildStamp::Options opt;
    opt.build_timestamp = absl::FromUnixSeconds(1000);
    opt.vcs_revision = "rev1";
    opt.build_label = "label1";
    opt.build_cl = 500;
    opt.base_source_hash = "source_hash_abc";
    ABSL_ASSIGN_OR_RETURN(PjRtBuildStamp stamp,
                          PjRtBuildStamp::Create(std::move(opt)));
    return std::make_shared<const PjRtBuildStamp>(std::move(stamp));
  };

  ASSERT_OK(registry.RegisterCompilerBuildStampFactory("cuda", "custom_variant",
                                                       factory));

  ASSERT_OK_AND_ASSIGN(
      auto stamp, registry.GetBuildStampForCompiler("cuda", "custom_variant"));
  EXPECT_EQ(stamp->build_cl(), 500);
  EXPECT_EQ(stamp->base_source_hash(), "source_hash_abc");
  EXPECT_EQ(factory_calls, 1);

  // Subsequent lookups must be memoized without re-invoking the factory.
  ASSERT_OK_AND_ASSIGN(
      auto stamp2, registry.GetBuildStampForCompiler("cuda", "custom_variant"));
  EXPECT_EQ(stamp2->build_cl(), 500);
  EXPECT_EQ(factory_calls, 1);

  // ClearCacheForTesting drops memoized stamp while keeping factory registered.
  registry.ClearCacheForTesting();
  ASSERT_OK_AND_ASSIGN(
      auto stamp3, registry.GetBuildStampForCompiler("cuda", "custom_variant"));
  EXPECT_EQ(stamp3->build_cl(), 500);
  EXPECT_EQ(factory_calls, 2);
}

TEST(PjRtBuildStampRegistryTest, DuplicateRegistrationFails) {
  PjRtBuildStampRegistry registry;
  auto factory = []() -> absl::StatusOr<std::shared_ptr<const PjRtBuildStamp>> {
    PjRtBuildStamp::Options opt;
    opt.base_source_hash = "h";
    ABSL_ASSIGN_OR_RETURN(PjRtBuildStamp stamp,
                          PjRtBuildStamp::Create(std::move(opt)));
    return std::make_shared<const PjRtBuildStamp>(std::move(stamp));
  };

  ASSERT_OK(
      registry.RegisterCompilerBuildStampFactory("tpu", "variant_a", factory));
  EXPECT_THAT(
      registry.RegisterCompilerBuildStampFactory("tpu", "variant_a", factory),
      StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(PjRtBuildStampRegistryTest, LookupNonExistentFails) {
  PjRtBuildStampRegistry registry;
  EXPECT_THAT(registry.GetBuildStampForCompiler("tpu", "unknown_variant"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(PjRtBuildStampRegistryTest, VariantPickerResolutionForPlatform) {
  PjRtBuildStampRegistry registry;
  auto linked_factory =
      []() -> absl::StatusOr<std::shared_ptr<const PjRtBuildStamp>> {
    PjRtBuildStamp::Options opt;
    opt.build_cl = 100;
    opt.base_source_hash = "linked_hash";
    ABSL_ASSIGN_OR_RETURN(PjRtBuildStamp stamp,
                          PjRtBuildStamp::Create(std::move(opt)));
    return std::make_shared<const PjRtBuildStamp>(std::move(stamp));
  };
  auto forge_factory =
      []() -> absl::StatusOr<std::shared_ptr<const PjRtBuildStamp>> {
    PjRtBuildStamp::Options opt;
    opt.build_cl = 200;
    opt.base_source_hash = "forge_hash";
    ABSL_ASSIGN_OR_RETURN(PjRtBuildStamp stamp,
                          PjRtBuildStamp::Create(std::move(opt)));
    return std::make_shared<const PjRtBuildStamp>(std::move(stamp));
  };

  ASSERT_OK(registry.RegisterPlatformBuildStampFactory("tpu", linked_factory));
  ASSERT_OK(registry.RegisterCompilerBuildStampFactory("tpu", "forge",
                                                       forge_factory));

  // Without a picker, defaults to linked variant.
  ASSERT_OK_AND_ASSIGN(auto default_stamp,
                       registry.GetBuildStampForPlatform("tpu"));
  EXPECT_EQ(default_stamp->base_source_hash(), "linked_hash");

  // Register a variant picker that selects "forge".
  PjRtBuildStampRegistry registry_with_picker;
  ASSERT_OK(registry_with_picker.RegisterPlatformBuildStampFactory(
      "tpu", linked_factory));
  ASSERT_OK(registry_with_picker.RegisterCompilerBuildStampFactory(
      "tpu", "forge", forge_factory));
  registry_with_picker.RegisterVariantPicker(
      "tpu", []() { return std::string("forge"); });

  ASSERT_OK_AND_ASSIGN(auto picked_stamp,
                       registry_with_picker.GetBuildStampForPlatform("tpu"));
  EXPECT_EQ(picked_stamp->base_source_hash(), "forge_hash");
}

TEST(PjRtBuildStampRegistryTest, ComponentFingerprintProviderIntegration) {
  PjRtBuildStampRegistry registry;
  auto base_factory =
      []() -> absl::StatusOr<std::shared_ptr<const PjRtBuildStamp>> {
    PjRtBuildStamp::Options opt;
    opt.base_source_hash = "base_cuda_compiler";
    ABSL_ASSIGN_OR_RETURN(PjRtBuildStamp stamp,
                          PjRtBuildStamp::Create(std::move(opt)));
    return std::make_shared<const PjRtBuildStamp>(std::move(stamp));
  };

  ASSERT_OK(registry.RegisterPlatformBuildStampFactory("cuda", base_factory));

  // Dynamically register a component provider for GPU custom calls.
  ASSERT_OK(registry.RegisterComponentFingerprintProvider(
      "cuda", []() -> absl::StatusOr<std::vector<PjRtComponentBuildStamp>> {
        return std::vector<PjRtComponentBuildStamp>{PjRtComponentBuildStamp(
            "xla.gpu.native_custom_call", "fused_attention",
            "//gpu:fused_attention", "attention_sha256")};
      }));

  ASSERT_OK_AND_ASSIGN(auto stamp, registry.GetBuildStampForPlatform("cuda"));
  EXPECT_EQ(stamp->base_source_hash(), "base_cuda_compiler");
  ASSERT_EQ(stamp->components().size(), 1);
  EXPECT_EQ(stamp->components()[0].name, "fused_attention");
  EXPECT_NE(stamp->CacheKeyFingerprint(), "base_cuda_compiler");
}

}  // namespace
}  // namespace xla
