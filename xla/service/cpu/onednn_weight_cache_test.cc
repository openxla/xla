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

#include "xla/service/cpu/onednn_weight_cache.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/tied_ref.h"

namespace xla::cpu {
namespace {

using LookupKind = OneDnnWeightCache::LookupKind;

OneDnnWeightCacheKey MakeKey(const std::shared_ptr<tsl::TiedAny>& identity,
                             int64_t columns = 2) {
  const dnnl::memory::dims dims = {1, columns};
  return {identity.get(),
          dnnl::memory::desc(dims, dnnl::memory::data_type::f32,
                             dnnl::memory::format_tag::ab),
          dnnl::memory::desc(dims, dnnl::memory::data_type::f32,
                             dnnl::memory::format_tag::ba),
          identity};
}

TEST(OneDnnWeightCacheTest,
     ReusesEquivalentDescriptorsAndSeparatesLayoutsSlicesAndGenerations) {
  OneDnnWeightCache cache(/*capacity_bytes=*/120);
  auto owner = std::make_shared<tsl::TiedAny>();
  auto new_generation = std::make_shared<tsl::TiedAny>();
  std::array<float, 12> source{};
  const dnnl::memory::dims dims = {2, 3};
  const dnnl::memory::desc row_major(dims, dnnl::memory::data_type::f32,
                                     dnnl::memory::format_tag::ab);
  const dnnl::memory::desc column_major(dims, dnnl::memory::data_type::f32,
                                        dnnl::memory::format_tag::ba);
  std::array<OneDnnWeightCacheKey, 5> keys = {{
      {source.data(), row_major, row_major, owner},
      {source.data(), column_major, row_major, owner},
      {source.data(), row_major, column_major, owner},
      // Slices share an identity, but their addresses must distinguish them.
      {source.data() + 6, row_major, row_major, owner},
      {source.data(), row_major, row_major, new_generation},
  }};
  for (const auto& key : keys) {
    auto miss = cache.LookupOrCreate(key);
    ASSERT_EQ(miss.kind, LookupKind::kMiss);
    ASSERT_NE(miss.weights, nullptr);
    cache.Complete(miss.weights, true);
    auto hit = cache.LookupOrCreate(key);
    EXPECT_EQ(hit.kind, LookupKind::kHit);
    EXPECT_EQ(hit.weights, miss.weights);
  }

  const dnnl::memory::desc strided(dims, dnnl::memory::data_type::f32,
                                   dnnl::memory::dims{3, 1});
  OneDnnWeightCacheKey strided_key{source.data(), strided, strided, owner};
  EXPECT_EQ(cache.LookupOrCreate(strided_key).kind, LookupKind::kHit);
}

TEST(OneDnnWeightCacheTest, EvictsLeastRecentlyUsedEntry) {
  OneDnnWeightCache cache(/*capacity_bytes=*/24);
  auto owner_a = std::make_shared<tsl::TiedAny>();
  auto owner_b = std::make_shared<tsl::TiedAny>();
  auto owner_c = std::make_shared<tsl::TiedAny>();
  auto owner_large = std::make_shared<tsl::TiedAny>();
  std::array<OneDnnWeightCacheKey, 3> keys = {
      MakeKey(owner_a), MakeKey(owner_b), MakeKey(owner_c)};
  OneDnnWeightCacheKey large_key = MakeKey(owner_large, 4);
  OneDnnWeightCache disabled(0);
  EXPECT_EQ(disabled.LookupOrCreate(keys[0]).kind, LookupKind::kBypass);
  EXPECT_EQ(cache.LookupOrCreate(MakeKey(owner_large, 7)).kind,
            LookupKind::kBypass);
  std::array<std::weak_ptr<OneDnnPackedWeights>, 3> packed;
  for (size_t index = 0; index < keys.size(); ++index) {
    auto miss = cache.LookupOrCreate(keys[index]);
    ASSERT_EQ(miss.kind, LookupKind::kMiss);
    ASSERT_NE(miss.weights, nullptr);
    packed[index] = miss.weights;
    cache.Complete(miss.weights, true);
  }

  EXPECT_EQ(cache.stats().live_bytes, 24);
  ASSERT_EQ(cache.LookupOrCreate(keys[0]).kind, LookupKind::kHit);
  auto large = cache.LookupOrCreate(large_key);
  ASSERT_EQ(large.kind, LookupKind::kMiss);
  ASSERT_NE(large.weights, nullptr);
  EXPECT_EQ(cache.stats().live_bytes, 24);
  EXPECT_FALSE(packed[0].expired());
  EXPECT_TRUE(packed[1].expired());
  EXPECT_TRUE(packed[2].expired());
  cache.Complete(large.weights, true);
  large.weights.reset();

  EXPECT_EQ(cache.LookupOrCreate(large_key).kind, LookupKind::kHit);
  EXPECT_EQ(cache.LookupOrCreate(keys[0]).kind, LookupKind::kHit);
}

TEST(OneDnnWeightCacheTest,
     ChargesRetainedAllocationsAndReleasesSourceTiedWeights) {
  OneDnnWeightCache cache(8);
  auto owner = std::make_shared<tsl::TiedAny>();
  auto key = MakeKey(owner);
  auto failed = cache.LookupOrCreate(key);
  ASSERT_EQ(failed.kind, LookupKind::kMiss);
  ASSERT_NE(failed.weights, nullptr);
  EXPECT_EQ(cache.LookupOrCreate(key).kind, LookupKind::kBypass);
  cache.Complete(failed.weights, false);
  EXPECT_EQ(cache.stats().live_bytes, 8);
  EXPECT_EQ(cache.LookupOrCreate(key).kind, LookupKind::kBypass);
  failed.weights.reset();
  EXPECT_EQ(cache.stats().live_bytes, 0);
  auto retry = cache.LookupOrCreate(key);
  ASSERT_EQ(retry.kind, LookupKind::kMiss);
  ASSERT_NE(retry.weights, nullptr);
  cache.Complete(retry.weights, true);
  std::weak_ptr<OneDnnPackedWeights> packed = retry.weights;
  retry.weights.reset();
  EXPECT_FALSE(packed.expired());
  owner.reset();
  EXPECT_TRUE(packed.expired());
  EXPECT_EQ(cache.stats().live_bytes, 0);
}

}  // namespace
}  // namespace xla::cpu
