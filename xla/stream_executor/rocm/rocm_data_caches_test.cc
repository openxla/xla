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

#include "xla/stream_executor/rocm/rocm_data_caches.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>

#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/smi_util.h"

namespace stream_executor::gpu {
namespace {

using ::testing::ElementsAre;

constexpr int64_t kKiB = 1024;
constexpr int64_t kMiB = 1024 * kKiB;
// DefaultL1CacheSizePerCore() for RDNA.
constexpr int64_t kDefaultL1 = 2 * kKiB;

TEST(BuildDataCacheHierarchyTest, Mi350x) {
  // MI350X-like. The scalar L1 is dropped and the L2 count comes from the XCDs.
  const SmiDataCache caches[] = {
      {/*level=*/1, /*size_bytes=*/32 * kKiB, /*num_instances=*/256},
      {/*level=*/1, /*size_bytes=*/16 * kKiB, /*num_instances=*/128},
      {/*level=*/2, /*size_bytes=*/4 * kMiB, /*num_instances=*/1},
      {/*level=*/3, /*size_bytes=*/256 * kMiB, /*num_instances=*/1},
  };

  EXPECT_THAT(
      BuildDataCacheHierarchy(caches, /*core_count=*/256, /*xcc_count=*/8,
                              /*hip_l2_cache_size=*/0,
                              /*default_l1_cache_size=*/kDefaultL1),
      ElementsAre(DataCacheInfo{1, 32 * kKiB, 256},
                  DataCacheInfo{2, 4 * kMiB, 8},
                  DataCacheInfo{3, 256 * kMiB, 1}));
}

TEST(BuildDataCacheHierarchyTest, Rdna3KeepsGl1AtLevelOne) {
  // RDNA3-like, shuffled. The GL1 is kept, the scalar cache dropped.
  const SmiDataCache caches[] = {
      {/*level=*/1, /*size_bytes=*/256 * kKiB, /*num_instances=*/12},
      {/*level=*/3, /*size_bytes=*/96 * kMiB, /*num_instances=*/1},
      {/*level=*/2, /*size_bytes=*/6 * kMiB, /*num_instances=*/1},
      {/*level=*/1, /*size_bytes=*/32 * kKiB, /*num_instances=*/96},
      {/*level=*/1, /*size_bytes=*/16 * kKiB, /*num_instances=*/48},
  };

  EXPECT_THAT(
      BuildDataCacheHierarchy(caches, /*core_count=*/96, /*xcc_count=*/1,
                              /*hip_l2_cache_size=*/0,
                              /*default_l1_cache_size=*/kDefaultL1),
      ElementsAre(
          DataCacheInfo{1, 32 * kKiB, 96}, DataCacheInfo{1, 256 * kKiB, 12},
          DataCacheInfo{2, 6 * kMiB, 1}, DataCacheInfo{3, 96 * kMiB, 1}));
}

TEST(BuildDataCacheHierarchyTest, DropsHarvestedScalarGroups) {
  // Scalar caches of half-disabled CU pairs form their own small group.
  const SmiDataCache caches[] = {
      {/*level=*/1, /*size_bytes=*/16 * kKiB, /*num_instances=*/8},
      {/*level=*/1, /*size_bytes=*/32 * kKiB, /*num_instances=*/304},
      {/*level=*/1, /*size_bytes=*/16 * kKiB, /*num_instances=*/148},
  };

  EXPECT_THAT(
      BuildDataCacheHierarchy(caches, /*core_count=*/304, /*xcc_count=*/8,
                              /*hip_l2_cache_size=*/0,
                              /*default_l1_cache_size=*/kDefaultL1),
      ElementsAre(DataCacheInfo{1, 32 * kKiB, 304}));
}

TEST(BuildDataCacheHierarchyTest, Gfx9CapsFoldedScalarCaches) {
  // rocm_smi folds orphaned gfx9 scalar caches into the vector L1 group.
  const SmiDataCache caches[] = {
      {/*level=*/1, /*size_bytes=*/16 * kKiB, /*num_instances=*/112},
      {/*level=*/1, /*size_bytes=*/16 * kKiB, /*num_instances=*/54},
      {/*level=*/2, /*size_bytes=*/8 * kMiB, /*num_instances=*/1},
  };

  EXPECT_THAT(
      BuildDataCacheHierarchy(caches, /*core_count=*/110, /*xcc_count=*/1,
                              /*hip_l2_cache_size=*/0,
                              /*default_l1_cache_size=*/kDefaultL1),
      ElementsAre(DataCacheInfo{1, 16 * kKiB, 110},
                  DataCacheInfo{2, 8 * kMiB, 1}));
}

TEST(BuildDataCacheHierarchyTest, SkipsInvalidEntries) {
  // Only the default L1 remains.
  const SmiDataCache caches[] = {
      {/*level=*/0, /*size_bytes=*/16 * kKiB, /*num_instances=*/4},
      {/*level=*/1, /*size_bytes=*/0, /*num_instances=*/4},
  };

  EXPECT_THAT(BuildDataCacheHierarchy(caches, /*core_count=*/4, /*xcc_count=*/1,
                                      /*hip_l2_cache_size=*/0,
                                      /*default_l1_cache_size=*/kDefaultL1),
              ElementsAre(DataCacheInfo{1, kDefaultL1, 4}));
}

TEST(BuildDataCacheHierarchyTest, DefaultL1FillsInWhenSmiHasNone) {
  // The GL1 is not per-CU.
  const SmiDataCache caches[] = {
      {/*level=*/1, /*size_bytes=*/256 * kKiB, /*num_instances=*/12},
  };

  EXPECT_THAT(
      BuildDataCacheHierarchy(caches, /*core_count=*/96, /*xcc_count=*/1,
                              /*hip_l2_cache_size=*/6 * kMiB,
                              /*default_l1_cache_size=*/kDefaultL1),
      ElementsAre(DataCacheInfo{1, kDefaultL1, 96},
                  DataCacheInfo{1, 256 * kKiB, 12},
                  DataCacheInfo{2, 6 * kMiB, 1}));
}

TEST(BuildDataCacheHierarchyTest, HipL2SizeWinsOverSmi) {
  const SmiDataCache caches[] = {
      {/*level=*/1, /*size_bytes=*/32 * kKiB, /*num_instances=*/256},
      {/*level=*/2, /*size_bytes=*/32 * kMiB, /*num_instances=*/1},
  };

  EXPECT_THAT(
      BuildDataCacheHierarchy(caches, /*core_count=*/256, /*xcc_count=*/8,
                              /*hip_l2_cache_size=*/4 * kMiB,
                              /*default_l1_cache_size=*/kDefaultL1),
      ElementsAre(DataCacheInfo{1, 32 * kKiB, 256},
                  DataCacheInfo{2, 4 * kMiB, 8}));
}

TEST(BuildDataCacheHierarchyTest, HipL2FillsInWhenSmiHasNone) {
  const SmiDataCache caches[] = {
      {/*level=*/1, /*size_bytes=*/32 * kKiB, /*num_instances=*/256},
      {/*level=*/3, /*size_bytes=*/256 * kMiB, /*num_instances=*/1},
  };

  EXPECT_THAT(
      BuildDataCacheHierarchy(caches, /*core_count=*/256, /*xcc_count=*/8,
                              /*hip_l2_cache_size=*/4 * kMiB,
                              /*default_l1_cache_size=*/kDefaultL1),
      ElementsAre(DataCacheInfo{1, 32 * kKiB, 256},
                  DataCacheInfo{2, 4 * kMiB, 8},
                  DataCacheInfo{3, 256 * kMiB, 1}));
}

TEST(GetRocmDataCachesTest, InvalidBdfReportsDefaultL1) {
  EXPECT_THAT(GetRocmDataCaches("invalid", /*core_count=*/1, /*xcc_count=*/1,
                                /*hip_l2_cache_size=*/0,
                                /*default_l1_cache_size=*/kDefaultL1),
              ElementsAre(DataCacheInfo{1, kDefaultL1, 1}));
}

TEST(GetRocmDataCachesTest, InvalidBdfStillReportsHipL2) {
  EXPECT_THAT(GetRocmDataCaches("invalid", /*core_count=*/256,
                                /*xcc_count=*/8,
                                /*hip_l2_cache_size=*/4 * kMiB,
                                /*default_l1_cache_size=*/32 * kKiB),
              ElementsAre(DataCacheInfo{1, 32 * kKiB, 256},
                          DataCacheInfo{2, 4 * kMiB, 8}));
}

}  // namespace
}  // namespace stream_executor::gpu
