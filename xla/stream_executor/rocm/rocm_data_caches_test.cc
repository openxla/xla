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

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/smi_util.h"

namespace stream_executor::gpu {
namespace {

using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

constexpr int64_t kKiB = 1024;
constexpr int64_t kMiB = 1024 * kKiB;

TEST(BuildDataCacheHierarchyTest, Mi355x) {
  // MI355X data caches as amd-smi reports them. The scalar caches are dropped
  // and the L2 count comes from the XCDs.
  const SmiDataCache caches[] = {
      {/*level=*/1, /*size_bytes=*/32 * kKiB, /*num_instances=*/256},
      {/*level=*/1, /*size_bytes=*/16 * kKiB, /*num_instances=*/112},
      {/*level=*/1, /*size_bytes=*/16 * kKiB, /*num_instances=*/32},
      {/*level=*/2, /*size_bytes=*/4 * kMiB, /*num_instances=*/1},
      {/*level=*/3, /*size_bytes=*/256 * kMiB, /*num_instances=*/1},
  };

  EXPECT_THAT(
      BuildDataCacheHierarchy(caches, /*core_count=*/256, /*xcc_count=*/8,
                              /*hip_l2_cache_size=*/0),
      UnorderedElementsAre(DataCacheInfo{1, 32 * kKiB, 256},
                           DataCacheInfo{2, 4 * kMiB, 8},
                           DataCacheInfo{3, 256 * kMiB, 1}));
}

TEST(BuildDataCacheHierarchyTest, Gfx1100KeepsGl1AtLevelOne) {
  // gfx1100 (RX 7900) data caches as amd-smi reports them, shuffled. The GL1
  // is kept, the scalar cache dropped.
  const SmiDataCache caches[] = {
      {/*level=*/1, /*size_bytes=*/256 * kKiB, /*num_instances=*/12},
      {/*level=*/3, /*size_bytes=*/96 * kMiB, /*num_instances=*/1},
      {/*level=*/2, /*size_bytes=*/6 * kMiB, /*num_instances=*/1},
      {/*level=*/1, /*size_bytes=*/32 * kKiB, /*num_instances=*/96},
      {/*level=*/1, /*size_bytes=*/16 * kKiB, /*num_instances=*/48},
  };

  EXPECT_THAT(
      BuildDataCacheHierarchy(caches, /*core_count=*/96, /*xcc_count=*/1,
                              /*hip_l2_cache_size=*/0),
      UnorderedElementsAre(
          DataCacheInfo{1, 32 * kKiB, 96}, DataCacheInfo{1, 256 * kKiB, 12},
          DataCacheInfo{2, 6 * kMiB, 1}, DataCacheInfo{3, 96 * kMiB, 1}));
}

TEST(BuildDataCacheHierarchyTest, DropsHarvestedScalarGroups) {
  // MI325X level 1 data caches as amd-smi reports them. Scalar caches of
  // half-disabled CU pairs form their own small group.
  const SmiDataCache caches[] = {
      {/*level=*/1, /*size_bytes=*/32 * kKiB, /*num_instances=*/304},
      {/*level=*/1, /*size_bytes=*/16 * kKiB, /*num_instances=*/144},
      {/*level=*/1, /*size_bytes=*/16 * kKiB, /*num_instances=*/16},
  };

  EXPECT_THAT(
      BuildDataCacheHierarchy(caches, /*core_count=*/304, /*xcc_count=*/8,
                              /*hip_l2_cache_size=*/0),
      UnorderedElementsAre(DataCacheInfo{1, 32 * kKiB, 304}));
}

TEST(BuildDataCacheHierarchyTest, Gfx9CapsFoldedScalarCaches) {
  const SmiDataCache caches[] = {
      {/*level=*/1, /*size_bytes=*/16 * kKiB, /*num_instances=*/112},
      {/*level=*/1, /*size_bytes=*/16 * kKiB, /*num_instances=*/48},
      {/*level=*/2, /*size_bytes=*/8 * kMiB, /*num_instances=*/1},
  };

  EXPECT_THAT(
      BuildDataCacheHierarchy(caches, /*core_count=*/104, /*xcc_count=*/1,
                              /*hip_l2_cache_size=*/0),
      UnorderedElementsAre(DataCacheInfo{1, 16 * kKiB, 104},
                           DataCacheInfo{2, 8 * kMiB, 1}));
}

TEST(BuildDataCacheHierarchyTest, TiedL1CountsPickTheLargerCache) {
  // A scalar cache listed first with as many instances as the vector L1.
  const SmiDataCache caches[] = {
      {/*level=*/1, /*size_bytes=*/16 * kKiB, /*num_instances=*/96},
      {/*level=*/1, /*size_bytes=*/32 * kKiB, /*num_instances=*/96},
  };

  EXPECT_THAT(
      BuildDataCacheHierarchy(caches, /*core_count=*/96, /*xcc_count=*/1,
                              /*hip_l2_cache_size=*/0),
      UnorderedElementsAre(DataCacheInfo{1, 32 * kKiB, 96}));
}

TEST(BuildDataCacheHierarchyTest, Gfx1150HasNoL3) {
  // Strix Point (gfx1150) data caches as amd-smi reports them. Like gfx1100
  // it has a GL1, but as an APU no Infinity Cache.
  const SmiDataCache caches[] = {
      {/*level=*/1, /*size_bytes=*/32 * kKiB, /*num_instances=*/16},
      {/*level=*/1, /*size_bytes=*/16 * kKiB, /*num_instances=*/8},
      {/*level=*/1, /*size_bytes=*/256 * kKiB, /*num_instances=*/2},
      {/*level=*/2, /*size_bytes=*/2 * kMiB, /*num_instances=*/1},
  };

  EXPECT_THAT(
      BuildDataCacheHierarchy(caches, /*core_count=*/16, /*xcc_count=*/1,
                              /*hip_l2_cache_size=*/0),
      UnorderedElementsAre(DataCacheInfo{1, 32 * kKiB, 16},
                           DataCacheInfo{1, 256 * kKiB, 2},
                           DataCacheInfo{2, 2 * kMiB, 1}));
}

TEST(BuildDataCacheHierarchyTest, SkipsInvalidEntries) {
  const SmiDataCache caches[] = {
      {/*level=*/0, /*size_bytes=*/16 * kKiB, /*num_instances=*/4},
      {/*level=*/1, /*size_bytes=*/0, /*num_instances=*/4},
  };

  EXPECT_THAT(BuildDataCacheHierarchy(caches, /*core_count=*/4, /*xcc_count=*/1,
                                      /*hip_l2_cache_size=*/0),
              IsEmpty());
}

TEST(BuildDataCacheHierarchyTest, KeepsSmiL1WithFewerInstancesThanCores) {
  // The SMI vector L1 is kept as reported.
  const SmiDataCache caches[] = {
      {/*level=*/1, /*size_bytes=*/32 * kKiB, /*num_instances=*/90},
  };

  EXPECT_THAT(
      BuildDataCacheHierarchy(caches, /*core_count=*/96, /*xcc_count=*/1,
                              /*hip_l2_cache_size=*/0),
      UnorderedElementsAre(DataCacheInfo{1, 32 * kKiB, 90}));
}

TEST(BuildDataCacheHierarchyTest, HipL2SizeWinsOverSmi) {
  const SmiDataCache caches[] = {
      {/*level=*/1, /*size_bytes=*/32 * kKiB, /*num_instances=*/256},
      {/*level=*/2, /*size_bytes=*/32 * kMiB, /*num_instances=*/1},
  };

  EXPECT_THAT(
      BuildDataCacheHierarchy(caches, /*core_count=*/256, /*xcc_count=*/8,
                              /*hip_l2_cache_size=*/4 * kMiB),
      UnorderedElementsAre(DataCacheInfo{1, 32 * kKiB, 256},
                           DataCacheInfo{2, 4 * kMiB, 8}));
}

TEST(BuildDataCacheHierarchyTest, HipL2FillsInWhenSmiHasNone) {
  const SmiDataCache caches[] = {
      {/*level=*/1, /*size_bytes=*/32 * kKiB, /*num_instances=*/256},
      {/*level=*/3, /*size_bytes=*/256 * kMiB, /*num_instances=*/1},
  };

  EXPECT_THAT(
      BuildDataCacheHierarchy(caches, /*core_count=*/256, /*xcc_count=*/8,
                              /*hip_l2_cache_size=*/4 * kMiB),
      UnorderedElementsAre(DataCacheInfo{1, 32 * kKiB, 256},
                           DataCacheInfo{2, 4 * kMiB, 8},
                           DataCacheInfo{3, 256 * kMiB, 1}));
}

TEST(GetRocmDataCachesTest, InvalidBdfReportsNothing) {
  EXPECT_THAT(GetRocmDataCaches("invalid", /*core_count=*/1, /*xcc_count=*/1,
                                /*hip_l2_cache_size=*/0),
              IsEmpty());
}

TEST(GetRocmDataCachesTest, InvalidBdfStillReportsHipL2) {
  EXPECT_THAT(GetRocmDataCaches("invalid", /*core_count=*/256,
                                /*xcc_count=*/8,
                                /*hip_l2_cache_size=*/4 * kMiB),
              UnorderedElementsAre(DataCacheInfo{2, 4 * kMiB, 8}));
}

}  // namespace
}  // namespace stream_executor::gpu
