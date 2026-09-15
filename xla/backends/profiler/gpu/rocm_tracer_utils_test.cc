/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"

#include <cstdint>
#include <string>  // AnnotationMap::Add takes const std::string&

#include <gtest/gtest.h>

namespace xla {
namespace profiler {
namespace {

// The AnnotationMap capacity is what gpu_max_annotation_strings ultimately
// controls, so it needs a test that does not require a GPU. rocm_tracer_utils
// has no ROCm dependency, which is what makes this file host-only.

TEST(AnnotationMapTest, HonoursConfiguredSize) {
  AnnotationMap map;
  map.Reset(2);
  map.Add(1, "first");
  map.Add(2, "second");
  map.Add(3, "third");

  EXPECT_EQ(map.LookUp(1), "first");
  EXPECT_EQ(map.LookUp(2), "second");
  // Over capacity: silently not retained. Add() is on the callback hot path
  // and deliberately does not log per-drop.
  EXPECT_TRUE(map.LookUp(3).empty());
}

TEST(AnnotationMapTest, FullStringSetAlsoStopsCorrelatingKnownStrings) {
  // Characterisation test, not an endorsement. The size check gates the whole
  // Add(), correlation-map insertion included, so once the distinct-string set
  // is full, later events are left uncorrelated even when their annotation is
  // one the map already holds. Repeating an annotation is therefore *not*
  // free at the boundary.
  //
  // CUPTI does the same thing (AnnotationMap::Add in cupti_buffer_events.cc
  // gates on annotation_deduper.Size() < max_size_ before correlation_map
  // insertion),
  // so this is shared cross-vendor behaviour rather than a ROCm defect, and
  // changing it is out of scope here. Pinned so that a future change to either
  // side is a deliberate one.
  AnnotationMap map;
  map.Reset(1);
  for (uint32_t i = 0; i < 100; ++i) {
    map.Add(i, "same_annotation");
  }
  EXPECT_EQ(map.LookUp(0), "same_annotation");
  EXPECT_TRUE(map.LookUp(99).empty());
}

TEST(AnnotationMapTest, ResetRaisesTheCapAndClearsEntries) {
  // Reset atomically clears entries and sets a new capacity -- the path
  // RocmTracer::Enable takes at the start of each session.
  AnnotationMap map;
  map.Reset(1);
  map.Add(1, "first");
  ASSERT_TRUE(map.LookUp(2).empty()) << "precondition: cap of 1 is in force";

  map.Reset(3);  // clears "first" and raises the cap in one step
  map.Add(1, "first");
  map.Add(2, "second");
  map.Add(3, "third");

  EXPECT_EQ(map.LookUp(1), "first");
  EXPECT_EQ(map.LookUp(2), "second");
  EXPECT_EQ(map.LookUp(3), "third");
}

TEST(AnnotationMapTest, ResetLowersTheCapForSubsequentAdds) {
  AnnotationMap map;
  map.Reset(100);
  map.Reset(1);
  map.Add(1, "first");
  map.Add(2, "second");

  EXPECT_EQ(map.LookUp(1), "first");
  EXPECT_TRUE(map.LookUp(2).empty());
}

TEST(AnnotationMapTest, ResetToZeroRetainsNothing) {
  // 0 means "retain no annotation strings", not "unlimited". RocmTracer::Enable
  // passes gpu_max_annotation_strings straight through, so this is the value a
  // user gets by asking for zero -- it must not be reinterpreted, and it must
  // not silently leave the previous capacity in force.
  AnnotationMap map;
  map.Reset(8);
  map.Reset(0);
  map.Add(1, "first");

  EXPECT_TRUE(map.LookUp(1).empty());
}

TEST(AnnotationMapTest, ResetClearsOldEntriesBeforeNewSession) {
  // Reset drops all entries from the old session and sets the new capacity in
  // one step. An entry that was live before Reset must not be visible after.
  AnnotationMap map;
  map.Reset(8);
  map.Add(1, "old_entry");
  ASSERT_EQ(map.LookUp(1), "old_entry");

  map.Reset(2);  // new session starts; old_entry must be gone
  EXPECT_TRUE(map.LookUp(1).empty()) << "Reset must clear old entries";

  map.Add(2, "new_a");
  map.Add(3, "new_b");
  map.Add(4, "past_cap");  // cap=2, so this is dropped
  EXPECT_EQ(map.LookUp(2), "new_a");
  EXPECT_EQ(map.LookUp(3), "new_b");
  EXPECT_TRUE(map.LookUp(4).empty());
}

TEST(AnnotationMapTest, EmptyAnnotationIsIgnored) {
  // Empty annotations must not consume capacity; the ROCTX path can produce
  // them and they would otherwise crowd out real ones.
  AnnotationMap map;
  map.Reset(1);
  map.Add(1, "");
  map.Add(2, "real");

  EXPECT_TRUE(map.LookUp(1).empty());
  EXPECT_EQ(map.LookUp(2), "real");
}

TEST(AnnotationMapTest, ResetDropsEntries) {
  AnnotationMap map;
  map.Reset(8);
  map.Add(1, "first");
  ASSERT_EQ(map.LookUp(1), "first");

  map.Reset(8);  // same cap; entries must be gone
  EXPECT_TRUE(map.LookUp(1).empty());
}

}  // namespace
}  // namespace profiler
}  // namespace xla
