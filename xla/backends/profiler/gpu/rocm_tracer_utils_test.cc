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
  AnnotationMap map(/*max_size=*/2);
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
  AnnotationMap map(/*max_size=*/1);
  for (uint32_t i = 0; i < 100; ++i) {
    map.Add(i, "same_annotation");
  }
  EXPECT_EQ(map.LookUp(0), "same_annotation");
  EXPECT_TRUE(map.LookUp(99).empty());
}

TEST(AnnotationMapTest, SetMaxSizeRaisesTheCapForSubsequentAdds) {
  // The path RocmTracer::Enable takes: Clear(), then SetMaxSize() with the
  // value derived from the flag or from gpu_max_annotation_strings.
  AnnotationMap map(/*max_size=*/1);
  map.Add(1, "first");
  map.Add(2, "second");
  ASSERT_TRUE(map.LookUp(2).empty()) << "precondition: cap of 1 is in force";

  map.Clear();
  map.SetMaxSize(3);
  map.Add(1, "first");
  map.Add(2, "second");
  map.Add(3, "third");

  EXPECT_EQ(map.LookUp(1), "first");
  EXPECT_EQ(map.LookUp(2), "second");
  EXPECT_EQ(map.LookUp(3), "third");
}

TEST(AnnotationMapTest, SetMaxSizeLowersTheCapForSubsequentAdds) {
  AnnotationMap map(/*max_size=*/100);
  map.SetMaxSize(1);
  map.Add(1, "first");
  map.Add(2, "second");

  EXPECT_EQ(map.LookUp(1), "first");
  EXPECT_TRUE(map.LookUp(2).empty());
}

TEST(AnnotationMapTest, SetMaxSizeZeroRetainsNothing) {
  // 0 means "retain no annotation strings", not "unlimited". RocmTracer::Enable
  // passes gpu_max_annotation_strings straight through, so this is the value a
  // user gets by asking for zero -- it must not be reinterpreted, and it must
  // not silently leave the previous capacity in force.
  AnnotationMap map(/*max_size=*/8);
  map.SetMaxSize(0);
  map.Add(1, "first");

  EXPECT_TRUE(map.LookUp(1).empty());
}

TEST(AnnotationMapTest, ClearPreservesTheConfiguredSize) {
  // The invariant RocmTracer::Enable's Clear-then-SetMaxSize ordering rests on:
  // Clear() drops entries but is not a reset of the capacity. If it ever became
  // one, Enable would still look correct while every session after the first
  // silently ran at the constructor default.
  AnnotationMap map(/*max_size=*/8);
  map.SetMaxSize(1);
  map.Clear();
  map.Add(1, "first");
  map.Add(2, "second");

  EXPECT_EQ(map.LookUp(1), "first");
  EXPECT_TRUE(map.LookUp(2).empty()) << "capacity of 1 must survive Clear()";
}

TEST(AnnotationMapTest, EmptyAnnotationIsIgnored) {
  // Empty annotations must not consume capacity; the ROCTX path can produce
  // them and they would otherwise crowd out real ones.
  AnnotationMap map(/*max_size=*/1);
  map.Add(1, "");
  map.Add(2, "real");

  EXPECT_TRUE(map.LookUp(1).empty());
  EXPECT_EQ(map.LookUp(2), "real");
}

TEST(AnnotationMapTest, ClearDropsEntries) {
  AnnotationMap map(/*max_size=*/8);
  map.Add(1, "first");
  ASSERT_EQ(map.LookUp(1), "first");

  map.Clear();
  EXPECT_TRUE(map.LookUp(1).empty());
}

}  // namespace
}  // namespace profiler
}  // namespace xla
