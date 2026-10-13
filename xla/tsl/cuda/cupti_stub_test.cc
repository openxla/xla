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

#include <gtest/gtest.h>

// Strong references make missing stubs fail at link time. These declarations
// intentionally use placeholder signatures: only take their addresses, never
// call them. This avoids requiring newer CUPTI headers that declare the V2
// APIs.
extern "C" {
void cuptiActivityDisable_v2();
void cuptiActivityEnable_v2();
void cuptiActivityGetNextRecord_v2();
void cuptiActivityGetNumDroppedRecords_v2();
void cuptiActivityRegisterCallbacks_v2();
void cuptiActivitySetAttribute_v2();
void cuptiGetTimestamp_v2();
void cuptiSubscribe_v2();
}

namespace {

struct CuptiSymbol {
  const char* name;
  // Force a linker reference even when the compiler knows function addresses
  // cannot be null. Taking the address does not load or call the CUPTI DSO.
  void (*volatile address)();
};

// V2 APIs used by the XLA and JAX Mosaic GPU profilers. This checks the XLA
// stubs, without requiring the runtime CUPTI library to implement these APIs.
const CuptiSymbol kCuptiV2Symbols[] = {
    {"cuptiActivityDisable_v2", &cuptiActivityDisable_v2},
    {"cuptiActivityEnable_v2", &cuptiActivityEnable_v2},
    {"cuptiActivityGetNextRecord_v2", &cuptiActivityGetNextRecord_v2},
    {"cuptiActivityGetNumDroppedRecords_v2",
     &cuptiActivityGetNumDroppedRecords_v2},
    {"cuptiActivityRegisterCallbacks_v2", &cuptiActivityRegisterCallbacks_v2},
    {"cuptiActivitySetAttribute_v2", &cuptiActivitySetAttribute_v2},
    {"cuptiGetTimestamp_v2", &cuptiGetTimestamp_v2},
    {"cuptiSubscribe_v2", &cuptiSubscribe_v2},
};

TEST(CuptiStubTest, LinksProfilerV2Symbols) {
  for (const CuptiSymbol& symbol : kCuptiV2Symbols) {
    EXPECT_NE(symbol.address, nullptr) << symbol.name;
  }
}

}  // namespace
