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

#include "xla/service/llvm_ir/error_handler.h"

#include <atomic>
#include <cstdint>

#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/barrier.h"
#include "llvm/Support/ErrorHandling.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace {

TEST(XlaScopedFatalErrorHandlerTest, MultiThreadedFatalError) {
  EXPECT_DEATH(
      {
        static std::atomic<int32_t> last_thread_id{-1};
        constexpr int32_t kNumThreads = 10;
        tsl::thread::ThreadPool pool(tsl::Env::Default(), "test", kNumThreads);
        absl::Barrier barrier(kNumThreads);

        for (int i = 0; i < kNumThreads; ++i) {
          pool.Schedule([i, &barrier]() {
            absl::AnyInvocable<void(absl::string_view)> handler =
                [i](absl::string_view reason) {
                  LOG(ERROR) << "Handler called for thread " << i
                             << " with reason " << reason;
                  last_thread_id = i;
                };
            XlaScopedFatalErrorHandler guard(&handler);
            barrier.Block();
            if (i == 4) {
              llvm::report_fatal_error("test error");
            }
          });
        }
        // Wait for threads to finish (though thread 4 should kill the process).
      },
      "Handler called for thread 4 with reason test error");
}

}  // namespace
}  // namespace xla
