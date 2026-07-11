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

#include "xla/service/gpu/execution_watchdog.h"

#include <atomic>
#include <optional>

#include <gtest/gtest.h>
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

TEST(ExecutionWatchdogScopeTest, CreateReturnsNulloptWhenTimeoutDisabled) {
  DebugOptions debug_options;
  ASSERT_OK_AND_ASSIGN(
      std::optional<ExecutionWatchdogScope> scope,
      ExecutionWatchdogScope::Create(&debug_options, "test_module",
                                     /*device_ordinal=*/0,
                                     /*gpu_run_options=*/nullptr,
                                     /*stream=*/nullptr,
                                     /*block_host_until_done=*/true));
  EXPECT_FALSE(scope.has_value());
}

TEST(ExecutionWatchdogScopeTest, CreateReturnsScopeWhenTimeoutEnabled) {
  DebugOptions debug_options;
  debug_options.set_xla_gpu_execution_terminate_timeout("30s");
  ASSERT_OK_AND_ASSIGN(
      std::optional<ExecutionWatchdogScope> scope,
      ExecutionWatchdogScope::Create(&debug_options, "test_module",
                                     /*device_ordinal=*/0,
                                     /*gpu_run_options=*/nullptr,
                                     /*stream=*/nullptr,
                                     /*block_host_until_done=*/false));
  ASSERT_TRUE(scope.has_value());
  EXPECT_FALSE(scope->IsArmed());
}

TEST(ExecutionWatchdogScopeTest, TimeoutHandlerFiresWhileScopeIsAlive) {
  DebugOptions debug_options;
  debug_options.set_xla_gpu_execution_terminate_timeout("50ms");

  std::atomic<bool> handler_called{false};
  absl::Notification handler_done;
  GpuExecutableRunOptions gpu_run_options;
  gpu_run_options.set_execution_timeout_handler(
      [&](absl::string_view /*action*/, absl::Duration /*timeout*/) {
        handler_called.store(true);
        handler_done.Notify();
      });

  ASSERT_OK_AND_ASSIGN(
      std::optional<ExecutionWatchdogScope> scope,
      ExecutionWatchdogScope::Create(&debug_options, "test_module",
                                     /*device_ordinal=*/0, &gpu_run_options,
                                     /*stream=*/nullptr,
                                     /*block_host_until_done=*/true));
  ASSERT_TRUE(scope.has_value());
  scope->Arm();

  EXPECT_TRUE(handler_done.WaitForNotificationWithTimeout(absl::Seconds(5)));
  EXPECT_TRUE(handler_called.load());

  // Scope destruction waits until after the timeout path above; the handler
  // must have been invoked before the scope is released.
}

TEST(ExecutionWatchdogScopeTest, ArmIsIdempotent) {
  DebugOptions debug_options;
  debug_options.set_xla_gpu_execution_terminate_timeout("30s");

  ASSERT_OK_AND_ASSIGN(
      std::optional<ExecutionWatchdogScope> scope,
      ExecutionWatchdogScope::Create(&debug_options, "test_module",
                                     /*device_ordinal=*/0,
                                     /*gpu_run_options=*/nullptr,
                                     /*stream=*/nullptr,
                                     /*block_host_until_done=*/true));
  ASSERT_TRUE(scope.has_value());
  scope->Arm();
  EXPECT_TRUE(scope->IsArmed());
  scope->Arm();
  EXPECT_TRUE(scope->IsArmed());
}

}  // namespace
}  // namespace xla::gpu
