/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/backends/profiler/gpu/cupti_error_manager.h"

#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_callbacks.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_driver_cbid.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_result.h"
#include "xla/backends/profiler/gpu/cuda_test.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#include "xla/backends/profiler/gpu/cupti_wrapper.h"
#include "xla/backends/profiler/gpu/mock_cupti.h"
#include "xla/tsl/profiler/utils/time_utils.h"

namespace xla {
namespace profiler {
namespace test {

using xla::profiler::CuptiInterface;
using xla::profiler::CuptiTracer;
using xla::profiler::CuptiTracerCollectorOptions;
using xla::profiler::CuptiTracerOptions;
using xla::profiler::CuptiWrapper;

using ::testing::_;
using ::testing::DoAll;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::Sequence;
using ::testing::SetArgPointee;
using ::testing::StrictMock;

auto SetTimestampAndReturnSuccess(uint64_t timestamp) {
  return DoAll(SetArgPointee<1>(timestamp), Return(CUPTI_SUCCESS));
}

// Needed to create different cupti tracer for each test cases.
class TestableCuptiTracer : public CuptiTracer {
 public:
  explicit TestableCuptiTracer(CuptiInterface* cupti_interface)
      : CuptiTracer(cupti_interface) {}
};

// CuptiErrorManagerTest verifies that an application is not killed due to an
// unexpected error in the underlying GPU hardware during tracing.
// MockCupti is used to simulate a CUPTI call failure.
class CuptiErrorManagerTest : public ::testing::Test {
 protected:
  CuptiErrorManagerTest() {}

  void SetUp() override {
    ASSERT_GT(CuptiTracer::NumGpus(), 0) << "No devices found";
    auto mock_cupti = std::make_unique<StrictMock<MockCupti>>();
    mock_ = mock_cupti.get();
    cupti_error_manager_ =
        std::make_unique<CuptiErrorManager>(std::move(mock_cupti));

    cupti_tracer_ =
        std::make_unique<TestableCuptiTracer>(cupti_error_manager_.get());
    cupti_wrapper_ = std::make_unique<CuptiWrapper>();

    CuptiTracerCollectorOptions collector_options;
    collector_options.num_gpus = CuptiTracer::NumGpus();
    uint64_t start_gputime_ns = CuptiTracer::GetTimestamp();
    uint64_t start_walltime_ns = tsl::profiler::GetCurrentTimeNanos();
    cupti_collector_ = CreateCuptiCollector(
        collector_options, start_walltime_ns, start_gputime_ns);
  }

  void EnableProfiling(const CuptiTracerOptions& option) {
    cupti_tracer_->Enable(option, cupti_collector_.get()).IgnoreError();
  }

  void DisableProfiling() { cupti_tracer_->Disable(); }

  bool CuptiDisabled() const { return cupti_error_manager_->Disabled(); }

  CuptiTracerOptions KernelTraceOptions() {
    CuptiTracerOptions options;
    options.activities_selected = {CUPTI_ACTIVITY_KIND_KERNEL};
    options.cbids_selected = {CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel};
    return options;
  }

  void ExpectSuccessfulV1KernelTrace(CUpti_SubscriberHandle subscriber) {
    const int resource_cb_count = IsCudaNewEnoughForGraphTraceTest() ? 5 : 0;
    EXPECT_CALL(*mock_,
                EnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, _))
        .Times(resource_cb_count)
        .WillRepeatedly(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_,
                EnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                               CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel))
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, SetThreadIdType(CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM))
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, ActivityUsePerThreadBuffer())
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, ActivityRegisterCallbacks(_, _))
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, ActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL))
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_,
                EnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RESOURCE, _))
        .Times(resource_cb_count)
        .WillRepeatedly(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_,
                EnableCallback(0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                               CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel))
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, Unsubscribe(subscriber))
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, ActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL))
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, ActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED))
        .WillOnce(Return(CUPTI_SUCCESS));
  }

  void ExpectV2ResourceCallbacks(Sequence& sequence,
                                 CUpti_SubscriberHandle subscriber,
                                 uint32_t enable) {
    const int count = IsCudaNewEnoughForGraphTraceTest() ? 5 : 0;
    if (count == 0) {
      return;
    }
    EXPECT_CALL(*mock_,
                EnableCallback(enable, subscriber, CUPTI_CB_DOMAIN_RESOURCE, _))
        .Times(count)
        .InSequence(sequence)
        .WillRepeatedly(Return(CUPTI_SUCCESS));
  }

  void ExpectV2KernelCallback(Sequence& sequence,
                              CUpti_SubscriberHandle subscriber,
                              uint32_t enable,
                              CUptiResult result = CUPTI_SUCCESS) {
    EXPECT_CALL(*mock_,
                EnableCallback(enable, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                               CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel))
        .InSequence(sequence)
        .WillOnce(Return(result));
  }

  void ExpectV2KernelSession(Sequence& sequence,
                             CUpti_SubscriberHandle subscriber,
                             uint64_t preflight_timestamp,
                             uint64_t fallback_stop_timestamp,
                             uint64_t stop_timestamp,
                             CUptiResult stop_timestamp_status = CUPTI_SUCCESS,
                             bool expect_preflight_timestamp = true) {
    if (expect_preflight_timestamp) {
      EXPECT_CALL(*mock_, GetTimestampV2(subscriber, _))
          .InSequence(sequence)
          .WillOnce(SetTimestampAndReturnSuccess(preflight_timestamp));
    }
    ExpectV2ResourceCallbacks(sequence, subscriber, /*enable=*/1);
    ExpectV2KernelCallback(sequence, subscriber, /*enable=*/1);
    EXPECT_CALL(*mock_, ActivityUseSystemThreadIdV2(subscriber))
        .InSequence(sequence)
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, ActivityUsePerThreadBufferV2())
        .InSequence(sequence)
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, ActivityRegisterCallbacksV2(subscriber, _, _))
        .InSequence(sequence)
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_,
                ActivityEnableV2(subscriber, CUPTI_ACTIVITY_KIND_KERNEL, _))
        .InSequence(sequence)
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, GetTimestampV2(subscriber, _))
        .InSequence(sequence)
        .WillOnce(SetTimestampAndReturnSuccess(fallback_stop_timestamp));
    ExpectV2ResourceCallbacks(sequence, subscriber, /*enable=*/0);
    ExpectV2KernelCallback(sequence, subscriber, /*enable=*/0);
    EXPECT_CALL(*mock_,
                ActivityDisableV2(subscriber, CUPTI_ACTIVITY_KIND_KERNEL, _))
        .InSequence(sequence)
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, ActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED))
        .InSequence(sequence)
        .WillOnce(Return(CUPTI_SUCCESS));
    if (stop_timestamp_status == CUPTI_SUCCESS) {
      EXPECT_CALL(*mock_, GetTimestampV2(subscriber, _))
          .InSequence(sequence)
          .WillOnce(SetTimestampAndReturnSuccess(stop_timestamp));
    } else {
      EXPECT_CALL(*mock_, GetTimestampV2(subscriber, _))
          .InSequence(sequence)
          .WillOnce(Return(stop_timestamp_status));
    }
  }

  void ExpectV2KernelSessionWithFatalStopTimestamp(
      Sequence& sequence, CUpti_SubscriberHandle subscriber,
      uint64_t preflight_timestamp, uint64_t fallback_stop_timestamp) {
    ExpectV2KernelSession(sequence, subscriber, preflight_timestamp,
                          fallback_stop_timestamp,
                          /*stop_timestamp=*/0, CUPTI_ERROR_INVALID_PARAMETER);
    EXPECT_CALL(*mock_, GetResultString(CUPTI_ERROR_INVALID_PARAMETER, _))
        .InSequence(sequence)
        .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::GetResultString));

    EXPECT_CALL(*mock_,
                ActivityDisableV2(subscriber, CUPTI_ACTIVITY_KIND_KERNEL, _))
        .InSequence(sequence)
        .WillOnce(Return(CUPTI_SUCCESS));
    ExpectV2KernelCallback(sequence, subscriber, /*enable=*/0);
    ExpectV2ResourceCallbacks(sequence, subscriber, /*enable=*/0);
    EXPECT_CALL(*mock_, Unsubscribe(subscriber))
        .InSequence(sequence)
        .WillOnce(Return(CUPTI_SUCCESS));
  }

  void RunGpuApp() {
    MemCopyH2D();
    PrintfKernel(/*iters=*/10);
    Synchronize();
    MemCopyD2H();
  }

  // Pointer to MockCupti passed to CuptiBase constructor.
  // Used to inject failures to be handled by CuptiErrorManager.
  // Wrapped in StrictMock so unexpected calls cause a test failure.
  StrictMock<MockCupti>* mock_;

  // CuptiTracer instance that uses MockCupti instead of CuptiWrapper.
  std::unique_ptr<TestableCuptiTracer> cupti_tracer_ = nullptr;

  std::unique_ptr<CuptiInterface> cupti_error_manager_;

  // CuptiWrapper instance to which mock_ calls are delegated.
  std::unique_ptr<CuptiWrapper> cupti_wrapper_;

  std::unique_ptr<xla::profiler::CuptiTraceCollector> cupti_collector_;
};

// Verifies that failed EnableProfiling() does not kill an application.
TEST_F(CuptiErrorManagerTest, GpuTraceActivityEnableTest) {
  // Enforces the order of execution below.
  Sequence s1;
  // CuptiBase::EnableProfiling()
  EXPECT_CALL(*mock_, SubscribeV2(_, _, _))
      .InSequence(s1)
      .WillOnce([](CUpti_SubscriberHandle* subscriber,
                   CUpti_CallbackFunc /*callback*/, void* /*userdata*/) {
        *subscriber = reinterpret_cast<CUpti_SubscriberHandle>(uintptr_t{1});
        return CUPTI_SUCCESS;
      });
  EXPECT_CALL(*mock_, GetTimestampV2(_, _))
      .InSequence(s1)
      .WillOnce(SetTimestampAndReturnSuccess(1));
  const int cb_enable_times = IsCudaNewEnoughForGraphTraceTest() ? 6 : 1;
  EXPECT_CALL(*mock_, EnableCallback(1, _, _, _))
      .Times(cb_enable_times)
      .InSequence(s1)
      .WillRepeatedly(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityUseSystemThreadIdV2(_))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityUsePerThreadBufferV2())
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityRegisterCallbacksV2(_, _, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityEnableV2(_, CUPTI_ACTIVITY_KIND_KERNEL, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_ERROR_UNKNOWN));  // injected error
  // CuptiErrorManager::ResultString()
  EXPECT_CALL(*mock_, GetResultString(CUPTI_ERROR_UNKNOWN, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::GetResultString));
  // CuptiErrorManager::UndoAndDisable()
  EXPECT_CALL(*mock_, EnableCallback(0, _, _, _))
      .Times(cb_enable_times)
      .InSequence(s1)
      .WillRepeatedly(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, Unsubscribe(_))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));

  EXPECT_FALSE(CuptiDisabled());
  CuptiTracerOptions options;
  options.activities_selected.push_back(CUPTI_ACTIVITY_KIND_KERNEL);
  options.cbids_selected.push_back(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
  EnableProfiling(options);  // CUPTI call fails due to injected error
  EXPECT_TRUE(CuptiDisabled());
  // Rollback already unsubscribed the handle and disabled CUPTI. Verify that a
  // later profiling request makes no additional CUPTI calls, including a second
  // unsubscribe.
  EnableProfiling(options);

  RunGpuApp();  // Application code runs normally

  EXPECT_TRUE(CuptiDisabled());
  DisableProfiling();  // CUPTI calls are ignored
  EXPECT_TRUE(CuptiDisabled());
}

// Verifies that failed EnableProfiling() does not kill an application.
TEST_F(CuptiErrorManagerTest, GpuTraceAutoEnableTest) {
  EXPECT_FALSE(CuptiDisabled());
  // Enforces the order of execution below.
  Sequence s1;
  EXPECT_CALL(*mock_, SubscribeV2(_, _, _))
      .InSequence(s1)
      .WillOnce([](CUpti_SubscriberHandle* subscriber,
                   CUpti_CallbackFunc /*callback*/, void* /*userdata*/) {
        *subscriber = reinterpret_cast<CUpti_SubscriberHandle>(uintptr_t{1});
        return CUPTI_SUCCESS;
      });
  EXPECT_CALL(*mock_, GetTimestampV2(_, _))
      .InSequence(s1)
      .WillOnce(SetTimestampAndReturnSuccess(1));
  const int cb_enable_times = IsCudaNewEnoughForGraphTraceTest() ? 5 : 0;
  if (cb_enable_times > 0) {
    EXPECT_CALL(*mock_, EnableCallback(1, _, _, _))
        .Times(cb_enable_times)
        .InSequence(s1)
        .WillRepeatedly(Return(CUPTI_SUCCESS));
  }
  EXPECT_CALL(*mock_, EnableDomain(1, _, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityUseSystemThreadIdV2(_))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityUsePerThreadBufferV2())
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityRegisterCallbacksV2(_, _, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityEnableV2(_, CUPTI_ACTIVITY_KIND_MEMCPY, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityEnableV2(_, CUPTI_ACTIVITY_KIND_MEMCPY2, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_ERROR_UNKNOWN));  // injected error
  // CuptiErrorManager::ResultString()
  EXPECT_CALL(*mock_, GetResultString(CUPTI_ERROR_UNKNOWN, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::GetResultString));
  // CuptiErrorManager::UndoAndDisable()
  EXPECT_CALL(*mock_, ActivityDisableV2(_, CUPTI_ACTIVITY_KIND_MEMCPY, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, EnableDomain(0, _, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  if (cb_enable_times > 0) {
    EXPECT_CALL(*mock_, EnableCallback(0, _, _, _))
        .Times(cb_enable_times)
        .InSequence(s1)
        .WillRepeatedly(Return(CUPTI_SUCCESS));
  }
  EXPECT_CALL(*mock_, Unsubscribe(_))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));

  EXPECT_FALSE(CuptiDisabled());
  CuptiTracerOptions options;
  options.activities_selected.push_back(CUPTI_ACTIVITY_KIND_MEMCPY);
  options.activities_selected.push_back(CUPTI_ACTIVITY_KIND_MEMCPY2);
  options.activities_selected.push_back(CUPTI_ACTIVITY_KIND_KERNEL);
  // options.cbids_selected.push_back(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
  EnableProfiling(options);  // CUPTI call fails due to injected error
  EXPECT_TRUE(CuptiDisabled());

  RunGpuApp();  // Application code runs normally

  EXPECT_TRUE(CuptiDisabled());
  DisableProfiling();  // CUPTI calls are ignored
  EXPECT_TRUE(CuptiDisabled());
}

TEST_F(CuptiErrorManagerTest, V2AvailabilityFailuresFallBackToV1) {
  EXPECT_FALSE(CuptiDisabled());

  // CuptiWrapper returns CUPTI_ERROR_NOT_SUPPORTED without calling
  // cuptiSubscribe_v2 when either cuptiSubscribe_v2 or cuptiGetTimestamp_v2 is
  // unavailable. CUPTI_ERROR_UNKNOWN from cuptiSubscribe_v2 is also safe to
  // fall back from because no V2 subscriber was successfully created.
  uintptr_t subscriber_id = 1;
  for (CUptiResult result : {CUPTI_ERROR_NOT_SUPPORTED, CUPTI_ERROR_UNKNOWN}) {
    auto* const v1_subscriber =
        reinterpret_cast<CUpti_SubscriberHandle>(subscriber_id++);
    EXPECT_CALL(*mock_, SubscribeV2(_, _, _)).WillOnce(Return(result));
    EXPECT_CALL(*mock_, Subscribe(_, _, _))
        .WillOnce(
            DoAll(SetArgPointee<0>(v1_subscriber), Return(CUPTI_SUCCESS)));
    ExpectSuccessfulV1KernelTrace(v1_subscriber);

    CuptiTracerOptions options = KernelTraceOptions();
    EnableProfiling(options);
    DisableProfiling();
  }

  EXPECT_FALSE(CuptiDisabled());
}

TEST_F(CuptiErrorManagerTest, V2TimestampFailuresAfterSubscribeFallBackToV1) {
  EXPECT_FALSE(CuptiDisabled());

  uintptr_t subscriber_id = 10;
  for (CUptiResult result : {CUPTI_ERROR_NOT_SUPPORTED,
                             CUPTI_ERROR_NOT_COMPATIBLE, CUPTI_ERROR_UNKNOWN}) {
    auto* const v2_subscriber =
        reinterpret_cast<CUpti_SubscriberHandle>(subscriber_id++);
    auto* const v1_subscriber =
        reinterpret_cast<CUpti_SubscriberHandle>(subscriber_id++);
    Sequence fallback_sequence;
    EXPECT_CALL(*mock_, SubscribeV2(_, _, _))
        .InSequence(fallback_sequence)
        .WillOnce(
            DoAll(SetArgPointee<0>(v2_subscriber), Return(CUPTI_SUCCESS)));
    EXPECT_CALL(*mock_, GetTimestampV2(v2_subscriber, _))
        .InSequence(fallback_sequence)
        .WillOnce(Return(result));
    EXPECT_CALL(*mock_, Unsubscribe(v2_subscriber))
        .InSequence(fallback_sequence)
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, Subscribe(_, _, _))
        .InSequence(fallback_sequence)
        .WillOnce(
            DoAll(SetArgPointee<0>(v1_subscriber), Return(CUPTI_SUCCESS)));
    ExpectSuccessfulV1KernelTrace(v1_subscriber);

    CuptiTracerOptions options = KernelTraceOptions();
    EnableProfiling(options);
    DisableProfiling();
  }

  EXPECT_FALSE(CuptiDisabled());
}

TEST_F(CuptiErrorManagerTest,
       FatalTimestampV2FailureDoesNotUnsubscribeFreshSubscriberTwice) {
  EXPECT_FALSE(CuptiDisabled());

  auto* const v2_subscriber =
      reinterpret_cast<CUpti_SubscriberHandle>(uintptr_t{1});
  EXPECT_CALL(*mock_, SubscribeV2(_, _, _))
      .WillOnce(DoAll(SetArgPointee<0>(v2_subscriber), Return(CUPTI_SUCCESS)));
  EXPECT_CALL(*mock_, GetTimestampV2(v2_subscriber, _))
      .WillOnce(Return(CUPTI_ERROR_INVALID_PARAMETER));
  EXPECT_CALL(*mock_, GetResultString(CUPTI_ERROR_INVALID_PARAMETER, _))
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::GetResultString));
  // The error-manager undo stack owns this newly created subscriber and must
  // be the only code path that unsubscribes it.
  EXPECT_CALL(*mock_, Unsubscribe(v2_subscriber))
      .WillOnce(Return(CUPTI_SUCCESS));

  CuptiTracerOptions options;
  EnableProfiling(options);

  EXPECT_TRUE(CuptiDisabled());
}

TEST_F(CuptiErrorManagerTest,
       PreparesFreshV2SubscriberBeforeEachSessionTimestamp) {
  EXPECT_FALSE(CuptiDisabled());

  Sequence sequence;
  CuptiTracerOptions options = KernelTraceOptions();

  for (uintptr_t session = 1; session <= 2; ++session) {
    auto* const subscriber = reinterpret_cast<CUpti_SubscriberHandle>(session);
    EXPECT_CALL(*mock_, SubscribeV2(_, _, _))
        .InSequence(sequence)
        .WillOnce(DoAll(SetArgPointee<0>(subscriber), Return(CUPTI_SUCCESS)));
    EXPECT_CALL(*mock_, GetTimestampV2(subscriber, _))
        .InSequence(sequence)
        .WillOnce(SetTimestampAndReturnSuccess(session * 10));

    absl::Status prepare_status =
        cupti_tracer_->PrepareForProfilerStart(options);
    ASSERT_TRUE(prepare_status.ok()) << prepare_status;

    EXPECT_CALL(*mock_, GetTimestampV2(subscriber, _))
        .InSequence(sequence)
        .WillOnce(SetTimestampAndReturnSuccess(session * 10 + 1));
    EXPECT_EQ(cupti_tracer_->GetTimestampForSubscriber(), session * 10 + 1);

    ExpectV2KernelSession(sequence, subscriber, /*preflight_timestamp=*/0,
                          /*fallback_stop_timestamp=*/session * 10 + 2,
                          /*stop_timestamp=*/session * 10 + 3, CUPTI_SUCCESS,
                          /*expect_preflight_timestamp=*/false);
    EXPECT_CALL(*mock_, Unsubscribe(subscriber))
        .InSequence(sequence)
        .WillOnce(Return(CUPTI_SUCCESS));

    EnableProfiling(options);
    DisableProfiling();
    EXPECT_EQ(cupti_collector_->GetTracingEndTimeNs(), session * 10 + 3);
  }

  EXPECT_FALSE(CuptiDisabled());
}

TEST_F(CuptiErrorManagerTest,
       FatalStopTimestampDoesNotUnsubscribeFreshSubscriberTwice) {
  Sequence s1;
  auto* const subscriber =
      reinterpret_cast<CUpti_SubscriberHandle>(uintptr_t{1});
  EXPECT_CALL(*mock_, SubscribeV2(_, _, _))
      .InSequence(s1)
      .WillOnce(DoAll(SetArgPointee<0>(subscriber), Return(CUPTI_SUCCESS)));
  ExpectV2KernelSessionWithFatalStopTimestamp(s1, subscriber,
                                              /*preflight_timestamp=*/1,
                                              /*fallback_stop_timestamp=*/2);

  CuptiTracerOptions options = KernelTraceOptions();
  EnableProfiling(options);
  DisableProfiling();
  EXPECT_EQ(cupti_collector_->GetTracingEndTimeNs(), 2);
  EXPECT_TRUE(CuptiDisabled());
  // A stale handle would make this retry issue an unexpected second
  // Unsubscribe call through the disabled error manager.
  EnableProfiling(options);
}

TEST_F(CuptiErrorManagerTest,
       V2CapabilityErrorsDoNotDisableCuptiOrUseV1Parser) {
  auto* const v2_subscriber =
      reinterpret_cast<CUpti_SubscriberHandle>(uintptr_t{1});
  EXPECT_CALL(*mock_, ActivityUseSystemThreadIdV2(v2_subscriber))
      .WillOnce(Return(CUPTI_ERROR_NOT_COMPATIBLE));
  EXPECT_CALL(*mock_, ActivityUsePerThreadBufferV2())
      .WillOnce(Return(CUPTI_ERROR_NOT_COMPATIBLE));

  EXPECT_EQ(cupti_error_manager_->ActivityUseSystemThreadIdV2(v2_subscriber),
            CUPTI_ERROR_NOT_COMPATIBLE);
  EXPECT_EQ(cupti_error_manager_->ActivityUsePerThreadBufferV2(),
            CUPTI_ERROR_NOT_COMPATIBLE);

  uint8_t buffer[1] = {};
  CUpti_Activity* record = nullptr;
  EXPECT_CALL(*mock_, ActivityGetNextRecord(_, _, _)).Times(0);

  for (CUptiResult result : {CUPTI_ERROR_NOT_SUPPORTED, CUPTI_ERROR_UNKNOWN}) {
    EXPECT_CALL(*mock_, ActivityGetNextRecordV2(_, buffer, sizeof(buffer), _))
        .WillOnce(Return(result));
    EXPECT_EQ(cupti_error_manager_->ActivityGetNextRecordV2(
                  /*subscriber=*/nullptr, buffer, sizeof(buffer), &record),
              result);
    EXPECT_EQ(record, nullptr);
  }
  EXPECT_FALSE(CuptiDisabled());
}

}  // namespace test
}  // namespace profiler
}  // namespace xla
