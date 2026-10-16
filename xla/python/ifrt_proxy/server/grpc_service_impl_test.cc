// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/server/grpc_service_impl.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/support/channel_arguments.h"
#include "grpcpp/support/status.h"
#include "grpcpp/support/sync_stream.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt_proxy/client/grpc_host_buffer.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.grpc.pb.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/versions.h"
#include "xla/python/ifrt_proxy/server/grpc_server.h"
#include "xla/python/ifrt_proxy/server/host_buffer.h"
#include "xla/python/ifrt_proxy/server/ifrt_backend.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

IfrtProxyVersion Version() {
  IfrtProxyVersion version;
  version.set_protocol_version(protocol_version::kServerMax);
  version.set_ifrt_serdes_version_number(
      SerDesVersion::current().version_number().value());
  return version;
}

// Sets up fresh GrpcServer for testing.
absl::StatusOr<std::unique_ptr<GrpcServer>> MakeGrpcServer() {
  // TODO(b/282993619): For external/GKE uses, we may need to find (or build)
  // a utility function that works similar to PickUnusedPortorDie().
  auto addr = absl::StrCat("localhost:", tsl::testing::PickUnusedPortOrDie());
  return GrpcServer::CreateFromIfrtClientFactory(
      addr, [](AttributeMap initialization_data) {
        return absl::UnimplementedError(
            "IFRT client creation fails. This test is not expected to "
            "instantiate any IFRT client");
      });
}

TEST(GrpcServiceImplTest, CanBeUsedToSetupAnGrpcServer) {
  ASSERT_THAT(MakeGrpcServer(), absl_testing::IsOk());
  // Also implicitly tests that destruction of both the server and the
  // implementation objects.
}

// Fails every request, and runs `on_destroy` when its session is torn down.
class FakeBackend final : public BackendInterface {
 public:
  explicit FakeBackend(std::function<void()> on_destroy)
      : on_destroy_(std::move(on_destroy)) {}
  ~FakeBackend() override { on_destroy_(); }

  tsl::Future<Response> Process(std::unique_ptr<IfrtRequest> request) override {
    return tsl::Future<Response>(absl::UnimplementedError("FakeBackend"));
  }

 private:
  std::function<void()> on_destroy_;
};

GrpcServiceImpl::BackendFactory FakeBackendFactory(
    std::function<void()> on_destroy) {
  return [on_destroy = std::move(on_destroy)](
             IfrtProxyVersion version, uint64_t session_id,
             std::shared_ptr<HostBufferStore> host_buffer_store,
             AttributeMap initialization_data)
             -> absl::StatusOr<std::unique_ptr<BackendInterface>> {
    return std::make_unique<FakeBackend>(on_destroy);
  };
}

// Opens an IFRT session on `service` and closes it without sending a request,
// which tears down the session. Returns once the server is done with it.
::grpc::Status OpenAndCloseSession(GrpcServiceImpl& service) {
  ::grpc::ServerBuilder builder;
  builder.RegisterService(&service);
  std::unique_ptr<::grpc::Server> server = builder.BuildAndStart();
  std::unique_ptr<grpc::GrpcIfrtService::Stub> stub =
      grpc::GrpcIfrtService::NewStub(
          server->InProcessChannel(::grpc::ChannelArguments()));

  GrpcIfrtSessionMetadata metadata;
  *metadata.mutable_version() = Version();
  ::grpc::ClientContext context;
  context.AddMetadata("ifrt-proxy-grpc-ifrt-session-metadata-bin",
                      metadata.SerializeAsString());
  std::unique_ptr<::grpc::ClientReaderWriter<IfrtRequest, IfrtResponse>>
      stream = stub->IfrtSession(&context);
  stream->WritesDone();
  return stream->Finish();
}

TEST(GrpcServiceImplTest, DefaultSessionTeardownTimeoutStartsNoWatchdog) {
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(_, _, HasSubstr("must finish within"))).Times(0);
  log.StartCapturingLogs();
  absl::Notification torn_down;
  GrpcServiceImpl service(FakeBackendFactory([&] { torn_down.Notify(); }));

  EXPECT_TRUE(OpenAndCloseSession(service).ok());
  EXPECT_TRUE(torn_down.HasBeenNotified());
}

TEST(GrpcServiceImplTest, SessionTeardownWithinTimeoutFinishes) {
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(_, _,
                       HasSubstr("Teardown of IFRT session 1 must finish "
                                 "within 5s")));
  log.StartCapturingLogs();
  absl::Notification torn_down;
  GrpcServiceImpl::Options options;
  options.session_teardown_timeout = absl::Seconds(5);
  GrpcServiceImpl service(FakeBackendFactory([&] { torn_down.Notify(); }),
                          std::move(options));

  // The session ends only after its watchdog thread exits, which the watchdog
  // does by exiting the process if it is not told that the teardown finished.
  EXPECT_TRUE(OpenAndCloseSession(service).ok());
  EXPECT_TRUE(torn_down.HasBeenNotified());
}

void RunSessionWhoseTeardownOutlastsTimeout() {
  absl::Notification never_notified;
  GrpcServiceImpl::Options options;
  options.session_teardown_timeout = absl::Seconds(1);
  GrpcServiceImpl service(
      FakeBackendFactory([&] {
        // Bounded, so that a process that does not exit
        // fails the test instead of timing it out.
        never_notified.WaitForNotificationWithTimeout(absl::Seconds(30));
      }),
      std::move(options));
  // Not reached: the watchdog exits the process before the session ends.
  ::grpc::Status status = OpenAndCloseSession(service);
  LOG(ERROR) << "Session ended without exiting the process: "
             << status.error_message();
}

TEST(GrpcServiceImplDeathTest, SessionTeardownTimeoutExitsProcess) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(RunSessionWhoseTeardownOutlastsTimeout(),
               "Teardown of IFRT session 1 did not finish within 1s; exiting "
               "so that the server can be restarted");
}

TEST(GrpcServiceImplTest, SessionTeardownStartGuardSpansTeardownOnItsThread) {
  // Each event with the thread it happened on. Only the session's thread
  // writes it, and the session ends before it is read.
  std::vector<std::string> events;
  std::vector<int64_t> threads;
  auto record = [&](std::string event) {
    events.push_back(std::move(event));
    threads.push_back(tsl::Env::Default()->GetCurrentThreadId());
  };
  GrpcServiceImpl::Options options;
  options.on_session_teardown_start =
      [&](uint64_t session_id) -> std::shared_ptr<void> {
    record(absl::StrCat("start ", session_id));
    return std::shared_ptr<void>(nullptr,
                                 [&](void*) { record("guard destroyed"); });
  };
  GrpcServiceImpl service(
      FakeBackendFactory([&] { record("backend destroyed"); }),
      std::move(options));

  ASSERT_TRUE(OpenAndCloseSession(service).ok());

  EXPECT_THAT(events,
              ElementsAre("start 1", "backend destroyed", "guard destroyed"));
  ASSERT_EQ(threads.size(), 3);
  EXPECT_EQ(threads[0], threads[1]);
  EXPECT_EQ(threads[1], threads[2]);
}

class GrpcIfrtServiceImplHostBufferTest
    : public testing::TestWithParam</*size=*/int64_t> {
 protected:
  GrpcIfrtServiceImplHostBufferTest()
      : impl_([](IfrtProxyVersion version, uint64_t session_id,
                 std::shared_ptr<HostBufferStore> host_buffer_store,
                 AttributeMap initialization_data) {
          return absl::UnimplementedError(
              "IFRT backend creation is not implemented");
        }) {
    ::grpc::ServerBuilder builder;
    builder.RegisterService(&impl_);
    server_ = builder.BuildAndStart();

    stub_ = grpc::GrpcIfrtService::NewStub(
        server_->InProcessChannel(::grpc::ChannelArguments()));
  }

  // Returns a string to be stored as a host buffer. The length is parameterized
  // so that we can test chunking.
  std::string GetTestData() const {
    std::string data;
    for (int i = 0; i < GetParam(); ++i) {
      data.push_back(i % 7);
    }
    return data;
  }

  GrpcServiceImpl impl_;
  std::unique_ptr<::grpc::Server> server_;
  std::shared_ptr<grpc::GrpcIfrtService::Stub> stub_;
};

TEST_P(GrpcIfrtServiceImplHostBufferTest, StoreAndLookupStringView) {
  static constexpr uint64_t kSessionId = 1;

  auto store = std::make_shared<HostBufferStore>();
  ASSERT_TRUE(impl_.Test_InsertHostBufferStore(kSessionId, store));
  GrpcClientHostBufferStore client(stub_, Version(), kSessionId);

  constexpr uint64_t kHandle = 2;
  const std::string data = GetTestData();
  absl::string_view source(data);

  ASSERT_THAT(client.Store(kHandle, source).Await(), absl_testing::IsOk());
  EXPECT_THAT(client.Lookup(kHandle).Await(), absl_testing::IsOkAndHolds(data));

  EXPECT_TRUE(impl_.Test_DeleteHostBufferStore(kSessionId));
}

TEST_P(GrpcIfrtServiceImplHostBufferTest, StoreAndLookupCord) {
  static constexpr uint64_t kSessionId = 1;

  auto store = std::make_shared<HostBufferStore>();
  ASSERT_TRUE(impl_.Test_InsertHostBufferStore(kSessionId, store));
  GrpcClientHostBufferStore client(stub_, Version(), kSessionId);

  constexpr uint64_t kHandle = 2;
  const std::string data = GetTestData();

  absl::Cord source(data);
  ASSERT_THAT(client.Store(kHandle, source).Await(), absl_testing::IsOk());
  EXPECT_THAT(client.Lookup(kHandle).Await(), absl_testing::IsOkAndHolds(data));

  EXPECT_TRUE(impl_.Test_DeleteHostBufferStore(kSessionId));
}

TEST_P(GrpcIfrtServiceImplHostBufferTest, Lookup) {
  static constexpr uint64_t kSessionId = 1;

  auto store = std::make_shared<HostBufferStore>();
  ASSERT_TRUE(impl_.Test_InsertHostBufferStore(kSessionId, store));
  GrpcClientHostBufferStore client(stub_, Version(), kSessionId);

  constexpr uint64_t kHandle = 2;
  const std::string data = GetTestData();
  ASSERT_THAT(store->Store(kHandle, data), absl_testing::IsOk());

  EXPECT_THAT(client.Lookup(kHandle).Await(), absl_testing::IsOkAndHolds(data));

  EXPECT_TRUE(impl_.Test_DeleteHostBufferStore(kSessionId));
}

TEST_P(GrpcIfrtServiceImplHostBufferTest, Delete) {
  static constexpr uint64_t kSessionId = 1;

  auto store = std::make_shared<HostBufferStore>();
  ASSERT_TRUE(impl_.Test_InsertHostBufferStore(kSessionId, store));
  GrpcClientHostBufferStore client(stub_, Version(), kSessionId);

  constexpr uint64_t kHandle = 2;
  const std::string data = GetTestData();
  ASSERT_THAT(store->Store(kHandle, data), absl_testing::IsOk());

  ASSERT_THAT(client.Delete(kHandle).Await(), absl_testing::IsOk());
  EXPECT_THAT(client.Lookup(kHandle).Await(),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));

  EXPECT_TRUE(impl_.Test_DeleteHostBufferStore(kSessionId));
}

INSTANTIATE_TEST_SUITE_P(
    DataSize, GrpcIfrtServiceImplHostBufferTest,
    testing::Values(0,                  // Empty host buffer.
                    16,                 // Small enough to fit in one chunk.
                    3 * 1024 * 1024));  // Requires multiple chunks

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
