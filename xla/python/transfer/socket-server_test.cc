/* Copyright 2025 The OpenXLA Authors.

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
#include "xla/python/transfer/socket-server.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/python/transfer/event_loop.h"
#include "xla/python/transfer/streaming.h"
#include "xla/python/transfer/transfer_socket.pb.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/test.h"

namespace aux {
namespace {

TEST(ServerTest, Basic) {
  sockaddr_in6 addr;
  memset(&addr, 0, sizeof(sockaddr_in6));
  addr.sin6_family = AF_INET6;
  auto local_factory = BulkTransportFactory::CreateLocal();
  auto servera = std::make_shared<SocketServer>();
  CHECK_OK(servera->Start(SocketAddress(addr), local_factory));
  auto serverb = std::make_shared<SocketServer>();
  CHECK_OK(serverb->Start(SocketAddress(addr), local_factory));

  std::string msg("secret message");
  uint64_t uuid = 5678;
  int buffer_id = 0;
  serverb->AwaitPull(uuid, PullTable::MakeStringEntry({msg}));

  auto [s, cd] = ChunkDestination::MakeStringDest();
  auto conn = servera->Connect(serverb->addr());
  conn->Pull(uuid, buffer_id, std::move(cd));

  CHECK_EQ(s.Await().value(), msg);
  absl::SleepFor(absl::Seconds(2));
  conn = {};
  serverb->WaitForQuiesce();
}

TEST(ServerTest, DelayedConnect) {
  auto addra = SocketAddress::Parse("0.0.0.0:0").value();
  int port = tsl::testing::PickUnusedPortOrDie();
  auto addrb = SocketAddress::Parse(absl::StrCat("0.0.0.0:", port)).value();
  auto local_factory = BulkTransportFactory::CreateLocal();
  auto servera = std::make_shared<SocketServer>();
  CHECK_OK(servera->Start(addra, local_factory));

  uint64_t uuid = 5678;
  int buffer_id = 0;

  auto [s, cd] = ChunkDestination::MakeStringDest();
  auto conn = servera->Connect(addrb);
  conn->Pull(uuid, buffer_id, std::move(cd));

  auto serverb = std::make_shared<SocketServer>();
  CHECK_OK(serverb->Start(addrb, local_factory));
  std::string msg("secret message");
  serverb->AwaitPull(uuid, PullTable::MakeStringEntry({msg}));

  CHECK_EQ(s.Await().value(), msg);
  absl::SleepFor(absl::Seconds(2));
  conn = {};
  serverb->WaitForQuiesce();
}

absl::Status SendFramedRequestAndExpectPeerClose(
    const SocketAddress& server_addr, const SocketTransferRequest& req) {
  int fd = socket(server_addr.address().sa_family, SOCK_STREAM | SOCK_CLOEXEC,
                  /*protocol=*/0);
  if (fd < 0) {
    return absl::InternalError("socket() failed");
  }
  auto close_fd = absl::MakeCleanup([fd]() { close(fd); });
  if (connect(fd, reinterpret_cast<const struct sockaddr*>(&server_addr),
              sizeof(server_addr)) != 0) {
    return absl::InternalError("connect() failed");
  }
  std::string payload = req.SerializeAsString();
  uint32_t header = static_cast<uint32_t>(payload.size());
  std::string frame(reinterpret_cast<const char*>(&header), sizeof(header));
  frame += payload;
  if (send(fd, frame.data(), frame.size(), MSG_NOSIGNAL) !=
      static_cast<ssize_t>(frame.size())) {
    return absl::InternalError("send() failed");
  }
  char buf[16];
  ssize_t n = recv(fd, buf, sizeof(buf), 0);
  if (n > 0) {
    return absl::InternalError("expected peer to close connection");
  }
  return absl::OkStatus();
}

TEST(ServerTest, RejectsUnestablishedAndMalformedFramesWithoutCrashing) {
  sockaddr_in6 addr;
  memset(&addr, 0, sizeof(sockaddr_in6));
  addr.sin6_family = AF_INET6;
  auto local_factory = BulkTransportFactory::CreateLocal();
  auto servera = std::make_shared<SocketServer>();
  ASSERT_OK(servera->Start(SocketAddress(addr), local_factory));
  auto serverb = std::make_shared<SocketServer>();
  ASSERT_OK(serverb->Start(SocketAddress(addr), local_factory));

  constexpr uint64_t kUuid = 0;
  std::string msg("protected payload");
  serverb->AwaitPull(kUuid, PullTable::MakeStringEntry({msg}));

  // 1. Empty SocketTransferRequest (MSG_NOT_SET) must poison the connection
  // rather than hitting LOG(FATAL).
  SocketTransferRequest empty_req;
  EXPECT_OK(SendFramedRequestAndExpectPeerClose(serverb->addr(), empty_req));

  // 2. Pull request sent before SocketTransferEstablishBulkTransport must be
  // rejected without touching PullTable or null dereferencing bulk_transport_.
  SocketTransferRequest premature_pull;
  premature_pull.mutable_pull()->set_uuid(kUuid);
  premature_pull.mutable_pull()->add_buffer_ids(0);
  EXPECT_OK(
      SendFramedRequestAndExpectPeerClose(serverb->addr(), premature_pull));

  // 3. Packet header sent before SocketTransferEstablishBulkTransport must also
  // be rejected cleanly.
  SocketTransferRequest premature_packet;
  premature_packet.mutable_packet()->set_req_id(0);
  EXPECT_OK(
      SendFramedRequestAndExpectPeerClose(serverb->addr(), premature_packet));

  // 4. Malformed SocketTransferEstablishBulkTransport must poison and close the
  // connection cleanly without corrupting event loop reference counts.
  SocketTransferRequest malformed_establish;
  malformed_establish.mutable_bulk_transport();
  EXPECT_OK(SendFramedRequestAndExpectPeerClose(serverb->addr(),
                                                malformed_establish));

  // 5. Duplicate SocketTransferEstablishBulkTransport on the same connection
  // must close the connection cleanly without racing on bulk_transport_.
  {
    int fd = socket(serverb->addr().address().sa_family,
                    SOCK_STREAM | SOCK_CLOEXEC, /*protocol=*/0);
    ASSERT_GE(fd, 0);
    auto close_fd = absl::MakeCleanup([fd]() { close(fd); });
    SocketAddress target = serverb->addr();
    ASSERT_EQ(connect(fd, reinterpret_cast<const struct sockaddr*>(&target),
                      sizeof(target)),
              0);
    auto init = local_factory->InitBulkTransport();
    SocketTransferRequest establish;
    *establish.mutable_bulk_transport() = init.request;
    std::string payload = establish.SerializeAsString();
    uint32_t header = static_cast<uint32_t>(payload.size());
    std::string frame(reinterpret_cast<const char*>(&header), sizeof(header));
    frame += payload;
    ASSERT_EQ(send(fd, frame.data(), frame.size(), MSG_NOSIGNAL),
              static_cast<ssize_t>(frame.size()));
    // Read the first handshake reply, then send a duplicate establish frame
    // coalesced with a trailing pull frame; the duplicate establish poisons
    // the connection and the trailing pull frame must be ignored without
    // consuming kUuid.
    uint32_t reply_len = 0;
    ASSERT_EQ(recv(fd, &reply_len, sizeof(reply_len), MSG_WAITALL),
              static_cast<ssize_t>(sizeof(reply_len)));
    std::string reply_body(reply_len, '\0');
    ASSERT_EQ(recv(fd, reply_body.data(), reply_len, MSG_WAITALL),
              static_cast<ssize_t>(reply_len));
    std::string pull_payload = premature_pull.SerializeAsString();
    uint32_t pull_header = static_cast<uint32_t>(pull_payload.size());
    std::string coalesced = frame;
    coalesced.append(reinterpret_cast<const char*>(&pull_header),
                     sizeof(pull_header));
    coalesced += pull_payload;
    ASSERT_EQ(send(fd, coalesced.data(), coalesced.size(), MSG_NOSIGNAL),
              static_cast<ssize_t>(coalesced.size()));
    char eof_buf[16];
    EXPECT_LE(recv(fd, eof_buf, sizeof(eof_buf), 0), 0);
  }

  // The legitimate entry registered on serverb must still be intact and
  // pullable by a valid client connection.
  auto [s, cd] = ChunkDestination::MakeStringDest();
  auto conn = servera->Connect(serverb->addr());
  conn->Pull(kUuid, /*buffer_id=*/0, std::move(cd));
  EXPECT_THAT(s.Await(), ::absl_testing::IsOkAndHolds(msg));
  conn = {};
  servera->WaitForQuiesce();
  serverb->WaitForQuiesce();
}

class PoisonDropsConnectionDest : public ChunkDestination {
 public:
  explicit PoisonDropsConnectionDest(
      tsl::RCReference<SocketServer::Connection>* conn_slot)
      : conn_slot_(conn_slot) {}

  absl::Status Put(const void* data, int64_t offset, size_t size,
                   absl::AnyInvocable<void() &&> on_done) override {
    std::move(on_done)();
    return absl::OkStatus();
  }

  void Poison(absl::Status s) override {
    poison_status_ = std::move(s);
    if (conn_slot_ != nullptr && *conn_slot_) {
      // Reacquire SocketNetworkState::mu_ via InjectFailure, then drop the
      // caller's Connection reference inside Poison. Both operations would
      // deadlock if InstallPull or InstallPullList held mu_ across Poison.
      (*conn_slot_)->InjectFailure(SocketServer::Connection::kPoison);
      *conn_slot_ = tsl::RCReference<SocketServer::Connection>();
    }
  }

  const absl::Status& poison_status() const { return poison_status_; }

 private:
  tsl::RCReference<SocketServer::Connection>* conn_slot_;
  absl::Status poison_status_;
};

TEST(ServerTest, PullOnPoisonedConnectionDoesNotHoldMutexAcrossPoison) {
  auto prev_sigpipe = signal(SIGPIPE, SIG_DFL);
  auto restore_sigpipe =
      absl::MakeCleanup([prev_sigpipe]() { signal(SIGPIPE, prev_sigpipe); });

  sockaddr_in6 addr;
  memset(&addr, 0, sizeof(sockaddr_in6));
  addr.sin6_family = AF_INET6;
  auto local_factory = BulkTransportFactory::CreateLocal();
  auto servera = std::make_shared<SocketServer>();
  ASSERT_OK(servera->Start(SocketAddress(addr), local_factory));
  auto serverb = std::make_shared<SocketServer>();
  ASSERT_OK(serverb->Start(SocketAddress(addr), local_factory));

  {
    tsl::RCReference<SocketServer::Connection> conn =
        servera->Connect(serverb->addr());
    conn->InjectFailure(SocketServer::Connection::kPoison);
    auto dest = tsl::MakeRef<PoisonDropsConnectionDest>(&conn);
    conn->Pull(/*uuid=*/1, /*buffer_id=*/0, dest);
    EXPECT_FALSE(conn);
    EXPECT_THAT(
        dest->poison_status(),
        ::absl_testing::StatusIs(absl::StatusCode::kInternal,
                                 ::testing::HasSubstr("InjectFailure")));
  }

  {
    tsl::RCReference<SocketServer::Connection> conn =
        servera->Connect(serverb->addr());
    conn->InjectFailure(SocketServer::Connection::kPoison);
    auto dest0 = tsl::MakeRef<PoisonDropsConnectionDest>(&conn);
    auto dest1 = tsl::MakeRef<PoisonDropsConnectionDest>(nullptr);
    std::vector<int> buffer_ids = {0, 1};
    std::vector<tsl::RCReference<ChunkDestination>> dests = {dest0, dest1};
    conn->Pull(/*uuid=*/2, buffer_ids, std::move(dests));
    EXPECT_FALSE(conn);
    EXPECT_THAT(
        dest0->poison_status(),
        ::absl_testing::StatusIs(absl::StatusCode::kInternal,
                                 ::testing::HasSubstr("InjectFailure")));
    EXPECT_THAT(
        dest1->poison_status(),
        ::absl_testing::StatusIs(absl::StatusCode::kInternal,
                                 ::testing::HasSubstr("InjectFailure")));
  }

  servera->WaitForQuiesce();
  serverb->WaitForQuiesce();
}

class RejectingPutDest : public ChunkDestination {
 public:
  absl::Status Put(const void* data, int64_t offset, size_t size,
                   absl::AnyInvocable<void() &&> on_done) override {
    std::move(on_done)();
    return absl::InvalidArgumentError("put rejected");
  }

  void Poison(absl::Status s) override {
    status_ = std::move(s);
    poisoned_.Notify();
  }

  absl::Status AwaitPoison() {
    poisoned_.WaitForNotification();
    return status_;
  }

 private:
  absl::Notification poisoned_;
  absl::Status status_;
};

TEST(ServerTest, MultiBufferPullListAndPacketHeaderValidation) {
  sockaddr_in6 addr;
  memset(&addr, 0, sizeof(sockaddr_in6));
  addr.sin6_family = AF_INET6;
  auto local_factory = BulkTransportFactory::CreateLocal();
  auto servera = std::make_shared<SocketServer>();
  ASSERT_OK(servera->Start(SocketAddress(addr), local_factory));
  auto serverb = std::make_shared<SocketServer>();
  ASSERT_OK(serverb->Start(SocketAddress(addr), local_factory));

  // Exercise healthy InstallPullList with multiple destinations.
  {
    constexpr uint64_t kUuid = 77;
    serverb->AwaitPull(kUuid, PullTable::MakeStringEntry({"first", "second"}));
    auto [s0, cd0] = ChunkDestination::MakeStringDest();
    auto [s1, cd1] = ChunkDestination::MakeStringDest();
    auto conn = servera->Connect(serverb->addr());
    std::vector<int> bids = {0, 1};
    std::vector<tsl::RCReference<ChunkDestination>> dests = {std::move(cd0),
                                                             std::move(cd1)};
    conn->Pull(kUuid, bids, std::move(dests));
    EXPECT_THAT(s0.Await(), ::absl_testing::IsOkAndHolds("first"));
    EXPECT_THAT(s1.Await(), ::absl_testing::IsOkAndHolds("second"));
    conn = {};
  }

  // Exercise HandlePacket(SocketTransferPacketHeader) with out of range
  // bulk_transport_id (> INT_MAX) via a raw listening socket peer.
  {
    int listen_fd = socket(AF_INET6, SOCK_STREAM | SOCK_CLOEXEC, 0);
    ASSERT_GE(listen_fd, 0);
    auto close_listen = absl::MakeCleanup([listen_fd]() { close(listen_fd); });
    sockaddr_in6 peer_addr;
    memset(&peer_addr, 0, sizeof(peer_addr));
    peer_addr.sin6_family = AF_INET6;
    peer_addr.sin6_addr = in6addr_loopback;
    ASSERT_EQ(bind(listen_fd, reinterpret_cast<const sockaddr*>(&peer_addr),
                   sizeof(peer_addr)),
              0);
    ASSERT_EQ(listen(listen_fd, 1), 0);
    socklen_t addr_len = sizeof(peer_addr);
    ASSERT_EQ(getsockname(listen_fd, reinterpret_cast<sockaddr*>(&peer_addr),
                          &addr_len),
              0);

    auto [s_bad, cd_bad] = ChunkDestination::MakeStringDest();
    auto rejecting_dest = tsl::MakeRef<RejectingPutDest>();
    auto conn = servera->Connect(SocketAddress(peer_addr));
    conn->Pull(/*uuid=*/88, /*buffer_id=*/0, std::move(cd_bad));
    conn->Pull(/*uuid=*/89, /*buffer_id=*/0, rejecting_dest);

    int accepted = accept4(listen_fd, nullptr, nullptr, SOCK_CLOEXEC);
    ASSERT_GE(accepted, 0);
    auto close_accepted = absl::MakeCleanup([accepted]() { close(accepted); });

    // Read client's establish frame, send establish reply, then test both
    // bulk_transport_id > INT_MAX and dest->Put returning non OK.
    uint32_t frame_len = 0;
    ASSERT_EQ(recv(accepted, &frame_len, sizeof(frame_len), MSG_WAITALL),
              static_cast<ssize_t>(sizeof(frame_len)));
    std::string body(frame_len, '\0');
    ASSERT_EQ(recv(accepted, body.data(), frame_len, MSG_WAITALL),
              static_cast<ssize_t>(frame_len));
    SocketTransferRequest client_establish;
    ASSERT_TRUE(client_establish.ParseFromString(body));
    ASSERT_OK_AND_ASSIGN(auto recv_bt, local_factory->RecvBulkTransport(
                                           client_establish.bulk_transport()));

    auto send_proto = [accepted](const SocketTransferRequest& proto) {
      std::string bytes = proto.SerializeAsString();
      uint32_t len = static_cast<uint32_t>(bytes.size());
      std::string wire(reinterpret_cast<const char*>(&len), sizeof(len));
      wire += bytes;
      return send(accepted, wire.data(), wire.size(), MSG_NOSIGNAL) ==
             static_cast<ssize_t>(wire.size());
    };

    SocketTransferRequest reply_establish;
    *reply_establish.mutable_bulk_transport() = recv_bt.request;
    ASSERT_TRUE(send_proto(reply_establish));

    SocketTransferRequest bad_pkt;
    bad_pkt.mutable_packet()->set_req_id(0);
    bad_pkt.mutable_packet()->set_offset(0);
    bad_pkt.mutable_packet()->set_size(0);
    bad_pkt.mutable_packet()->set_is_largest(true);
    bad_pkt.mutable_packet()->set_bulk_transport_id(0x100000000ULL);
    ASSERT_TRUE(send_proto(bad_pkt));

    EXPECT_THAT(s_bad.Await(), ::absl_testing::StatusIs(
                                   absl::StatusCode::kInvalidArgument,
                                   ::testing::HasSubstr("bulk_transport_id")));

    char empty = 0;
    BulkTransportInterface::SendMessage zero_bulk;
    zero_bulk.data = &empty;
    zero_bulk.size = 0;
    zero_bulk.on_send = [&](absl::StatusOr<int> bond_id, size_t sz) {
      ASSERT_OK(bond_id);
      SocketTransferRequest pkt1;
      pkt1.mutable_packet()->set_req_id(1);
      pkt1.mutable_packet()->set_offset(0);
      pkt1.mutable_packet()->set_size(sz);
      pkt1.mutable_packet()->set_is_largest(true);
      pkt1.mutable_packet()->set_bulk_transport_id(*bond_id);
      ASSERT_TRUE(send_proto(pkt1));
    };
    zero_bulk.on_done = []() {};
    recv_bt.bulk_transport->Send(std::move(zero_bulk));

    EXPECT_THAT(rejecting_dest->AwaitPoison(),
                ::absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                         ::testing::HasSubstr("put rejected")));

    // Exercise SendClosed poisoning a pending destination before RecvClosed by
    // holding the event loop after poll() returns revents == 0, resetting the
    // peer socket with SO_LINGER=0 while queueing a Pull write, and verifying
    // the destination receives the send() failure status.
    absl::Notification loop_parked;
    absl::Notification release_loop;
    PollEventLoop::GetDefault()->Schedule([&]() {
      loop_parked.Notify();
      release_loop.WaitForNotification();
    });
    loop_parked.WaitForNotification();

    auto [s_send_closed, cd_send_closed] = ChunkDestination::MakeStringDest();
    conn->Pull(/*uuid=*/90, /*buffer_id=*/0, std::move(cd_send_closed));
    PollEventLoop::GetDefault()->Schedule([&]() {
      struct linger rst_linger = {/*l_onoff=*/1, /*l_linger=*/0};
      // NOLINTNEXTLINE(misc-include-cleaner)
      setsockopt(accepted, SOL_SOCKET, SO_LINGER, &rst_linger,
                 sizeof(rst_linger));
      close(accepted);
    });
    std::move(close_accepted).Cancel();
    release_loop.Notify();

    EXPECT_THAT(
        s_send_closed.Await(),
        ::absl_testing::StatusIs(absl::StatusCode::kInternal,
                                 ::testing::HasSubstr("send() failed")));
    conn = {};
  }

  // Also exercise clean peer EOF (RecvClosed with OkStatus -> Connection closed
  // recv() == 0) poisoning a pending pull destination.
  {
    int listen_fd = socket(AF_INET6, SOCK_STREAM | SOCK_CLOEXEC, 0);
    ASSERT_GE(listen_fd, 0);
    auto close_listen = absl::MakeCleanup([listen_fd]() { close(listen_fd); });
    sockaddr_in6 peer_addr;
    memset(&peer_addr, 0, sizeof(peer_addr));
    peer_addr.sin6_family = AF_INET6;
    peer_addr.sin6_addr = in6addr_loopback;
    ASSERT_EQ(bind(listen_fd, reinterpret_cast<const sockaddr*>(&peer_addr),
                   sizeof(peer_addr)),
              0);
    ASSERT_EQ(listen(listen_fd, 1), 0);
    socklen_t addr_len = sizeof(peer_addr);
    ASSERT_EQ(getsockname(listen_fd, reinterpret_cast<sockaddr*>(&peer_addr),
                          &addr_len),
              0);

    auto [s_eof, cd_eof] = ChunkDestination::MakeStringDest();
    auto conn = servera->Connect(SocketAddress(peer_addr));
    conn->Pull(/*uuid=*/91, /*buffer_id=*/0, std::move(cd_eof));
    int accepted = accept4(listen_fd, nullptr, nullptr, SOCK_CLOEXEC);
    ASSERT_GE(accepted, 0);
    auto close_accepted = absl::MakeCleanup([accepted]() { close(accepted); });
    uint32_t frame_len = 0;
    ASSERT_EQ(recv(accepted, &frame_len, sizeof(frame_len), MSG_WAITALL),
              static_cast<ssize_t>(sizeof(frame_len)));
    ASSERT_EQ(shutdown(accepted, SHUT_WR), 0);
    EXPECT_THAT(s_eof.Await(),
                ::absl_testing::StatusIs(
                    absl::StatusCode::kInternal,
                    ::testing::HasSubstr("Connection closed recv() == 0.")));
    conn = {};
  }

  servera->WaitForQuiesce();
  serverb->WaitForQuiesce();
}

}  // namespace
}  // namespace aux
