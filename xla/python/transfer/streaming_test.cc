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
#include "xla/python/transfer/streaming.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <string.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tsl/platform/env.h"
#include "xla/python/transfer/transfer_socket.pb.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace aux {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class LocalConnectionState : public ConnectionState {
 public:
  explicit LocalConnectionState(std::shared_ptr<PullTable> table)
      : table_(std::move(table)) {}
  ~LocalConnectionState() override = default;

  void Send(size_t req_id, const void* data, size_t offset, size_t size,
            bool is_largest, absl::AnyInvocable<void() &&> on_done) override {
    tsl::RCReference<ChunkDestination> dest;
    {
      absl::MutexLock l(mu_);
      auto it = dests_.find(req_id);
      CHECK(it != dests_.end());
      if (is_largest) {
        it->second.transferred_size += offset;
      } else {
        it->second.transferred_size -= size;
      }
      if (it->second.transferred_size == 0) {
        dest = std::move(it->second.dest);
        dests_.erase(it);
      } else {
        dest = it->second.dest;
      }
    }
    CHECK_OK(dest->Put(data, offset, size, std::move(on_done)));
  }

  void Pull(uint64_t uuid, int buf_id,
            tsl::RCReference<ChunkDestination> dest) {
    size_t req_id;
    {
      absl::MutexLock l(mu_);
      dests_[next_req_id_].dest = std::move(dest);
      req_id = next_req_id_;
      ++next_req_id_;
    }
    SocketTransferPullRequest req;
    req.set_uuid(uuid);
    req.add_buffer_ids(buf_id);
    table_->Handle(tsl::FormRef(this), req, req_id);
  }

 private:
  absl::Mutex mu_;
  std::shared_ptr<PullTable> table_;
  uint64_t next_req_id_ = 0;
  struct DestState {
    ssize_t transferred_size = 0;
    tsl::RCReference<ChunkDestination> dest;
  };
  absl::flat_hash_map<uint64_t, DestState> dests_;
};

class ErrorCapturingConnectionState : public ConnectionState {
 public:
  void Send(size_t req_id, const void* data, size_t offset, size_t size,
            bool is_largest, absl::AnyInvocable<void() &&> on_done) override {
    std::move(on_done)();
  }

  void SendError(size_t req_id, size_t offset, size_t size, bool is_largest,
                 absl::Status status) override {
    absl::MutexLock l(mu);
    errors.push_back(status);
  }

  absl::Mutex mu;
  std::vector<absl::Status> errors;
};

class MockEntry : public PullTable::Entry {
 public:
  bool Handle(tsl::RCReference<ConnectionState> state,
              const SocketTransferPullRequest& req,
              size_t base_req_id) override {
    absl::MutexLock l(mu);
    called = true;
    return true;
  }

  bool IsCalled() {
    absl::MutexLock l(mu);
    return called;
  }

 private:
  absl::Mutex mu;
  bool called = false;
};

TEST(BulkTransferInterface, PullTableInterfaces) {
  auto table = std::make_shared<PullTable>();
  std::string msg("secret message");
  uint64_t uuid = 5678;
  table->AwaitPull(uuid, PullTable::MakeStringEntry({msg}));
  auto ref = tsl::MakeRef<LocalConnectionState>(table);
  auto [s, cd] = ChunkDestination::MakeStringDest();
  ref->Pull(uuid, 0, std::move(cd));
  EXPECT_EQ(s.Await().value(), msg);
}

TEST(BulkTransferInterface, AwaitAfterInterfaces) {
  auto table = std::make_shared<PullTable>();
  std::string msg("secret message");
  uint64_t uuid = 5678;
  auto ref = tsl::MakeRef<LocalConnectionState>(table);
  auto [s, cd] = ChunkDestination::MakeStringDest();
  ref->Pull(uuid, 0, std::move(cd));
  table->AwaitPull(uuid, PullTable::MakeStringEntry({msg}));
  EXPECT_EQ(s.Await().value(), msg);
}

TEST(BulkTransferInterface, LocalTransport) {
  auto transport_factory = BulkTransportFactory::CreateLocal();
  auto a = transport_factory->InitBulkTransport();
  ASSERT_OK_AND_ASSIGN(auto b, transport_factory->RecvBulkTransport(a.request));
  std::move(a.start_bulk_transport)(b.request);

  std::string test_message = "secret message";

  absl::Notification is_sent;
  int assigned_bond_id;
  size_t send_size;
  a.bulk_transport->Send(BulkTransportInterface::MakeMessage(
      test_message, [&](int bond_id, size_t size) {
        assigned_bond_id = bond_id;
        send_size = size;
        is_sent.Notify();
      }));

  is_sent.WaitForNotification();
  absl::Notification recv_done;
  absl::StatusOr<BulkTransportInterface::Message> recv_message;
  b.bulk_transport->Recv(
      send_size, assigned_bond_id,
      [&](absl::StatusOr<BulkTransportInterface::Message> message) {
        recv_message = std::move(message);
        recv_done.Notify();
      });
  recv_done.WaitForNotification();
  ASSERT_TRUE(recv_message.status().ok()) << recv_message.status();
  EXPECT_EQ(test_message,
            std::string(reinterpret_cast<const char*>(recv_message->data),
                        recv_message->size));
}

TEST(BulkTransferInterface, ClosedLocalTransport) {
  auto transport_factory = BulkTransportFactory::CreateLocal();
  auto a = transport_factory->InitBulkTransport();
  ASSERT_OK_AND_ASSIGN(auto b, transport_factory->RecvBulkTransport(a.request));
  std::move(a.start_bulk_transport)(b.request);
  a.bulk_transport = nullptr;
  absl::Notification recv_done;
  absl::StatusOr<BulkTransportInterface::Message> recv_message;
  b.bulk_transport->Recv(
      80, 0, [&](absl::StatusOr<BulkTransportInterface::Message> message) {
        recv_message = std::move(message);
        recv_done.Notify();
      });
  recv_done.WaitForNotification();
  ASSERT_FALSE(recv_message.ok());
}

absl::Status PutSync(ChunkDestination* dest, size_t offset, std::string chunk) {
  absl::Notification on_done;
  auto s = dest->Put(chunk.data(), offset, chunk.size(),
                     [&on_done]() { on_done.Notify(); });
  if (!s.ok()) {
    return s;
  }
  on_done.WaitForNotification();
  return absl::OkStatus();
}

TEST(ChunkDestination, StringChunkDest) {
  auto [result, dest] = ChunkDestination::MakeStringDest();
  auto s = PutSync(dest.get(), 7, "message");
  ASSERT_TRUE(s.ok()) << s;
  s = PutSync(dest.get(), 0, "secret ");
  ASSERT_TRUE(s.ok()) << s;
  dest = {};

  auto value = result.Await();
  ASSERT_TRUE(value.ok()) << value.status();
  EXPECT_EQ(*value, "secret message");
}

TEST(ChunkDestination, StringChunkDestWithGaps) {
  auto [result, dest] = ChunkDestination::MakeStringDest();
  auto s = PutSync(dest.get(), 8, "message");
  ASSERT_TRUE(s.ok()) << s;
  s = PutSync(dest.get(), 0, "secret ");
  ASSERT_TRUE(s.ok()) << s;
  dest = {};

  auto value = result.Await();
  ASSERT_FALSE(value.ok());
}

TEST(SlabAllocator, BasicSubAllocations) {
  auto alloc1_or = AllocateNetworkPinnedMemory(4096 * 4);
  if (absl::IsNotFound(alloc1_or.status())) {
    GTEST_SKIP() << "SO_ZEROCOPY not supported: " << alloc1_or.status();
  }
  auto alloc2_or = AllocateAlignedMemory(4096 * 4);
  ASSERT_TRUE(alloc1_or.ok()) << alloc1_or.status();
  ASSERT_TRUE(alloc2_or.ok()) << alloc2_or.status();
  for (auto alloc : {*alloc1_or, *alloc2_or}) {
    absl::Mutex mu;
    std::deque<SlabAllocator::Allocation> allocs;
    auto thread = std::unique_ptr<tsl::Thread>(
        tsl::Env::Default()->StartThread({}, "test-thread", [&] {
          for (size_t i = 0; i < 200; ++i) {
            absl::MutexLock l(mu);
            auto cond = [&]() {
              return allocs.size() >= std::min(static_cast<size_t>(200 - i),
                                               static_cast<size_t>(4));
            };
            mu.Await(absl::Condition(&cond));
            std::move(allocs.front().on_done)();
            allocs.pop_front();
          }
        }));
    SlabAllocator allocator(alloc, 4096);
    for (size_t i = 0; i < 200; ++i) {
      auto alloc = allocator.Allocate(allocator.max_allocation_size());
      absl::MutexLock l(mu);
      allocs.push_back(std::move(alloc));
    }
  }
}

TEST(InvalidAllocator, InvalidPinnedAlloc) {
  auto alloc1_or = AllocateNetworkPinnedMemory(1l << 49);
  ASSERT_FALSE(alloc1_or.ok());
}

TEST(InvalidAllocator, InvalidAlignedAlloc) {
  auto alloc2_or = AllocateAlignedMemory(1l << 49);
  ASSERT_FALSE(alloc2_or.ok());
}

class SelfResettingPullTableEntry : public PullTable::Entry {
 public:
  explicit SelfResettingPullTableEntry(std::shared_ptr<PullTable> table)
      : table_(std::move(table)) {}

  bool Handle(tsl::RCReference<ConnectionState> state,
              const SocketTransferPullRequest& req,
              size_t base_req_id) override {
    table_->Reset();
    return true;
  }

 private:
  std::shared_ptr<PullTable> table_;
};

TEST(PullTable, PullTableRace) {
  auto table = std::make_shared<PullTable>();
  table->AwaitPull(6, tsl::MakeRef<SelfResettingPullTableEntry>(table));
  SocketTransferPullRequest req;
  req.set_uuid(6);
  table->Handle({}, req, 0);
}

TEST(PullTable, DropExpiredAwaitPulls) {
  auto table = std::make_shared<PullTable>();
  uint64_t uuid = 1234;
  auto entry = PullTable::MakeStringEntry({"data"});

  absl::Time now = absl::UnixEpoch();
  absl::Time timeout = now - absl::Seconds(10);

  table->AwaitPull(uuid, entry, timeout);

  table->DropExpiredPulls(now);

  auto state = tsl::MakeRef<ErrorCapturingConnectionState>();
  SocketTransferPullRequest req;
  req.set_uuid(uuid);
  req.add_buffer_ids(0);
  table->Handle(state, req, 0);

  auto mock_entry = tsl::MakeRef<MockEntry>();
  table->AwaitPull(uuid, mock_entry);

  EXPECT_TRUE(mock_entry->IsCalled());
}

TEST(PullTable, DropExpiredPausedPulls) {
  auto table = std::make_shared<PullTable>();
  uint64_t uuid = 1234;
  auto state = tsl::MakeRef<ErrorCapturingConnectionState>();

  absl::Time now = absl::UnixEpoch();
  absl::Time timeout = now - absl::Seconds(10);

  SocketTransferPullRequest req;
  req.set_uuid(uuid);
  req.add_buffer_ids(0);
  table->Handle(state, req, 0, timeout);

  table->DropExpiredPulls(now);

  {
    absl::MutexLock l(state->mu);
    ASSERT_EQ(state->errors.size(), 1);
    EXPECT_EQ(state->errors[0].code(), absl::StatusCode::kDeadlineExceeded);
  }

  auto mock_entry = tsl::MakeRef<MockEntry>();
  table->AwaitPull(uuid, mock_entry);

  EXPECT_FALSE(mock_entry->IsCalled());
}

class RecordingPullConnectionState : public ConnectionState {
 public:
  explicit RecordingPullConnectionState(
      absl::Notification* on_first_error = nullptr)
      : on_first_error_(on_first_error) {}

  void Send(size_t req_id, const void* data, size_t offset, size_t size,
            bool is_largest, absl::AnyInvocable<void() &&> on_done) override {
    std::string payload =
        size == 0 ? std::string()
                  : std::string(reinterpret_cast<const char*>(data), size);
    std::move(on_done)();
    absl::MutexLock l(mu_);
    EXPECT_EQ(offset, 0u);
    EXPECT_TRUE(is_largest);
    payloads_[req_id] = std::move(payload);
  }

  void SendError(size_t req_id, size_t offset, size_t size, bool is_largest,
                 absl::Status status) override {
    EXPECT_EQ(offset, 0u);
    EXPECT_TRUE(is_largest);
    bool should_notify = false;
    {
      absl::MutexLock l(mu_);
      errors_[req_id] = std::move(status);
      error_sizes_[req_id] = size;
      if (on_first_error_ != nullptr && !notified_first_error_) {
        notified_first_error_ = true;
        should_notify = true;
      }
    }
    if (should_notify) {
      on_first_error_->Notify();
    }
  }

  absl::flat_hash_map<size_t, std::string> payloads() {
    absl::MutexLock l(mu_);
    return payloads_;
  }

  absl::flat_hash_map<size_t, absl::Status> errors() {
    absl::MutexLock l(mu_);
    return errors_;
  }

  absl::flat_hash_map<size_t, size_t> error_sizes() {
    absl::MutexLock l(mu_);
    return error_sizes_;
  }

 private:
  absl::Notification* on_first_error_;
  absl::Mutex mu_;
  bool notified_first_error_ ABSL_GUARDED_BY(mu_) = false;
  absl::flat_hash_map<size_t, std::string> payloads_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<size_t, absl::Status> errors_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<size_t, size_t> error_sizes_ ABSL_GUARDED_BY(mu_);
};

TEST(PullTable, PausedDuplicateFetchesAreNotRequeuedAfterEntryCompletes) {
  auto table = std::make_shared<PullTable>();
  constexpr uint64_t kUuid = 42;
  auto conn = tsl::MakeRef<RecordingPullConnectionState>();

  SocketTransferPullRequest req1;
  req1.set_uuid(kUuid);
  req1.add_buffer_ids(0);
  table->Handle(conn, req1, /*base_req_id=*/10);

  SocketTransferPullRequest req2;
  req2.set_uuid(kUuid);
  req2.add_buffer_ids(0);
  req2.add_buffer_ids(9);
  table->Handle(conn, req2, /*base_req_id=*/20);

  table->AwaitPull(kUuid, PullTable::MakeStringEntry({"first_entry_payload"}),
                   absl::UnixEpoch() + absl::Seconds(10));

  auto payloads = conn->payloads();
  auto errors = conn->errors();
  auto error_sizes = conn->error_sizes();
  ASSERT_EQ(payloads.size(), 1u);
  EXPECT_EQ(payloads[10], "first_entry_payload");
  ASSERT_EQ(errors.size(), 2u);
  EXPECT_THAT(errors[20],
              StatusIs(absl::StatusCode::kInternal, HasSubstr("already")));
  EXPECT_EQ(error_sizes[20], std::string("first_entry_payload").size());
  EXPECT_THAT(errors[21], StatusIs(absl::StatusCode::kInvalidArgument,
                                   HasSubstr("StringVectorPullTableEntry")));
  EXPECT_EQ(error_sizes[21], 0u);

  // Verify AwaitPull immediately erased the completed entry from entries_
  // before any expiration pass runs (so a subsequent pull pauses rather than
  // hitting the completed entry), and verify DropExpiredPulls sees no dangling
  // entry_heap_ node.
  auto post_conn = tsl::MakeRef<RecordingPullConnectionState>();
  SocketTransferPullRequest post_req;
  post_req.set_uuid(kUuid);
  post_req.add_buffer_ids(0);
  table->Handle(post_conn, post_req, /*base_req_id=*/30);
  EXPECT_TRUE(post_conn->errors().empty());
  table->DropExpiredPulls(absl::UnixEpoch() + absl::Seconds(20));
  table->Reset();
  EXPECT_THAT(post_conn->errors()[30],
              StatusIs(absl::StatusCode::kInternal, HasSubstr("Reset")));

  auto second_entry = tsl::MakeRef<MockEntry>();
  table->AwaitPull(kUuid, second_entry);
  EXPECT_FALSE(second_entry->IsCalled());

  // Verify RemoveEntryIfMatching does not erase a replacement entry installed
  // reentrantly under the same uuid during Handle.
  class ReplacingEntry : public PullTable::Entry {
   public:
    ReplacingEntry(PullTable* table, uint64_t uuid,
                   tsl::RCReference<PullTable::Entry> replacement)
        : table_(table), uuid_(uuid), replacement_(std::move(replacement)) {}
    bool Handle(tsl::RCReference<ConnectionState> state,
                const SocketTransferPullRequest& req,
                size_t base_req_id) override {
      state->Send(base_req_id, "r", 0, 1, true, []() {});
      table_->AwaitPull(uuid_, replacement_);
      return true;
    }

   private:
    PullTable* table_;
    uint64_t uuid_;
    tsl::RCReference<PullTable::Entry> replacement_;
  };
  constexpr uint64_t kReplaceUuid = 777;
  auto repl_conn = tsl::MakeRef<RecordingPullConnectionState>();
  SocketTransferPullRequest repl_planted;
  repl_planted.set_uuid(kReplaceUuid);
  repl_planted.add_buffer_ids(0);
  table->Handle(repl_conn, repl_planted, /*base_req_id=*/0);
  auto installed_replacement = tsl::MakeRef<MockEntry>();
  table->AwaitPull(kReplaceUuid,
                   tsl::MakeRef<ReplacingEntry>(table.get(), kReplaceUuid,
                                                installed_replacement));
  EXPECT_FALSE(installed_replacement->IsCalled());
  table->Handle(repl_conn, repl_planted, /*base_req_id=*/1);
  EXPECT_TRUE(installed_replacement->IsCalled());

  // Verify RemoveIntrusive unlinks completed entries from entry_heap_ so heap
  // slot reuse cannot mask expiration of remaining entries.
  PullTable heap_table;
  const absl::Time t0 = absl::UnixEpoch();
  SocketTransferPullRequest heap_planted;
  heap_planted.set_uuid(1001);
  heap_planted.add_buffer_ids(0);
  auto heap_conn = tsl::MakeRef<RecordingPullConnectionState>();
  heap_table.Handle(heap_conn, heap_planted, /*base_req_id=*/0);

  auto expiring_entry = tsl::MakeRef<MockEntry>();
  heap_table.AwaitPull(/*uuid=*/1003, expiring_entry, t0 + absl::Seconds(10));
  heap_table.AwaitPull(/*uuid=*/1001, PullTable::MakeStringEntry({"v"}),
                       t0 + absl::Seconds(5));
  auto long_lived_entry = tsl::MakeRef<MockEntry>();
  heap_table.AwaitPull(/*uuid=*/1002, long_lived_entry,
                       t0 + absl::Seconds(100));
  heap_table.DropExpiredPulls(t0 + absl::Seconds(20));

  SocketTransferPullRequest check_expired;
  check_expired.set_uuid(1003);
  check_expired.add_buffer_ids(0);
  heap_table.Handle(heap_conn, check_expired, /*base_req_id=*/10);
  EXPECT_FALSE(expiring_entry->IsCalled());
}

TEST(PullTable, MultiBufferStringEntryAcrossSeparatePullsAndInvalidIds) {
  auto table = std::make_shared<PullTable>();
  constexpr uint64_t kUuid = 99;
  table->AwaitPull(kUuid, PullTable::MakeStringEntry({"alpha", "", "omega"}));

  auto err_conn = tsl::MakeRef<RecordingPullConnectionState>();
  SocketTransferPullRequest bad_req;
  bad_req.set_uuid(kUuid);
  bad_req.add_buffer_ids(3);
  bad_req.add_buffer_ids(4);
  bad_req.add_buffer_ids(5);
  table->Handle(err_conn, bad_req, /*base_req_id=*/100);
  auto bad_errors = err_conn->errors();
  ASSERT_EQ(bad_errors.size(), 3u);
  for (size_t req_id : {size_t{100}, size_t{101}, size_t{102}}) {
    EXPECT_THAT(bad_errors[req_id],
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("StringVectorPullTableEntry")));
  }

  // Out of range buffer_ids must not evict the entry before valid buffers are
  // pulled.
  auto ref = tsl::MakeRef<LocalConnectionState>(table);
  auto [f0, cd0] = ChunkDestination::MakeStringDest();
  auto [f1, cd1] = ChunkDestination::MakeStringDest();
  auto [f2, cd2] = ChunkDestination::MakeStringDest();
  ref->Pull(kUuid, 0, std::move(cd0));
  ref->Pull(kUuid, 1, std::move(cd1));
  ref->Pull(kUuid, 2, std::move(cd2));

  EXPECT_THAT(f0.Await(), IsOkAndHolds("alpha"));
  EXPECT_THAT(f1.Await(), IsOkAndHolds(""));
  EXPECT_THAT(f2.Await(), IsOkAndHolds("omega"));

  // Now that all 3 valid buffers were pulled, the entry must be evicted; a
  // subsequent pull queues in paused_fetches_ and receives PullTable::Reset.
  auto post_conn = tsl::MakeRef<RecordingPullConnectionState>();
  SocketTransferPullRequest post_req;
  post_req.set_uuid(kUuid);
  post_req.add_buffer_ids(0);
  table->Handle(post_conn, post_req, /*base_req_id=*/200);
  EXPECT_TRUE(post_conn->errors().empty());
  table->Reset();
  EXPECT_THAT(post_conn->errors()[200],
              StatusIs(absl::StatusCode::kInternal, HasSubstr("Reset")));
}

TEST(PullTable, ConcurrentAwaitPullReplayAndHandle) {
  constexpr uint64_t kUuid = 7;
  constexpr int kIterations = 16;
  constexpr int kNumBuffers = 4;

  for (int iter = 0; iter < kIterations; ++iter) {
    PullTable table;
    absl::Notification replay_started;
    auto replay_conn =
        tsl::MakeRef<RecordingPullConnectionState>(&replay_started);

    SocketTransferPullRequest planted;
    planted.set_uuid(kUuid);
    planted.add_buffer_ids(999);
    for (int pad = 0; pad < 256; ++pad) {
      planted.add_buffer_ids(999);
    }
    planted.add_buffer_ids(0);
    planted.add_buffer_ids(1);
    table.Handle(replay_conn, planted, /*base_req_id=*/0);

    auto race_conn = tsl::MakeRef<RecordingPullConnectionState>();
    std::unique_ptr<tsl::Thread> worker(
        tsl::Env::Default()->StartThread({}, "pull-worker", [&]() {
          replay_started.WaitForNotification();
          for (int bid = 0; bid < kNumBuffers; ++bid) {
            SocketTransferPullRequest req;
            req.set_uuid(kUuid);
            req.add_buffer_ids(bid);
            table.Handle(race_conn, req, /*base_req_id=*/1000 + bid);
          }
        }));

    table.AwaitPull(kUuid,
                    PullTable::MakeStringEntry({"b0", "b1", "b2", "b3"}));
    worker.reset();

    auto replay_payloads = replay_conn->payloads();
    auto race_payloads = race_conn->payloads();
    EXPECT_EQ(replay_payloads.size() + race_payloads.size(),
              static_cast<size_t>(kNumBuffers));
  }
}

TEST(PullTable, EmptyEntryIsNotRetainedInTable) {
  PullTable table;
  constexpr uint64_t kUuid = 333;
  table.AwaitPull(kUuid, PullTable::MakeStringEntry({}));

  auto conn = tsl::MakeRef<RecordingPullConnectionState>();
  SocketTransferPullRequest req;
  req.set_uuid(kUuid);
  req.add_buffer_ids(0);
  table.Handle(conn, req, /*base_req_id=*/50);
  EXPECT_TRUE(conn->errors().empty());
  table.Reset();
  EXPECT_THAT(conn->errors()[50],
              StatusIs(absl::StatusCode::kInternal, HasSubstr("Reset")));
}

class DestructorChecksTableLockEntry : public PullTable::Entry {
 public:
  DestructorChecksTableLockEntry(PullTable* table, bool* destroyed)
      : table_(table), destroyed_(destroyed) {}
  ~DestructorChecksTableLockEntry() override {
    table_->DropExpiredPulls(absl::InfinitePast());
    *destroyed_ = true;
  }

  bool Handle(tsl::RCReference<ConnectionState> state,
              const SocketTransferPullRequest& req,
              size_t base_req_id) override {
    return true;
  }

 private:
  PullTable* table_;
  bool* destroyed_;
};

TEST(PullTable, EntryDestructionRunsOutsideTableLockOnReplaceAndReset) {
  PullTable table;
  const absl::Time t0 = absl::UnixEpoch();

  bool destroyed_on_replace = false;
  table.AwaitPull(/*uuid=*/401,
                  tsl::MakeRef<DestructorChecksTableLockEntry>(
                      &table, &destroyed_on_replace),
                  t0 + absl::Seconds(10));
  EXPECT_FALSE(destroyed_on_replace);
  table.AwaitPull(/*uuid=*/401, PullTable::MakeStringEntry({"replacement"}));
  EXPECT_TRUE(destroyed_on_replace);
  table.DropExpiredPulls(t0 + absl::Seconds(20));

  bool destroyed_on_empty_replace = false;
  table.AwaitPull(/*uuid=*/402, tsl::MakeRef<DestructorChecksTableLockEntry>(
                                    &table, &destroyed_on_empty_replace));
  EXPECT_FALSE(destroyed_on_empty_replace);
  table.AwaitPull(/*uuid=*/402, PullTable::MakeStringEntry({}));
  EXPECT_TRUE(destroyed_on_empty_replace);
  auto post_empty_replace_conn = tsl::MakeRef<RecordingPullConnectionState>();
  SocketTransferPullRequest req_402;
  req_402.set_uuid(402);
  req_402.add_buffer_ids(0);
  table.Handle(post_empty_replace_conn, req_402, /*base_req_id=*/99);
  EXPECT_TRUE(post_empty_replace_conn->errors().empty());
  table.Reset();
  EXPECT_THAT(post_empty_replace_conn->errors()[99],
              StatusIs(absl::StatusCode::kInternal, HasSubstr("Reset")));

  bool destroyed_on_reset = false;
  table.AwaitPull(
      /*uuid=*/403,
      tsl::MakeRef<DestructorChecksTableLockEntry>(&table, &destroyed_on_reset),
      t0 + absl::Seconds(10));
  EXPECT_FALSE(destroyed_on_reset);
  table.Reset();
  EXPECT_TRUE(destroyed_on_reset);
  table.DropExpiredPulls(t0 + absl::Seconds(20));
}

TEST(StringDestAndLocalBulkTransport, PoisonAndRecvValidation) {
  {
    auto [fut, dest] = ChunkDestination::MakeStringDest();
    dest->Poison(absl::InternalError("poisoned string dest"));
    dest->Poison(absl::InternalError("ignored second poison"));
    dest = tsl::RCReference<ChunkDestination>();
    EXPECT_THAT(fut.Await(), StatusIs(absl::StatusCode::kInternal,
                                      HasSubstr("poisoned string dest")));
  }

  auto factory = BulkTransportFactory::CreateLocal();
  auto init = factory->InitBulkTransport();
  ASSERT_OK_AND_ASSIGN(auto recv_info,
                       factory->RecvBulkTransport(init.request));
  std::move(init.start_bulk_transport)(recv_info.request);

  {
    absl::StatusOr<BulkTransportInterface::Message> got;
    recv_info.bulk_transport->Recv(
        /*size=*/4, /*bond_id=*/1,
        [&got](absl::StatusOr<BulkTransportInterface::Message> msg) {
          got = std::move(msg);
        });
    EXPECT_THAT(got, StatusIs(absl::StatusCode::kInvalidArgument,
                              HasSubstr("Invalid bulk_transport_id")));
  }

  {
    char buf[8] = {};
    bool send_on_done = false;
    BulkTransportInterface::SendMessage send_msg;
    send_msg.data = buf;
    send_msg.size = sizeof(buf);
    send_msg.on_send = [](absl::StatusOr<int> bond_id, size_t size) {
      EXPECT_THAT(bond_id, IsOkAndHolds(0));
      EXPECT_EQ(size, 8u);
    };
    send_msg.on_done = [&send_on_done]() { send_on_done = true; };
    init.bulk_transport->Send(std::move(send_msg));

    absl::StatusOr<BulkTransportInterface::Message> got;
    recv_info.bulk_transport->Recv(
        /*size=*/4, /*bond_id=*/0,
        [&got](absl::StatusOr<BulkTransportInterface::Message> msg) {
          got = std::move(msg);
        });
    EXPECT_TRUE(send_on_done);
    EXPECT_THAT(got,
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Mismatched bulk transport recv size")));
  }
}

}  // namespace
}  // namespace aux
