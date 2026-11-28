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

#include "xla/backends/cpu/collectives/in_process_communicator.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

#include <gmock/gmock.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/time/time.h"
#include "xla/backends/cpu/collectives/cpu_clique_key.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/backends/cpu/collectives/in_process_collectives.h"
#include "xla/backends/cpu/runtime/all_reduce_thunk.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/collective_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/executable_run_options.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_rendezvous.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

using ::absl_testing::StatusIs;

constexpr absl::Duration kTimeout = absl::Seconds(5);

int64_t NextRunId() {
  static std::atomic<int64_t> counter{1};
  return counter.fetch_add(1, std::memory_order_relaxed);
}

std::vector<std::unique_ptr<Communicator>> MakeCommunicators(size_t num_ranks) {
  InProcessCollectives collectives;
  std::vector<GlobalDeviceId> devices;
  std::vector<CpuCollectives::DeviceRank> ranks;
  devices.reserve(num_ranks);
  ranks.reserve(num_ranks);
  for (size_t r = 0; r < num_ranks; ++r) {
    devices.push_back(GlobalDeviceId(static_cast<int64_t>(r)));
    ranks.emplace_back(nullptr, RankId(static_cast<int64_t>(r)));
  }
  CpuCliqueKey key(devices);
  auto comms = collectives.CreateCommunicators(key, std::nullopt, ranks,
                                               CpuCollectives::Config());
  CHECK_OK(comms.status());
  return *std::move(comms);
}

RendezvousKey MakeKey(size_t num_ranks, int64_t op_id = 0) {
  std::vector<GlobalDeviceId> devices;
  devices.reserve(num_ranks);
  for (size_t r = 0; r < num_ranks; ++r) {
    devices.push_back(GlobalDeviceId(static_cast<int64_t>(r)));
  }
  return RendezvousKey(RunId(NextRunId()), std::move(devices),
                       static_cast<int>(num_ranks),
                       RendezvousKey::kCrossReplica, op_id);
}

template <typename T>
se::DeviceAddressBase Addr(T* ptr, size_t count) {
  return se::DeviceAddressBase(ptr, count * sizeof(T));
}

template <typename Fn>
std::vector<absl::Status> RunAcrossRanks(size_t num_ranks, Fn fn) {
  auto comms = MakeCommunicators(num_ranks);
  RendezvousKey key = MakeKey(num_ranks);
  std::vector<absl::Status> statuses(num_ranks, absl::OkStatus());
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "in_process_comm_test",
                                 static_cast<int>(num_ranks));
    for (size_t r = 0; r < num_ranks; ++r) {
      pool.Schedule([&, r]() {
        CpuCollectives::Executor exec(key, kTimeout);
        statuses[r] = fn(r, *comms[r], exec);
      });
    }
  }
  return statuses;
}

template <typename T>
std::vector<T> AllReduceSync(PrimitiveType dtype, ReductionKind kind,
                             const std::vector<std::vector<T>>& inputs) {
  const size_t n = inputs.size();
  const size_t count = inputs[0].size();
  std::vector<std::vector<T>> outputs(n, std::vector<T>(count));
  auto statuses = RunAcrossRanks(n, [&](size_t r, Communicator& comm,
                                        const CpuCollectives::Executor& exec) {
    return comm
        .AllReduce(Addr(const_cast<T*>(inputs[r].data()), count),
                   Addr(outputs[r].data(), count), dtype, count, kind, exec)
        .Await();
  });
  for (const absl::Status& s : statuses) {
    CHECK_OK(s);
  }
  for (size_t r = 1; r < n; ++r) {
    for (size_t i = 0; i < count; ++i) {
      using Bits =
          typename Eigen::numext::get_integer_by_size<sizeof(T)>::unsigned_type;
      EXPECT_EQ(absl::bit_cast<Bits>(outputs[r][i]),
                absl::bit_cast<Bits>(outputs[0][i]));
    }
  }
  return outputs[0];
}

template <typename T>
std::vector<T> ReduceScatterSync(PrimitiveType dtype, ReductionKind kind,
                                 const std::vector<std::vector<T>>& inputs) {
  const size_t n = inputs.size();
  const size_t count = inputs[0].size() / n;
  std::vector<std::vector<T>> outputs(n, std::vector<T>(count));
  auto statuses = RunAcrossRanks(n, [&](size_t r, Communicator& comm,
                                        const CpuCollectives::Executor& exec) {
    return comm
        .ReduceScatter(Addr(const_cast<T*>(inputs[r].data()), n * count),
                       Addr(outputs[r].data(), count), dtype, count, kind, exec)
        .Await();
  });
  for (const absl::Status& s : statuses) {
    CHECK_OK(s);
  }
  std::vector<T> flat;
  flat.reserve(n * count);
  for (size_t r = 0; r < n; ++r) {
    flat.insert(flat.end(), outputs[r].begin(), outputs[r].end());
  }
  return flat;
}

template <typename T>
void CheckFloatMinMaxNanAndZero(PrimitiveType dtype) {
  using Bits =
      typename Eigen::numext::get_integer_by_size<sizeof(T)>::unsigned_type;
  const T nan = std::numeric_limits<T>::quiet_NaN();
  const T one = static_cast<T>(1.0f);
  const T two = static_cast<T>(2.0f);
  const T lowest = std::numeric_limits<T>::lowest();
  const T highest = std::numeric_limits<T>::max();

  // Single-rank and multi-rank NaN propagation in rank 0 and last rank.
  for (ReductionKind kind : {ReductionKind::MIN, ReductionKind::MAX}) {
    auto single = AllReduceSync<T>(dtype, kind, {{nan}});
    EXPECT_TRUE(Eigen::numext::isnan(single[0])) << PrimitiveType_Name(dtype);

    auto multi = AllReduceSync<T>(
        dtype, kind, {{nan, one, lowest, highest}, {one, nan, nan, nan}});
    for (size_t i = 0; i < multi.size(); ++i) {
      EXPECT_TRUE(Eigen::numext::isnan(multi[i]))
          << PrimitiveType_Name(dtype) << " i=" << i;
    }

    auto rs = ReduceScatterSync<T>(dtype, kind, {{nan, one}, {two, nan}});
    EXPECT_TRUE(Eigen::numext::isnan(rs[0])) << PrimitiveType_Name(dtype);
    EXPECT_TRUE(Eigen::numext::isnan(rs[1])) << PrimitiveType_Name(dtype);
  }

  // Signed-zero ordering: MIN(-0, +0) == -0 and MAX(-0, +0) == +0 regardless of
  // which rank holds -0.
  if constexpr (has_negative_zero_v<T>) {
    constexpr Bits kSignBit = Bits{1} << (sizeof(Bits) * 8 - 1);
    const T pos_zero = absl::bit_cast<T>(Bits{0});
    const T neg_zero = absl::bit_cast<T>(kSignBit);

    auto mins = AllReduceSync<T>(dtype, ReductionKind::MIN,
                                 {{neg_zero, pos_zero}, {pos_zero, neg_zero}});
    EXPECT_EQ(absl::bit_cast<Bits>(mins[0]), kSignBit)
        << PrimitiveType_Name(dtype);
    EXPECT_EQ(absl::bit_cast<Bits>(mins[1]), kSignBit)
        << PrimitiveType_Name(dtype);

    auto maxs = AllReduceSync<T>(dtype, ReductionKind::MAX,
                                 {{neg_zero, pos_zero}, {pos_zero, neg_zero}});
    EXPECT_EQ(absl::bit_cast<Bits>(maxs[0]), Bits{0})
        << PrimitiveType_Name(dtype);
    EXPECT_EQ(absl::bit_cast<Bits>(maxs[1]), Bits{0})
        << PrimitiveType_Name(dtype);

    auto rs_max =
        ReduceScatterSync<T>(dtype, ReductionKind::MAX,
                             {{neg_zero, pos_zero}, {pos_zero, neg_zero}});
    EXPECT_EQ(absl::bit_cast<Bits>(rs_max[0]), Bits{0});
    EXPECT_EQ(absl::bit_cast<Bits>(rs_max[1]), Bits{0});
  }

  // Signaling NaN quieting on IEEE binary formats.
  if constexpr (std::numeric_limits<T>::has_infinity &&
                std::numeric_limits<T>::has_signaling_NaN) {
    constexpr int kMantissaBits = std::numeric_limits<T>::digits - 1;
    constexpr Bits kSignBit = Bits{1} << (sizeof(Bits) * 8 - 1);
    constexpr Bits kAbsMask = static_cast<Bits>(kSignBit - 1);
    constexpr Bits kMantissaMask =
        static_cast<Bits>((Bits{1} << kMantissaBits) - 1);
    constexpr Bits kInfBits = static_cast<Bits>(kAbsMask ^ kMantissaMask);
    constexpr Bits kQuietBit =
        static_cast<Bits>(Bits{1} << (kMantissaBits - 1));
    const T snan = absl::bit_cast<T>(static_cast<Bits>(kInfBits | Bits{1}));
    for (ReductionKind kind : {ReductionKind::MIN, ReductionKind::MAX}) {
      auto r_single = AllReduceSync<T>(dtype, kind, {{snan}});
      EXPECT_EQ(absl::bit_cast<Bits>(r_single[0]) & kQuietBit, kQuietBit)
          << PrimitiveType_Name(dtype);
      auto r_pair = AllReduceSync<T>(dtype, kind, {{snan, one}, {one, snan}});
      EXPECT_EQ(absl::bit_cast<Bits>(r_pair[0]) & kQuietBit, kQuietBit)
          << PrimitiveType_Name(dtype);
      EXPECT_EQ(absl::bit_cast<Bits>(r_pair[1]) & kQuietBit, kQuietBit)
          << PrimitiveType_Name(dtype);
    }
  }
}

TEST(InProcessCommunicatorTest, FloatMinMaxNanSignedZeroAndSignalingNan) {
  CheckFloatMinMaxNanAndZero<float>(F32);
  CheckFloatMinMaxNanAndZero<double>(F64);
  CheckFloatMinMaxNanAndZero<half>(F16);
  CheckFloatMinMaxNanAndZero<bfloat16>(BF16);
  CheckFloatMinMaxNanAndZero<tsl::float8_e5m2>(F8E5M2);
  CheckFloatMinMaxNanAndZero<tsl::float8_e4m3>(F8E4M3);
  CheckFloatMinMaxNanAndZero<tsl::float8_e3m4>(F8E3M4);
  CheckFloatMinMaxNanAndZero<tsl::float8_e4m3fn>(F8E4M3FN);
  CheckFloatMinMaxNanAndZero<tsl::float8_e4m3fnuz>(F8E4M3FNUZ);
  CheckFloatMinMaxNanAndZero<tsl::float8_e4m3b11fnuz>(F8E4M3B11FNUZ);
  CheckFloatMinMaxNanAndZero<tsl::float8_e5m2fnuz>(F8E5M2FNUZ);
  CheckFloatMinMaxNanAndZero<tsl::float8_e8m0fnu>(F8E8M0FNU);

  // Also check negative NaN (0xFF) on F8E4M3FN.
  const auto neg_nan = absl::bit_cast<tsl::float8_e4m3fn>(uint8_t{0xFF});
  const auto pos_one = static_cast<tsl::float8_e4m3fn>(1.0f);
  auto fn_out = AllReduceSync<tsl::float8_e4m3fn>(
      F8E4M3FN, ReductionKind::MAX, {{pos_one, neg_nan}, {neg_nan, pos_one}});
  EXPECT_EQ(absl::bit_cast<uint8_t>(fn_out[0]), uint8_t{0xFF});
  EXPECT_EQ(absl::bit_cast<uint8_t>(fn_out[1]), uint8_t{0xFF});

  // Unpacked 6-bit FN types (1 byte per element, sign bit 0x20, no NaN) obey
  // signed-zero MIN/MAX ordering as well.
  const auto f6_pos_zero = absl::bit_cast<tsl::float6_e2m3fn>(uint8_t{0x00});
  const auto f6_neg_zero = absl::bit_cast<tsl::float6_e2m3fn>(uint8_t{0x20});
  auto f6_max = AllReduceSync<tsl::float6_e2m3fn>(
      F6E2M3FN, ReductionKind::MAX,
      {{f6_neg_zero, f6_pos_zero}, {f6_pos_zero, f6_neg_zero}});
  EXPECT_EQ(absl::bit_cast<uint8_t>(f6_max[0]), uint8_t{0x00});
  EXPECT_EQ(absl::bit_cast<uint8_t>(f6_max[1]), uint8_t{0x00});
  auto f6_min = AllReduceSync<tsl::float6_e2m3fn>(
      F6E2M3FN, ReductionKind::MIN,
      {{f6_neg_zero, f6_pos_zero}, {f6_pos_zero, f6_neg_zero}});
  EXPECT_EQ(absl::bit_cast<uint8_t>(f6_min[0]), uint8_t{0x20});
  EXPECT_EQ(absl::bit_cast<uint8_t>(f6_min[1]), uint8_t{0x20});
}

#if defined(__x86_64__) || defined(_M_X64)
class ScopedFtzDaz {
 public:
  ScopedFtzDaz() : saved_csr_(_mm_getcsr()) {
    // Enable MXCSR DAZ (bit 6, 0x0040) and FTZ (bit 15, 0x8000).
    constexpr unsigned int kDazAndFtz = 0x8040U;
    _mm_setcsr(saved_csr_ | kDazAndFtz);
  }
  ~ScopedFtzDaz() { _mm_setcsr(saved_csr_); }

 private:
  unsigned int saved_csr_;
};

TEST(InProcessCommunicatorTest, SubnormalMinMaxPreservedUnderFtzDaz) {
  const float pos_sub = absl::bit_cast<float>(uint32_t{0x00000001});
  const float pos_sub2 = absl::bit_cast<float>(uint32_t{0x00000002});
  const float neg_sub = absl::bit_cast<float>(uint32_t{0x80000001});
  const float pos_zero = absl::bit_cast<float>(uint32_t{0x00000000});

  auto comms = MakeCommunicators(2);
  RendezvousKey key_min = MakeKey(2, 1);
  RendezvousKey key_max = MakeKey(2, 2);
  std::vector<std::vector<float>> in = {
      {pos_sub, pos_zero, neg_sub},
      {pos_sub2, pos_sub, pos_zero},
  };
  std::vector<std::vector<float>> out_min(2, std::vector<float>(3));
  std::vector<std::vector<float>> out_max(2, std::vector<float>(3));
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "ftz_daz", 2);
    for (size_t r = 0; r < 2; ++r) {
      pool.Schedule([&, r]() {
        ScopedFtzDaz guard;
        CpuCollectives::Executor e1(key_min, kTimeout);
        CpuCollectives::Executor e2(key_max, kTimeout);
        ASSERT_OK(comms[r]
                      ->AllReduce(Addr(in[r].data(), 3),
                                  Addr(out_min[r].data(), 3), F32, 3,
                                  ReductionKind::MIN, e1)
                      .Await());
        ASSERT_OK(comms[r]
                      ->AllReduce(Addr(in[r].data(), 3),
                                  Addr(out_max[r].data(), 3), F32, 3,
                                  ReductionKind::MAX, e2)
                      .Await());
      });
    }
  }
  EXPECT_EQ(absl::bit_cast<uint32_t>(out_min[0][0]), uint32_t{0x00000001});
  EXPECT_EQ(absl::bit_cast<uint32_t>(out_min[0][1]), uint32_t{0x00000000});
  EXPECT_EQ(absl::bit_cast<uint32_t>(out_min[0][2]), uint32_t{0x80000001});
  EXPECT_EQ(absl::bit_cast<uint32_t>(out_max[0][0]), uint32_t{0x00000002});
  EXPECT_EQ(absl::bit_cast<uint32_t>(out_max[0][1]), uint32_t{0x00000001});
  EXPECT_EQ(absl::bit_cast<uint32_t>(out_max[0][2]), uint32_t{0x00000000});
}
#endif

TEST(InProcessCommunicatorTest, NoSyntheticIdentityCorruption) {
  // F8E8M0FNU has no zero; static_cast<float8_e8m0fnu>(0) is NaN (0xFF).
  const auto e8_one = static_cast<tsl::float8_e8m0fnu>(1.0f);
  const auto e8_two = static_cast<tsl::float8_e8m0fnu>(2.0f);
  const auto e8_four = static_cast<tsl::float8_e8m0fnu>(4.0f);
  auto e8_sum = AllReduceSync<tsl::float8_e8m0fnu>(
      F8E8M0FNU, ReductionKind::SUM, {{e8_one, e8_two}, {e8_one, e8_two}});
  EXPECT_EQ(static_cast<float>(e8_sum[0]), 2.0f);
  EXPECT_EQ(static_cast<float>(e8_sum[1]), 4.0f);

  auto e8_prod = AllReduceSync<tsl::float8_e8m0fnu>(
      F8E8M0FNU, ReductionKind::PRODUCT, {{e8_two}, {e8_four}});
  EXPECT_EQ(static_cast<float>(e8_prod[0]), 8.0f);

  auto e8_min = AllReduceSync<tsl::float8_e8m0fnu>(
      F8E8M0FNU, ReductionKind::MIN, {{e8_four}, {e8_one}});
  EXPECT_EQ(static_cast<float>(e8_min[0]), 1.0f);

  // SUM([-0.0, -0.0]) preserves -0.0 per IEEE 754 (no leading +0.0 + (-0.0)).
  const float neg_zero = absl::bit_cast<float>(uint32_t{0x80000000});
  auto sum_nz =
      AllReduceSync<float>(F32, ReductionKind::SUM, {{neg_zero}, {neg_zero}});
  EXPECT_EQ(absl::bit_cast<uint32_t>(sum_nz[0]), uint32_t{0x80000000});

  auto single_nz = AllReduceSync<float>(F32, ReductionKind::SUM, {{neg_zero}});
  EXPECT_EQ(absl::bit_cast<uint32_t>(single_nz[0]), uint32_t{0x80000000});

  // Single-rank C64 PRODUCT([(inf, 1)]) must not multiply by synthetic (1, 0),
  // which turns the imaginary component into 0 * inf = NaN.
  const complex64 inf_one(std::numeric_limits<float>::infinity(), 1.0f);
  auto cprod =
      AllReduceSync<complex64>(C64, ReductionKind::PRODUCT, {{inf_one}});
  EXPECT_TRUE(std::isinf(cprod[0].real()));
  EXPECT_EQ(cprod[0].imag(), 1.0f);
}

TEST(InProcessCommunicatorTest, RejectsInvalidAndSubByteTypesAndComplexMinMax) {
  auto comms = MakeCommunicators(1);
  RendezvousKey key = MakeKey(1);
  CpuCollectives::Executor exec(key, kTimeout);
  uint64_t buf_in = 0;
  uint64_t buf_out = 0;
  se::DeviceAddressBase in(&buf_in, sizeof(buf_in));
  se::DeviceAddressBase out(&buf_out, sizeof(buf_out));

  const PrimitiveType unsupported[] = {
      PRIMITIVE_TYPE_INVALID,
      TUPLE,
      TOKEN,
      OPAQUE_TYPE,
      S1,
      U1,
      S2,
      U2,
      S4,
      U4,
      F4E2M1FN,
  };
  for (PrimitiveType dt : unsupported) {
    EXPECT_THAT(
        comms[0]->AllReduce(in, out, dt, 2, ReductionKind::SUM, exec).Await(),
        StatusIs(absl::StatusCode::kUnimplemented))
        << PrimitiveType_Name(dt);
    EXPECT_THAT(comms[0]
                    ->ReduceScatter(in, out, dt, 2, ReductionKind::SUM, exec)
                    .Await(),
                StatusIs(absl::StatusCode::kUnimplemented))
        << PrimitiveType_Name(dt);
    EXPECT_THAT(comms[0]->AllGather(in, out, dt, 2, exec).Await(),
                StatusIs(absl::StatusCode::kUnimplemented))
        << PrimitiveType_Name(dt);
    EXPECT_THAT(comms[0]->AllToAll({in}, {out}, dt, 2, exec).Await(),
                StatusIs(absl::StatusCode::kUnimplemented))
        << PrimitiveType_Name(dt);
    EXPECT_THAT(
        comms[0]
            ->CollectivePermute(in, out, dt, 2, RankId(0), {RankId(0)}, exec)
            .Await(),
        StatusIs(absl::StatusCode::kUnimplemented))
        << PrimitiveType_Name(dt);
  }

  // Complex MIN/MAX rejected consistently on every rank, even when count == 0
  // or count < num_ranks (where tail ranks receive an empty chunk).
  for (PrimitiveType dt : {C64, C128}) {
    for (ReductionKind kind : {ReductionKind::MIN, ReductionKind::MAX}) {
      for (size_t count : {size_t{0}, size_t{1}}) {
        auto st = RunAcrossRanks(
            3, [&](size_t, Communicator& c, const CpuCollectives::Executor& e) {
              complex128 v(1.0, 2.0);
              se::DeviceAddressBase a(&v, sizeof(v));
              return c.AllReduce(a, a, dt, count, kind, e).Await();
            });
        for (const absl::Status& s : st) {
          EXPECT_THAT(s, StatusIs(absl::StatusCode::kInvalidArgument));
        }
      }
    }
  }

  // PRED (1 byte per element) remains supported, including staged reductions.
  bool p_in[2] = {true, false};
  bool p_out[2] = {false, false};
  ASSERT_OK(comms[0]
                ->AllReduce(Addr(p_in, 2), Addr(p_out, 2), PRED, 2,
                            ReductionKind::SUM, exec)
                .Await());
  EXPECT_TRUE(p_out[0]);
  EXPECT_FALSE(p_out[1]);

  bool pr0[2] = {false, true};
  bool pr1[2] = {true, false};
  auto pst = RunAcrossRanks(2, [&](size_t r, Communicator& c,
                                   const CpuCollectives::Executor& e) {
    bool* b = r == 0 ? pr0 : pr1;
    return c
        .ReduceScatter(Addr(b, 2), Addr(b, 1), PRED, 1, ReductionKind::MAX, e)
        .Await();
  });
  ASSERT_OK(pst[0]);
  ASSERT_OK(pst[1]);
  EXPECT_TRUE(pr0[0]);
  EXPECT_TRUE(pr1[0]);
}

TEST(InProcessCommunicatorTest, AllReduceInPlaceShiftedCrossRankAndZeroCount) {
  // Rank 1 in-place while rank 0 is out-of-place, plus count < num_ranks with
  // cross-rank aliasing (ensures empty-chunk tail ranks hit staging barrier).
  {
    std::vector<int32_t> s0 = {10, 20, 30, 40, 50};
    std::vector<int32_t> d0(5, 0);
    std::vector<int32_t> sd1 = {1, 2, 3, 4, 5};
    std::vector<int32_t> sd2 = {100, 200, 300, 400, 500};
    auto st = RunAcrossRanks(
        3, [&](size_t r, Communicator& c, const CpuCollectives::Executor& e) {
          se::DeviceAddressBase src =
              r == 0 ? Addr(s0.data(), 5)
                     : (r == 1 ? Addr(sd1.data(), 5) : Addr(sd2.data(), 5));
          se::DeviceAddressBase dst =
              r == 0 ? Addr(d0.data(), 5)
                     : (r == 1 ? Addr(sd1.data(), 5) : Addr(sd2.data(), 5));
          return c.AllReduce(src, dst, S32, 5, ReductionKind::SUM, e).Await();
        });
    for (const absl::Status& s : st) {
      ASSERT_OK(s);
    }
    const std::vector<int32_t> expected = {111, 222, 333, 444, 555};
    EXPECT_EQ(d0, expected);
    EXPECT_EQ(sd1, expected);
    EXPECT_EQ(sd2, expected);
  }

  // Partial shifted self-overlap and cross-rank dest0 == src1 swap with
  // count = 2 < num_ranks = 3 (rank 2 has chunk_count == 0).
  {
    std::vector<int32_t> buf0 = {10, 20, 99};  // src=[0..2), dest=[1..3)
    std::vector<int32_t> buf1 = {1, 2};
    std::vector<int32_t> buf2 = {100, 200};
    std::vector<int32_t> out2(2, 0);
    auto st = RunAcrossRanks(
        3, [&](size_t r, Communicator& c, const CpuCollectives::Executor& e) {
          if (r == 0) {
            return c
                .AllReduce(Addr(buf0.data(), 2), Addr(buf0.data() + 1, 2), S32,
                           2, ReductionKind::SUM, e)
                .Await();
          }
          if (r == 1) {
            return c
                .AllReduce(Addr(buf1.data(), 2), Addr(buf2.data(), 2), S32, 2,
                           ReductionKind::SUM, e)
                .Await();
          }
          return c
              .AllReduce(Addr(buf2.data(), 2), Addr(out2.data(), 2), S32, 2,
                         ReductionKind::SUM, e)
              .Await();
        });
    for (const absl::Status& s : st) {
      ASSERT_OK(s);
    }
    EXPECT_EQ(buf0[1], 111);
    EXPECT_EQ(buf0[2], 222);
    EXPECT_EQ(buf2, (std::vector<int32_t>{111, 222}));
    EXPECT_EQ(out2, (std::vector<int32_t>{111, 222}));
  }

  // Zero-element AllReduce with null pointers.
  auto z = RunAcrossRanks(
      2, [&](size_t, Communicator& c, const CpuCollectives::Executor& e) {
        return c
            .AllReduce(se::DeviceAddressBase(), se::DeviceAddressBase(), F32, 0,
                       ReductionKind::SUM, e)
            .Await();
      });
  ASSERT_OK(z[0]);
  ASSERT_OK(z[1]);
}

TEST(InProcessCommunicatorTest, ReduceScatterInPlaceAndOverlapping) {
  // Both base-in-place (dest == src) and own-shard-in-place (dest == src + r*k)
  // across 3 ranks, repeated to catch races under TSan.
  for (bool own_shard : {false, true}) {
    for (int iter = 0; iter < 16; ++iter) {
      std::vector<std::vector<int32_t>> bufs = {
          {1, 2, 3, 4, 5, 6},
          {10, 20, 30, 40, 50, 60},
          {100, 200, 300, 400, 500, 600},
      };
      auto st = RunAcrossRanks(
          3, [&](size_t r, Communicator& c, const CpuCollectives::Executor& e) {
            int32_t* dst = own_shard ? bufs[r].data() + r * 2 : bufs[r].data();
            return c
                .ReduceScatter(Addr(bufs[r].data(), 6), Addr(dst, 2), S32, 2,
                               ReductionKind::SUM, e)
                .Await();
          });
      for (const absl::Status& s : st) {
        ASSERT_OK(s);
      }
      const size_t o0 = 0;
      const size_t o1 = own_shard ? 2 : 0;
      const size_t o2 = own_shard ? 4 : 0;
      EXPECT_EQ((std::vector<int32_t>{bufs[0][o0], bufs[0][o0 + 1]}),
                (std::vector<int32_t>{111, 222}));
      EXPECT_EQ((std::vector<int32_t>{bufs[1][o1], bufs[1][o1 + 1]}),
                (std::vector<int32_t>{333, 444}));
      EXPECT_EQ((std::vector<int32_t>{bufs[2][o2], bufs[2][o2 + 1]}),
                (std::vector<int32_t>{555, 666}));
    }
  }
}

TEST(InProcessCommunicatorTest, AllGatherInPlaceShiftedAndZeroCount) {
  // Base-in-place (src == dest for all ranks), own-slot in-place
  // (src == dest + rank * count), and shifted partial overlap.
  for (int mode = 0; mode < 3; ++mode) {
    for (int iter = 0; iter < 16; ++iter) {
      std::vector<std::vector<int32_t>> bufs(3, std::vector<int32_t>(6, -1));
      for (size_t r = 0; r < 3; ++r) {
        const size_t off = mode == 0 ? 0 : (mode == 1 ? r * 2 : 1);
        bufs[r][off] = static_cast<int32_t>(r * 10 + 1);
        bufs[r][off + 1] = static_cast<int32_t>(r * 10 + 2);
      }
      auto st = RunAcrossRanks(
          3, [&](size_t r, Communicator& c, const CpuCollectives::Executor& e) {
            const size_t off = mode == 0 ? 0 : (mode == 1 ? r * 2 : 1);
            return c
                .AllGather(Addr(bufs[r].data() + off, 2),
                           Addr(bufs[r].data(), 6), S32, 2, e)
                .Await();
          });
      for (const absl::Status& s : st) {
        ASSERT_OK(s);
      }
      const std::vector<int32_t> expected = {1, 2, 11, 12, 21, 22};
      for (size_t r = 0; r < 3; ++r) {
        EXPECT_EQ(bufs[r], expected) << "mode=" << mode << " rank=" << r;
      }
    }
  }

  auto z = RunAcrossRanks(2, [&](size_t, Communicator& c,
                                 const CpuCollectives::Executor& e) {
    return c
        .AllGather(se::DeviceAddressBase(), se::DeviceAddressBase(), S32, 0, e)
        .Await();
  });
  ASSERT_OK(z[0]);
  ASSERT_OK(z[1]);
}

TEST(InProcessCommunicatorTest, AllToAllInPlaceEmptyAndMismatchedSize) {
  constexpr size_t kRanks = 4;
  constexpr size_t kCount = 3;
  for (int iter = 0; iter < 16; ++iter) {
    std::vector<std::vector<int32_t>> bufs(
        kRanks, std::vector<int32_t>(kRanks * kCount));
    for (size_t r = 0; r < kRanks; ++r) {
      for (size_t dst = 0; dst < kRanks; ++dst) {
        for (size_t i = 0; i < kCount; ++i) {
          bufs[r][dst * kCount + i] =
              static_cast<int32_t>(r * 100 + dst * 10 + i);
        }
      }
    }
    auto st = RunAcrossRanks(kRanks, [&](size_t r, Communicator& c,
                                         const CpuCollectives::Executor& e) {
      absl::InlinedVector<se::DeviceAddressBase, 4> slices;
      slices.reserve(kRanks);
      for (size_t k = 0; k < kRanks; ++k) {
        slices.push_back(Addr(bufs[r].data() + k * kCount, kCount));
      }
      return c.AllToAll(slices, slices, S32, kCount, e).Await();
    });
    for (const absl::Status& s : st) {
      ASSERT_OK(s);
    }
    for (size_t r = 0; r < kRanks; ++r) {
      for (size_t src = 0; src < kRanks; ++src) {
        for (size_t i = 0; i < kCount; ++i) {
          EXPECT_EQ(bufs[r][src * kCount + i],
                    static_cast<int32_t>(src * 100 + r * 10 + i));
        }
      }
    }
  }

  // Mismatched buffer list size returns InvalidArgument cleanly.
  auto bad = RunAcrossRanks(
      2, [&](size_t, Communicator& c, const CpuCollectives::Executor& e) {
        int32_t x = 1;
        return c.AllToAll({Addr(&x, 1)}, {Addr(&x, 1)}, S32, 1, e).Await();
      });
  EXPECT_THAT(bad[0], StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(bad[1], StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(InProcessCommunicatorTest,
     CollectivePermuteInPlaceSwapShiftSelfAndInvalidRank) {
  // In-place 3-cycle: 0 -> 1 -> 2 -> 0.
  for (int iter = 0; iter < 16; ++iter) {
    std::vector<std::vector<int32_t>> bufs = {{1, 2}, {10, 20}, {100, 200}};
    auto st = RunAcrossRanks(
        3, [&](size_t r, Communicator& c, const CpuCollectives::Executor& e) {
          const RankId src(static_cast<int64_t>((r + 2) % 3));
          const RankId dst(static_cast<int64_t>((r + 1) % 3));
          return c
              .CollectivePermute(Addr(bufs[r].data(), 2),
                                 Addr(bufs[r].data(), 2), S32, 2, src, {dst}, e)
              .Await();
        });
    for (const absl::Status& s : st) {
      ASSERT_OK(s);
    }
    EXPECT_EQ(bufs[0], (std::vector<int32_t>{100, 200}));
    EXPECT_EQ(bufs[1], (std::vector<int32_t>{1, 2}));
    EXPECT_EQ(bufs[2], (std::vector<int32_t>{10, 20}));
  }

  // In-place one-way shift 0 -> 1: rank 0 zeroes dest0 (== src0) while rank 1
  // receives from src0, and rank 2 performs an exact in-place self-copy.
  for (int iter = 0; iter < 16; ++iter) {
    std::vector<std::vector<int32_t>> bufs = {{7, 8}, {9, 10}, {42, 43}};
    auto st = RunAcrossRanks(3, [&](size_t r, Communicator& c,
                                    const CpuCollectives::Executor& e) {
      std::optional<RankId> src;
      std::vector<RankId> targets;
      if (r == 0) {
        targets = {RankId(1)};
      } else if (r == 1) {
        src = RankId(0);
      } else {
        src = RankId(2);
        targets = {RankId(2)};
      }
      return c
          .CollectivePermute(Addr(bufs[r].data(), 2), Addr(bufs[r].data(), 2),
                             S32, 2, src, targets, e)
          .Await();
    });
    for (const absl::Status& s : st) {
      ASSERT_OK(s);
    }
    EXPECT_EQ(bufs[0], (std::vector<int32_t>{0, 0}));
    EXPECT_EQ(bufs[1], (std::vector<int32_t>{7, 8}));
    EXPECT_EQ(bufs[2], (std::vector<int32_t>{42, 43}));
  }

  // Out-of-range source_rank returns InvalidArgument without throwing.
  for (int64_t bad_rank : {int64_t{-1}, int64_t{2}}) {
    auto st = RunAcrossRanks(2, [&](size_t r, Communicator& c,
                                    const CpuCollectives::Executor& e) {
      int32_t v = 5;
      std::optional<RankId> src =
          r == 0 ? std::optional<RankId>(RankId(bad_rank)) : std::nullopt;
      return c.CollectivePermute(Addr(&v, 1), Addr(&v, 1), S32, 1, src, {}, e)
          .Await();
    });
    EXPECT_THAT(st[0], StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_THAT(st[1], StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(InProcessCommunicatorTest, SingleReplicaAllReduceThunkAliasedAndEmpty) {
  std::vector<float> aliased = {1.5f, -2.5f, 3.25f};
  std::vector<float> src = {4.0f, 5.0f};
  std::vector<float> dst = {0.0f, 0.0f};

  BufferAllocation alloc_aliased(/*index=*/0, aliased.size() * sizeof(float),
                                 /*color=*/0);
  BufferAllocation alloc_src(/*index=*/1, src.size() * sizeof(float),
                             /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/2, dst.size() * sizeof(float),
                             /*color=*/0);
  BufferAllocation alloc_empty(/*index=*/3, /*size=*/0, /*color=*/0);

  BufferAllocation::Slice slice_aliased(&alloc_aliased, 0,
                                        aliased.size() * sizeof(float));
  BufferAllocation::Slice slice_src(&alloc_src, 0, src.size() * sizeof(float));
  BufferAllocation::Slice slice_dst(&alloc_dst, 0, dst.size() * sizeof(float));
  BufferAllocation::Slice slice_empty(&alloc_empty, 0, 0);

  Shape shape3 = ShapeUtil::MakeShape(F32, {3});
  Shape shape2 = ShapeUtil::MakeShape(F32, {2});
  Shape shape0 = ShapeUtil::MakeShape(F32, {0});

  CollectiveThunk::OpParams op_params;
  op_params.op_id = 0;
  op_params.has_channel_id = false;

  CollectiveThunk::OpBuffers op_buffers;
  op_buffers.source_buffers = {slice_aliased, slice_src, slice_empty};
  op_buffers.source_shapes = {shape3, shape2, shape0};
  op_buffers.destination_buffers = {slice_aliased, slice_dst, slice_empty};
  op_buffers.destination_shapes = {shape3, shape2, shape0};

  ASSERT_OK_AND_ASSIGN(auto thunk,
                       AllReduceThunk::Create(
                           {"single-replica-ar"}, ReductionKind::SUM, op_params,
                           op_buffers, CollectiveThunk::OpResources{},
                           /*single_replica=*/true));

  InProcessCommunicator direct_comm(/*rank=*/0, /*num_ranks=*/1);
  EXPECT_EQ(direct_comm.ToString(),
            "InProcessCommunicator [rank: 0 num_ranks: 1]");

  BufferAllocations::Buffers raw_buffers = {
      Addr(aliased.data(), aliased.size()), Addr(src.data(), src.size()),
      Addr(dst.data(), dst.size()), se::DeviceAddressBase(nullptr, 0)};
  BufferAllocations allocations(raw_buffers);
  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto event = thunk->Execute(params);
  tsl::BlockUntilReady(event);
  EXPECT_FALSE(event.IsError());
  EXPECT_EQ(aliased, (std::vector<float>{1.5f, -2.5f, 3.25f}));
  EXPECT_EQ(dst, (std::vector<float>{4.0f, 5.0f}));
}

}  // namespace
}  // namespace xla::cpu
