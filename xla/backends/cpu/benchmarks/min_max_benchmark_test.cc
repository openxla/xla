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

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

// Elementwise maximum(x, 0) for F32
static void BM_ElementwiseMaxF32(benchmark::State& state,
                                 HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule max_f32_$d0

    ENTRY e {
      p0 = f32[1,2,1,$d0,256] parameter(0)
      c0 = f32[] constant(0)
      b0 = f32[1,2,1,$d0,256] broadcast(c0), dimensions={}
      ROOT max = f32[1,2,1,$d0,256] maximum(p0, b0)
    }
  )";

  std::minstd_rand0 engine;
  auto shape = ShapeUtil::MakeShape(F32, {1, 2, 1, d0, 256});
  ASSERT_OK_AND_ASSIGN(
      Literal p0, LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f));

  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

// Elementwise maximum(x, 0) for BF16
static void BM_ElementwiseMaxBF16(benchmark::State& state,
                                  HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule max_bf16_$d0

    ENTRY e {
      p0 = bf16[1,2,1,$d0,256] parameter(0)
      c0 = bf16[] constant(0)
      b0 = bf16[1,2,1,$d0,256] broadcast(c0), dimensions={}
      ROOT max = bf16[1,2,1,$d0,256] maximum(p0, b0)
    }
  )";

  std::minstd_rand0 engine;
  auto shape = ShapeUtil::MakeShape(BF16, {1, 2, 1, d0, 256});
  ASSERT_OK_AND_ASSIGN(
      Literal p0, LiteralUtil::CreateRandomLiteral<BF16>(shape, &engine, 1.0f, 0.1f));

  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

// Reduce-max over inner dimension for F32
static void BM_ReduceMaxInnerF32(benchmark::State& state,
                                 HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule reduce_max_inner_f32_$d0

    max_fn {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT max = f32[] maximum(p0, p1)
    }

    ENTRY e {
      p0 = f32[1,2,1,$d0,256] parameter(0)
      c0 = f32[] constant(-inf)
      ROOT reduce = f32[1,2,1,$d0] reduce(p0, c0), dimensions={4}, to_apply=max_fn
    }
  )";

  std::minstd_rand0 engine;
  auto shape = ShapeUtil::MakeShape(F32, {1, 2, 1, d0, 256});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

// Reduce-max over outer dimension for F32
static void BM_ReduceMaxOuterF32(benchmark::State& state,
                                 HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule reduce_max_outer_f32_$d0

    max_fn {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT max = f32[] maximum(p0, p1)
    }

    ENTRY e {
      p0 = f32[1,2,1,$d0,256] parameter(0)
      c0 = f32[] constant(-inf)
      ROOT reduce = f32[1,2,1,256] reduce(p0, c0), dimensions={3}, to_apply=max_fn
    }
  )";

  std::minstd_rand0 engine;
  auto shape = ShapeUtil::MakeShape(F32, {1, 2, 1, d0, 256});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

// Reduce-max over inner dimension for BF16
static void BM_ReduceMaxInnerBF16(benchmark::State& state,
                                  HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule reduce_max_inner_bf16_$d0

    max_fn {
      p0 = bf16[] parameter(0)
      p1 = bf16[] parameter(1)
      ROOT max = bf16[] maximum(p0, p1)
    }

    ENTRY e {
      p0 = bf16[1,2,1,$d0,256] parameter(0)
      c0 = bf16[] constant(-inf)
      ROOT reduce = bf16[1,2,1,$d0] reduce(p0, c0), dimensions={4}, to_apply=max_fn
    }
  )";

  std::minstd_rand0 engine;
  auto shape = ShapeUtil::MakeShape(BF16, {1, 2, 1, d0, 256});
  auto p0 = *LiteralUtil::CreateRandomLiteral<BF16>(shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

// Reduce-max over outer dimension for BF16
static void BM_ReduceMaxOuterBF16(benchmark::State& state,
                                  HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule reduce_max_outer_bf16_$d0

    max_fn {
      p0 = bf16[] parameter(0)
      p1 = bf16[] parameter(1)
      ROOT max = bf16[] maximum(p0, p1)
    }

    ENTRY e {
      p0 = bf16[1,2,1,$d0,256] parameter(0)
      c0 = bf16[] constant(-inf)
      ROOT reduce = bf16[1,2,1,256] reduce(p0, c0), dimensions={3}, to_apply=max_fn
    }
  )";

  std::minstd_rand0 engine;
  auto shape = ShapeUtil::MakeShape(BF16, {1, 2, 1, d0, 256});
  auto p0 = *LiteralUtil::CreateRandomLiteral<BF16>(shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

// Reduce-window max (maxpool) for F32
static void BM_ReduceWindowMaxF32(benchmark::State& state,
                                  HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);
  int64_t d1 = (d0 - 1) / 2;

  absl::string_view hlo = R"(
    HloModule reduce_window_max_f32_$d0

    max_fn {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT max = f32[] maximum(p0, p1)
    }

    ENTRY e {
      p0 = f32[$d0,$d0] parameter(0)
      c0 = f32[] constant(-inf)
      ROOT rw = f32[$d1,$d1] reduce-window(p0, c0),
           window={size=3x3 stride=2x2 pad=0_0x0_0}, to_apply=max_fn
    }
  )";

  std::minstd_rand0 engine;
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(
      ShapeUtil::MakeShape(F32, {d0, d0}), &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0};
  CHECK_OK(RunHloBenchmark(
      state, hlo, args, {{"$d0", absl::StrCat(d0)}, {"$d1", absl::StrCat(d1)}},
      options));
}

// Reduce-window max (maxpool) for BF16
static void BM_ReduceWindowMaxBF16(benchmark::State& state,
                                   HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);
  int64_t d1 = (d0 - 1) / 2;

  absl::string_view hlo = R"(
    HloModule reduce_window_max_bf16_$d0

    max_fn {
      p0 = bf16[] parameter(0)
      p1 = bf16[] parameter(1)
      ROOT max = bf16[] maximum(p0, p1)
    }

    ENTRY e {
      p0 = bf16[$d0,$d0] parameter(0)
      c0 = bf16[] constant(-inf)
      ROOT rw = bf16[$d1,$d1] reduce-window(p0, c0),
           window={size=3x3 stride=2x2 pad=0_0x0_0}, to_apply=max_fn
    }
  )";

  std::minstd_rand0 engine;
  auto p0 = *LiteralUtil::CreateRandomLiteral<BF16>(
      ShapeUtil::MakeShape(BF16, {d0, d0}), &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0};
  CHECK_OK(RunHloBenchmark(
      state, hlo, args, {{"$d0", absl::StrCat(d0)}, {"$d1", absl::StrCat(d1)}},
      options));
}

#define BENCHMARK_SIZES(NAME)   \
  XLA_CPU_BENCHMARK(NAME)       \
      ->MeasureProcessCPUTime() \
      ->Arg(128)                \
      ->Arg(256)                \
      ->Arg(512)                \
      ->Arg(1024)

BENCHMARK_SIZES(BM_ElementwiseMaxF32);
BENCHMARK_SIZES(BM_ElementwiseMaxBF16);
BENCHMARK_SIZES(BM_ReduceMaxInnerF32);
BENCHMARK_SIZES(BM_ReduceMaxOuterF32);
BENCHMARK_SIZES(BM_ReduceMaxInnerBF16);
BENCHMARK_SIZES(BM_ReduceMaxOuterBF16);

#define BENCHMARK_2D_SIZES(NAME) \
  XLA_CPU_BENCHMARK(NAME)        \
      ->MeasureProcessCPUTime()  \
      ->Arg(128)                 \
      ->Arg(256)                 \
      ->Arg(512)

BENCHMARK_2D_SIZES(BM_ReduceWindowMaxF32);
BENCHMARK_2D_SIZES(BM_ReduceWindowMaxBF16);

}  // namespace xla::cpu
