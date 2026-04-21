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

// Benchmarks for the CPU emit of `rsqrt`, supporting the performance
// discussion on openxla/xla#40844. The f64 variants are the point of
// interest: with the PR applied, f64 rsqrt lowers to llvm.sqrt + fdiv
// regardless of AVX support; without the PR, it takes the SIMD
// vrsqrt + Newton-Raphson refinement path when avx512f is available.
// The delta between those two emit strategies is what the
// BM_RsqrtF64Avx512 benchmark targets.

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

static void BM_RsqrtF32(benchmark::State& state, HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule rsqrt_f32_$d0

    ENTRY e {
      input = f32[$d0] parameter(0)
      ROOT output = rsqrt(input)
    }
  )";

  std::minstd_rand0 engine;

  auto input_shape = ShapeUtil::MakeShape(F32, {d0});
  auto p0 =
      *LiteralUtil::CreateRandomLiteral<F32>(input_shape, &engine, 2.0f, 0.5f);
  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

static void BM_RsqrtF64(benchmark::State& state, HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule rsqrt_f64_$d0

    ENTRY e {
      input = f64[$d0] parameter(0)
      ROOT output = rsqrt(input)
    }
  )";

  std::minstd_rand0 engine;

  auto input_shape = ShapeUtil::MakeShape(F64, {d0});
  auto p0 =
      *LiteralUtil::CreateRandomLiteral<F64>(input_shape, &engine, 2.0, 0.5);
  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

// Forces AVX512F + vector-width 512 so the f64 rsqrt emit actually hits
// the SIMD vrsqrt + Newton-Raphson path on pre-PR builds. With the PR
// applied, the intrinsic falls back to llvm.sqrt + fdiv even in this
// config — so the BM_RsqrtF64Avx512 number is the direct measurement
// of the emit-strategy change.
static void BM_RsqrtF64Avx512(benchmark::State& state) {
  int64_t d0 = state.range(0);
  HloBenchmarkOptions options;
  options.aot_options = std::make_unique<CpuAotCompilationOptions>(
      /*triple=*/"x86_64-unknown-linux-gnu", /*cpu_name=*/"skylake-avx512",
      /*features=*/"+avx,+avx512f,+avx512vl",
      /*entry_point_name=*/"rsqrt_f64",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static);
  options.aot_options->mutable_debug_options()->set_xla_cpu_prefer_vector_width(
      512);

  absl::string_view hlo = R"(
    HloModule rsqrt_f64_$d0

    ENTRY e {
      input = f64[$d0] parameter(0)
      ROOT output = rsqrt(input)
    }
  )";
  std::minstd_rand0 engine;

  auto input_shape = ShapeUtil::MakeShape(F64, {d0});
  auto p0 =
      *LiteralUtil::CreateRandomLiteral<F64>(input_shape, &engine, 2.0, 0.5);
  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

#define REGISTER_RSQRT_BENCHMARK(NAME) \
  XLA_CPU_BENCHMARK(NAME)              \
      ->MeasureProcessCPUTime()        \
      ->Arg(128)                       \
      ->Arg(256)                       \
      ->Arg(512)                       \
      ->Arg(1024)                      \
      ->Arg(4096);

REGISTER_RSQRT_BENCHMARK(BM_RsqrtF32);
REGISTER_RSQRT_BENCHMARK(BM_RsqrtF64);
BENCHMARK(BM_RsqrtF64Avx512)
    ->MeasureProcessCPUTime()
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(4096);

}  // namespace xla::cpu
