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

// ROCm implementation of the nvtx_utils.h interface using roctx.
//
// Dual push: RangePush/RangePop populate BOTH the AnnotationStack (Pipeline A
// -- annotations reach kernel events via correlation ID, producing the kTfOp
// and hlo_op stats) AND emit roctx markers via librocprofiler-sdk-roctx.so
// (Pipeline B -- named ranges captured by MarkerCallback in rocm_tracer.cc and
// rendered as "Host Threads/<tid>/ROCTX" timeline bands).
//
// Deliberate divergence from CUDA: nvtx_utils.cc does NOT push the
// AnnotationStack on the domain path, because there the NVTX ranges feed a
// separate consumer (Nsight) and losing kTfOp is acceptable. On ROCm both
// pipelines feed the SAME XSpace, so we need the stats and the bands together.
// Pushing both keeps the stacks balanced and costs one atomic read.
//
// The domain is opt-in via XLA_ROCM_ENABLE_ROCTX; see DefaultProfilerDomain.

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/base/const_init.h"
#include "absl/container/node_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "rocm/include/rocprofiler-sdk-roctx/roctx.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/profiler/lib/nvtx_utils.h"

namespace tsl::profiler {

ProfilerDomainHandle DefaultProfilerDomain() {
  // Opt-in, and LATCHED: computed exactly once for the process lifetime.
  //
  // Stability is a correctness requirement, not an optimisation.
  // scoped_annotation.h branches on this handle in BOTH overloads of
  // PushAnnotation and in PopAnnotation. A value that changed mid-process
  // would push
  // down one path and pop down the other, leaving the AnnotationStack and the
  // roctx range stack permanently unbalanced -- the same class of bug as
  // returning non-null while detail::RangePush was a no-op. So this must not
  // be derived from anything dynamic (profiler enabled/disabled, tracer state,
  // AnnotationStack::IsEnabled): only from process-lifetime configuration.
  //
  // Default off, mirroring the `enable_nvtx_tracking` default in
  // CuptiTracerOptions. When off, PushAnnotation takes the AnnotationStack
  // branch, which is exactly the pre-existing nvtx_utils_stub.cc behaviour and
  // keeps kTfOp/hlo_op intact.
  //
  // TODO: promote this to a real profiler option plumbed through
  // RocmTracerOptions, matching CUDA's "gpu_enable_nvtx_tracking". An env var
  // is used here because DefaultProfilerDomain() is reached from static and
  // pre-profiler-init contexts where the options struct does not yet exist.
  static ProfilerDomainHandle domain = []() -> ProfilerDomainHandle {
    bool enabled = false;
    // ReadBoolFromEnvVar accepts only 0/false/1/true (case-insensitive). A
    // malformed value leaves `enabled` false, which is the right VALUE -- but
    // silence is the wrong behaviour: "yes" or "on" would disable the feature
    // and leave the user staring at an empty timeline with nothing to go on.
    // Warn, then take the safe default.
    if (absl::Status s = ReadBoolFromEnvVar("XLA_ROCM_ENABLE_ROCTX",
                                            /*default_val=*/false, &enabled);
        !s.ok()) {
      LOG(WARNING) << "XLA_ROCM_ENABLE_ROCTX is set to an unrecognised value; "
                      "expected one of 0/false/1/true. ROCTX emission stays "
                      "disabled. "
                   << s.message();
    }
    if (!enabled) return nullptr;
    // ProfilerDomain is a forward-declared opaque struct and roctx has no
    // domain concept, so any stable non-null address will do. It is never
    // dereferenced and never passed to a roctx API.
    static char sentinel;
    return reinterpret_cast<ProfilerDomainHandle>(&sentinel);
  }();
  return domain;
}

void RangePush(ProfilerDomainHandle /*domain*/, const char* ascii) {
  // Pipeline A: populate AnnotationStack so the HIP API callback in
  // RocmTracer::InitProfiling can read it and attach annotations to kernel
  // dispatch events via correlation ID.
  //
  // The IsEnabled() guard mirrors the one in scoped_annotation.h's
  // AnnotationStack branch. If Enable() is
  // toggled between a push and its matching pop, the generation-based
  // cleanup in AnnotationStack resets thread-local state, so the stack
  // cannot become permanently unbalanced.
  if (AnnotationStack::IsEnabled()) {
    AnnotationStack::PushAnnotation(ascii);
  }

  // Pipeline B: emit a roctx marker so RocmTracer::MarkerCallback can capture
  // it as a named range event carrying the kNVTXRange stat.
  roctxRangePushA(ascii);
}

void RangePop(ProfilerDomainHandle /*domain*/) {
  if (AnnotationStack::IsEnabled()) {
    AnnotationStack::PopAnnotation();
  }

  roctxRangePop();
}

// Return values from roctx naming APIs intentionally discarded to match
// the void signatures declared in nvtx_utils.h.
void NameCurrentThread(const std::string& name) {
  (void)roctxNameOsThread(name.c_str());
}

void NameDevice(int device_id, const std::string& name) {
  (void)roctxNameHipDevice(name.c_str(), device_id);
}

void NameStream(StreamHandle stream, const std::string& name) {
  // StreamHandle is an opaque tsl::profiler::Stream*. Callers pass
  // hipStream_t (== ihipStream_t*) through this opaque handle, mirroring
  // how nvtx_utils.cc casts StreamHandle to CUstream.
  (void)roctxNameHipStream(name.c_str(),
                           reinterpret_cast<const ihipStream_t*>(stream));
}

namespace detail {
// Reached by the range-generator overload of ScopedAnnotation -- the path XLA
// uses for HLO ops (xla::gpu::RangePush in xla/backends/gpu/runtime/
// annotation.cc -> nvtx_utils.h's 3-arg RangePush template). CUDA implements
// this with nvtxDomainRangePushEx, attaching a registered string and a
// structured payload. roctx has neither, so we recover the interned title from
// RegisterString and fall through to the plain dual push.
//
// `schema_id` and `payload` are dropped. That is a permanent partial-parity
// ceiling, not a TODO: roctx has no NVTX_PAYLOAD_EVTATTR_SET equivalent, so
// InstructionAnnotation's structured metadata cannot be carried. The title
// survives, which is what renders as the timeline band label.
//
// KNOWN LIMITATION, and not fixable from this file. What lands on the
// AnnotationStack here is nvtx_name_ (MakeInstructionTitle), NOT xprof_name_
// (MakeInstructionName). scoped_annotation.h's two-generator PushAnnotation
// returns immediately after RangePush on the domain path, so the annotation
// generator -- the one that yields xprof_name_ -- is never invoked. Below
// TraceAnnotationLevel::kDetailed the two strings are identical, which is the
// default, so there is no difference in the common case. At
// --xla_gpu_trace_annotation_level=1 kernel events lose the op_type, op_name,
// source_file, source_line and shape stats that rocm_collector.cc parses out
// of the detailed form.
//
// There is no local fix: this function receives only the nvtx_name_ handle and
// an opaque payload blob (the Basic/Detailed/Collective struct, not the
// InstructionAnnotation), so xprof_name_str_ is unreachable from here. Pushing
// xprof_name_ in scoped_annotation.h instead would double-push against the
// single PopAnnotation; changing annotation.cc's RangePush would relabel CUDA's
// Nsight bands. Fixing it means changing shared TSL surface, which belongs in
// its own review.
//
// A null title means RegisterString was never called for this annotation. We
// still push -- an empty band is wrong, but silently skipping the push while
// ~ScopedAnnotation goes on to pop would unbalance both stacks, which is
// strictly worse and much harder to diagnose.
void RangePush(ProfilerDomainHandle domain, StringHandle title, uint64_t,
               const void*, size_t) {
  const auto* text = reinterpret_cast<const std::string*>(title);
  ::tsl::profiler::RangePush(domain, text != nullptr ? text->c_str() : "");
}
}  // namespace detail

// roctx has no schema concept; payloads are dropped in detail::RangePush.
uint64_t RegisterSchema(ProfilerDomainHandle, const void*) { return 0; }

StringHandle RegisterString(ProfilerDomainHandle, const std::string& str) {
  // roctx has no nvtxDomainRegisterStringA equivalent, so intern here and hand
  // back a stable address for detail::RangePush to recover. node_hash_set is
  // required over flat_hash_set: the returned pointer is dereferenced later and
  // must survive rehashing.
  //
  // The pool is process-lifetime and never freed, so it MUST be bounded. It is
  // not "one entry per distinct annotation" in any small sense:
  //
  //   * InstructionAnnotation's constructor (xla/backends/gpu/runtime/
  //     annotation.cc) calls RegisterString up to 12 times per HLO
  //     INSTRUCTION, including InstructionAsString(inst) -- the full ToString
  //     -- and "\n" + CalledInstructionsAsString(inst), which for a fusion is
  //     the entire body of the called computation. Multiple KB per
  //     instruction, tens of thousands of instructions per module.
  //   * ROCm dereferences exactly ONE of those handles: nvtx_name_, in
  //     detail::RangePush. Every payload handle is interned and then discarded,
  //     because roctx has no schema/payload concept. Pure cost.
  //   * ModuleAnnotations is built per GpuExecutable construction, not per
  //     dispatch -- so the mutex is off the hot path -- but the pool grows with
  //     the number of COMPILATIONS, which is unbounded over the lifetime of a
  //     JAX or serving process (retracing, shape variants, autotuning).
  //     Executables are destroyed; interned strings are not.
  //
  // Two bounds, therefore.
  //
  // Per-string: 4 KiB, deliberately far below CUDA's 65330. CUDA needs the
  // large cap because it genuinely registers those multi-KB payload strings
  // with NVTX and Nsight displays them. ROCm reads only nvtx_name_, which is
  // MakeInstructionTitle -- "Thunk:#name=...,hlo_op=...,unique_hlo_op_id=N#",
  // realistically a few hundred bytes. Sizing for the strings we discard would
  // let one large module consume the whole pool. Like CUDA we append a marker
  // rather than cutting silently, so a truncated label says so in the trace.
  //
  // Total: 64 MiB across the process. This is the bound that matters, since
  // the pool outlives every executable that contributed to it.
  static constexpr absl::string_view kTruncationMarker = "...[truncated]";
  static constexpr size_t kMaxStringBytes = 4096;
  static constexpr size_t kMaxPoolBytes = 64u << 20;  // 64 MiB

  // Returned once the cap is reached, in preference to nullptr or "".
  //
  // Not for the reason an earlier version of this comment gave: an empty label
  // is NOT a parsing hazard, because ParseAnnotationStack splits with
  // absl::SkipEmpty() and "outer::::inner" parses to ["outer", "inner"]. The
  // actual reason is diagnostic. A null or empty label makes the band silently
  // vanish -- MarkerCallback drops unlabelled ranges at pop -- so the user sees
  // a thinning trace with no cause. A real, obviously-wrong string surfaces the
  // exhaustion in the artifact itself, where someone will actually notice it.
  static const std::string* const kPoolExhausted =
      new std::string("<xla annotation pool exhausted>");

  static absl::Mutex mu(absl::kConstInit);
  static auto* pool = new absl::node_hash_set<std::string>();
  static size_t pool_bytes = 0;

  // Build the key once. Under the cap this is a view onto the caller's string
  // with no copy; over it, the one allocation is the price of the marker.
  std::string truncated_storage;
  absl::string_view key(str);
  if (key.size() > kMaxStringBytes) {
    truncated_storage.assign(
        key.substr(0, kMaxStringBytes - kTruncationMarker.size()));
    truncated_storage.append(kTruncationMarker);
    key = truncated_storage;
  }

  absl::MutexLock lock(&mu);
  if (auto it = pool->find(key); it != pool->end()) {
    return reinterpret_cast<StringHandle>(const_cast<std::string*>(&*it));
  }
  if (pool_bytes + key.size() > kMaxPoolBytes) {
    LOG_FIRST_N(WARNING, 1)
        << "XLA ROCm annotation intern pool reached its " << kMaxPoolBytes
        << "-byte cap; further annotation labels will render as \""
        << *kPoolExhausted
        << "\". This is a symptom of a very large or repeatedly recompiled "
           "module with XLA_ROCM_ENABLE_ROCTX set.";
    return reinterpret_cast<StringHandle>(
        const_cast<std::string*>(kPoolExhausted));
  }
  pool_bytes += key.size();
  return reinterpret_cast<StringHandle>(
      const_cast<std::string*>(&*pool->emplace(key).first));
}

// roctx has no memory-marking API (no nvtxMemMarkInitialized equivalent).
void MarkMemoryInitialized(void const*, size_t, StreamHandle) {}

}  // namespace tsl::profiler
