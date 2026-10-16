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

// ROCm implementation of range_annotations.h, using roctx.
//
// RangePush/RangePop dual-push: AnnotationStack, which carries the kTfOp and
// hlo_op stats, and roctxRangePushA, which rocm_tracer.cc renders as a
// "Host Threads/<tid>/ROCTX" band. nvtx_range_annotations.cc pushes only NVTX,
// because CUPTI recovers the range itself -- it attaches kNVTXRange to the
// kernel event from the MARKER activity (cupti_collector.cc). The ROCm tracer
// has no equivalent marker-to-stat path, so without this second push the stats
// would be lost on every event.
//
// Opt-in via XLA_ROCM_ENABLE_ROCTX; see DefaultProfilerDomain.

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/base/no_destructor.h"
#include "absl/container/node_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "rocm/include/rocprofiler-sdk-roctx/roctx.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/profiler/lib/range_annotations.h"

namespace tsl::profiler {

ProfilerDomainHandle DefaultProfilerDomain() {
  // Latched: computed once per process. This is a correctness requirement, not
  // a cache -- scoped_annotation.h reads the handle separately when pushing and
  // when popping, so a value that moved mid-process would push down one path
  // and pop down the other, unbalancing both stacks. Never derive it from
  // anything dynamic (profiler state, tracer state, AnnotationStack::IsEnabled)
  // -- only from process-lifetime configuration.
  //
  // TODO(rocm-profiler): promote to a RocmTracerOptions field, matching CUDA's
  // gpu_enable_nvtx_tracking. An env var is used because this is reached from
  // pre-profiler-init contexts, where the options struct does not yet exist.
  static ProfilerDomainHandle domain = []() -> ProfilerDomainHandle {
    bool enabled = false;
    // ReadBoolFromEnvVar accepts only 0/false/1/true and leaves `enabled` false
    // otherwise. That is the right value, but silently disabling on "yes" or
    // "on" would leave the user staring at an empty timeline; warn first.
    if (absl::Status s = ReadBoolFromEnvVar("XLA_ROCM_ENABLE_ROCTX",
                                            /*default_val=*/false, &enabled);
        !s.ok()) {
      LOG(WARNING) << "XLA_ROCM_ENABLE_ROCTX is set to an unrecognised value; "
                      "expected one of 0/false/1/true. ROCTX emission stays "
                      "disabled. "
                   << s.message();
    }
    if (!enabled) return nullptr;
    // roctx has no domain concept, so any stable non-null address will do; it
    // is never dereferenced and never passed to a roctx API.
    static char sentinel;
    return reinterpret_cast<ProfilerDomainHandle>(&sentinel);
  }();
  return domain;
}

// `domain` is unused: callers gate on it (scoped_annotation.h), and roctx has
// no domain to pass it to.
void RangePush(ProfilerDomainHandle /*domain*/, const char* ascii) {
  // The IsEnabled() guard can disagree between a push and its matching pop, but
  // not harmfully: Enable() bumps generation_ on every state change, and
  // GetAnnotationData() wipes the thread-local stack on the first access after
  // a bump, so an unmatched pop always finds an empty stack and no-ops.
  if (AnnotationStack::IsEnabled()) {
    AnnotationStack::PushAnnotation(ascii);
  }

  roctxRangePushA(ascii);
}

void RangePop(ProfilerDomainHandle /*domain*/) {
  if (AnnotationStack::IsEnabled()) {
    AnnotationStack::PopAnnotation();
  }

  roctxRangePop();
}

// Naming takes no domain and so is ungated, matching nvtx_range_annotations.cc.
// A ROCm process therefore emits a few MARKER_NAME_API records even with
// XLA_ROCM_ENABLE_ROCTX unset -- no ranges, and no per-op cost.
//
// TODO(rocm-profiler): rocm_tracer.cc does not subscribe MARKER_NAME_API, so
// these names reach only an external rocprofv3 run. NameStream has no ROCm
// caller at all.
void NameCurrentThread(const std::string& name) {
  (void)roctxNameOsThread(name.c_str());
}

void NameDevice(int device_id, const std::string& name) {
  (void)roctxNameHipDevice(name.c_str(), device_id);
}

void NameStream(StreamHandle stream, const std::string& name) {
  // Contract: stream must have been cast from a hipStream_t, mirroring how
  // nvtx_range_annotations.cc casts StreamHandle to CUstream. The DCHECK
  // catches a null handle without pulling in a HIP header.
  DCHECK(stream != nullptr) << "NameStream called with null StreamHandle; "
                               "caller must pass a valid hipStream_t";
  (void)roctxNameHipStream(
      name.c_str(), reinterpret_cast<const struct ihipStream_t*>(stream));
}

namespace detail {
// The HLO-op path. roctx has no registered-string or payload API, so `title` is
// resolved back to the interned text and schema_id/payload are dropped.
//
// A null title still pushes: RegisterOptionalString returns null for empty
// strings, and skipping the push while ~ScopedAnnotation still pops would
// unbalance both stacks.
//
// TODO(rocm-profiler): the AnnotationStack receives nvtx_name_, not
// xprof_name_, because scoped_annotation.h returns before invoking the
// annotation generator. The two are identical at TraceAnnotationLevel::kBasic,
// which is the default; they diverge at kDetailed
// (--xla_gpu_trace_annotation_level=1).
void RangePush(ProfilerDomainHandle domain, StringHandle title, uint64_t,
               const void*, size_t) {
  const char* text = reinterpret_cast<const char*>(title);
  ::tsl::profiler::RangePush(domain, text != nullptr ? text : "");
}
}  // namespace detail

uint64_t RegisterSchema(ProfilerDomainHandle, const void*) { return 0; }

// Interning, not borrowing: annotation.cc passes temporaries, so the caller's
// storage is gone by the time the handle is used. node_hash_set gives pointer
// stability and entries are never erased, matching NVTX's registered-string
// contract -- so the table grows with the number of distinct strings and is
// never reclaimed.
//
// TODO(rocm-profiler): expose a kSupportsPayload constant from
// range_annotations.h so annotation.cc can skip the payload-only registrations,
// which on ROCm are interned and then never read.
StringHandle RegisterString(ProfilerDomainHandle, const std::string& str) {
  static absl::NoDestructor<absl::Mutex> mu;
  static absl::NoDestructor<absl::node_hash_set<std::string>> interned;
  absl::MutexLock lock(mu.get());
  return reinterpret_cast<StringHandle>(
      const_cast<char*>(interned->insert(str).first->c_str()));
}

// Permanently a no-op, not a TODO: roctx has no equivalent, and the callers
// exist to suppress compute-sanitizer initcheck false positives, a CUDA tool.
void MarkMemoryInitialized(void const*, size_t, StreamHandle) {}

}  // namespace tsl::profiler
