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

#include "absl/log/log.h"
#include "absl/status/status.h"
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
  // TODO(rocm-profiler): promote this to a real profiler option plumbed through
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
  // AnnotationStack branch. If Enable() toggles between a push and its
  // matching pop: Enable() strictly increases generation_ on every state
  // change, and GetAnnotationData() wipes the thread-local stack on the first
  // access after a bump -- i.e. at or before the unmatched pop. The unmatched
  // pop therefore always finds an empty stack, which PopAnnotation no-ops.
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
//
// TODO(rocm-profiler): these land under ROCPROFILER_CALLBACK_TRACING_
// MARKER_NAME_API, which rocm_tracer.cc does not subscribe, so the names never
// reach XProf and help only an external rocprofv3 run. NameStream has no ROCm
// caller at all.
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
  (void)roctxNameHipStream(
      name.c_str(), reinterpret_cast<const struct ihipStream_t*>(stream));
}

namespace detail {
// Reached by the range-generator overload of ScopedAnnotation -- the path XLA
// uses for HLO ops. CUDA implements this with nvtxDomainRangePushEx, attaching
// a registered string and a structured payload. roctx has neither, so the
// `title` handle is unusable here and we push `title_text` instead.
//
// `schema_id` and `payload` are dropped: roctx has no NVTX_PAYLOAD_EVTATTR_SET
// equivalent, so InstructionAnnotation's structured metadata cannot be carried.
// The title survives, which is what labels the timeline band.
//
// A null `title_text` still pushes. Skipping the push while ~ScopedAnnotation
// goes on to pop would unbalance both stacks, which is strictly worse than an
// empty band and much harder to diagnose.
//
// TODO(rocm-profiler): what lands on the AnnotationStack here is nvtx_name_
// (MakeInstructionTitle), not xprof_name_ (MakeInstructionName), because
// scoped_annotation.h's two-generator PushAnnotation returns before invoking
// the annotation generator. The two are identical below
// TraceAnnotationLevel::kDetailed, which is the default; at
// --xla_gpu_trace_annotation_level=1 kernel events lose the op_type, op_name,
// source_file, source_line and shape stats.
void RangePush(ProfilerDomainHandle domain, StringHandle /*title*/,
               const char* title_text, uint64_t, const void*, size_t) {
  ::tsl::profiler::RangePush(domain, title_text != nullptr ? title_text : "");
}
}  // namespace detail

// roctx has no schema concept; payloads are dropped in detail::RangePush.
uint64_t RegisterSchema(ProfilerDomainHandle, const void*) { return 0; }

// roctx has no nvtxDomainRegisterStringA equivalent, so there is nothing to
// register: detail::RangePush takes the title text directly. Callers keep the
// strings alive, exactly as on the plain RangePush path.
StringHandle RegisterString(ProfilerDomainHandle, const std::string&) {
  return {};
}

// roctx has no memory-marking API (no nvtxMemMarkInitialized equivalent).
// Permanently a no-op rather than a TODO: the callers exist to suppress false
// positives from compute-sanitizer initcheck, which is CUDA-only.
void MarkMemoryInitialized(void const*, size_t, StreamHandle) {}

}  // namespace tsl::profiler
