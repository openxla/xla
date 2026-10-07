/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PROFILER_LIB_RANGE_ANNOTATIONS_H_
#define TENSORFLOW_TSL_PROFILER_LIB_RANGE_ANNOTATIONS_H_

#include <stddef.h>

#include <cstdint>
#include <string>

// Backend-neutral interface for emitting named ranges to a GPU profiler.
// Implemented by nvtx_range_annotations.cc (NVTX), roctx_range_annotations.cc
// (ROCTX) and range_annotations_stub.cc (no-op); exactly one is linked in, and
// callers must not depend on which.
namespace tsl::profiler {
struct String;
// Opaque handle to a string pre-registered with the active backend. The value
// is meaningful only to that backend and must stay valid for as long as the
// caller may push it -- in practice, for the process lifetime.
using StringHandle = String*;

struct ProfilerDomain;
// Opaque handle to a range-emission domain in the active backend.
using ProfilerDomainHandle = ProfilerDomain*;

// Non-null if range emission is enabled for the active backend, otherwise null.
// The value is opaque -- it need not point at anything -- but it MUST be stable
// for the process lifetime: callers (see scoped_annotation.h) read it
// separately when pushing and when popping a range, so a value that changed
// mid-process would push down one path and pop down the other. Deriving it from
// anything dynamic -- profiler or tracer state, whether tracing is active -- is
// therefore a bug, even when the value happens not to move.
//
// Returning non-null also opts the backend OUT of AnnotationStack:
// scoped_annotation.h routes to RangePush/RangePop instead, so a backend whose
// ranges feed the same XSpace must push and pop AnnotationStack itself or lose
// the kTfOp and hlo_op stats on every event.
ProfilerDomainHandle DefaultProfilerDomain();

// Assign a human-readable name to the current thread
void NameCurrentThread(const std::string&);

// Assign a human-readable name to the given local device
void NameDevice(int device_id, const std::string& device_name);

struct Stream;
// Opaque handle to an execution stream
using StreamHandle = Stream*;

// Assign a human-readable name to the given execution stream
void NameStream(StreamHandle stream, const std::string& stream_name);

// Register a string with the active backend for faster repeated use. The
// returned handle stays valid for the process lifetime; the caller's string
// need not outlive the call.
StringHandle RegisterString(ProfilerDomainHandle, const std::string&);

// End a range that was created on this thread by RangePush
void RangePop(ProfilerDomainHandle);

// Older/simpler version. Backends must copy the string: the pointer is
// guaranteed valid only for the duration of the call.
void RangePush(ProfilerDomainHandle domain, const char*);
inline void RangePush(ProfilerDomainHandle domain, const std::string& str) {
  RangePush(domain, str.c_str());
}

namespace detail {
// `title` may be null: RegisterOptionalString yields null for an empty string.
// Backends must still push, so the matching RangePop stays balanced.
void RangePush(ProfilerDomainHandle domain, StringHandle title,
               uint64_t schema_id, const void* payload, size_t payload_size);
}  // namespace detail

// More powerful version: pass a registered string instead of a C-style
// string, and attach a generic payload. The Annotation type must implement a
// method called NvtxSchemaId() -- named for the backend that introduced it --
// identifying the schema the payload conforms to.
//
// Payload support is optional. A backend with no structured-payload concept
// (ROCTX) drops schema_id and payload and emits the title alone, so callers
// must not rely on the payload reaching the trace.
template <typename Annotation>
void RangePush(ProfilerDomainHandle domain, StringHandle title,
               const Annotation& annotation) {
  return detail::RangePush(domain, title, annotation.NvtxSchemaId(),
                           &annotation, sizeof(Annotation));
}

// Register the schema of a custom payload type, for use with the more powerful
// version of RangePush. Backends without payload support return 0.
uint64_t RegisterSchema(ProfilerDomainHandle domain, const void* schemaAttr);

// Mark a memory region as initialized.
// This mitigates false positives from CUDA's compute sanitizer (initcheck), and
// is a no-op on backends with no equivalent tool.
void MarkMemoryInitialized(void const* address, size_t size,
                           StreamHandle stream);
}  // namespace tsl::profiler
#endif  // TENSORFLOW_TSL_PROFILER_LIB_RANGE_ANNOTATIONS_H_
