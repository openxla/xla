/* Copyright 2020 The OpenXLA Authors.

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
#include "xla/python/profiler/internal/python_hooks.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <stack>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xla/python/profiler/internal/traceme_state.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"

#if PY_VERSION_HEX < 0x030C0000
#include "tsl/platform/platform.h"
#endif  // PY_VERSION_HEX < 0x030C0000

#ifdef Py_GIL_DISABLED
#include "absl/synchronization/mutex.h"
#endif  // Py_GIL_DISABLED

namespace xla {
namespace profiler {

namespace {

bool SysSetProfileNone() {
  PyObject* sys_mod = PyImport_ImportModule("sys");
  if (sys_mod == nullptr) {
    return false;
  }
  PyObject* res = PyObject_CallMethod(sys_mod, "setprofile", "O", Py_None);
  Py_DECREF(sys_mod);
  Py_XDECREF(res);
  return res != nullptr;
}

bool ThreadingSetProfile(PyObject* callback) {
  PyObject* threading_mod = PyImport_ImportModule("threading");
  if (threading_mod == nullptr) {
    return false;
  }
  PyObject* res = PyObject_CallMethod(threading_mod, "setprofile", "O",
                                      callback != nullptr ? callback : Py_None);
  Py_DECREF(threading_mod);
  Py_XDECREF(res);
  return res != nullptr;
}

const char* PyUnicodeToStringOrUnknown(PyObject* obj) {
  if (obj != nullptr && PyUnicode_Check(obj)) {
    const char* str = PyUnicode_AsUTF8(obj);
    if (str != nullptr) {
      return str;
    }
    PyErr_Clear();
  }
  return "<unknown>";
}

// Attempts to extract the UTF-8 string view from `obj` without holding the GIL.
// Safe when `obj` has a positive reference count held by the caller:
// - Compact ASCII `PyUnicodeObject` instances store immutable UTF-8 bytes
//   inline right after `PyASCIIObject`.
// - Non-unicode objects (e.g., `Py_None` or `PyModuleObject` in `m_module`)
//   always map to `"<unknown>"`.
// Returns false only for non-ASCII `PyUnicodeObject` instances (or builds using
// `Py_LIMITED_API`), which must fall back to `PyUnicode_AsUTF8` under the GIL.
inline bool TryExtractWithoutGil(PyObject* obj, absl::string_view* out) {
  if (obj == nullptr) {
    *out = "<unknown>";
    return true;
  }
#if !defined(Py_LIMITED_API)
  if (PyUnicode_CheckExact(obj)) {
    if (PyUnicode_IS_COMPACT_ASCII(obj)) {
      const char* data =
          reinterpret_cast<const char*>(_PyASCIIObject_CAST(obj) + 1);
      *out = absl::string_view(data, PyUnicode_GET_LENGTH(obj));
      return true;
    }
    return false;
  }
  if (!PyUnicode_Check(obj)) {
    *out = "<unknown>";
    return true;
  }
#endif  // !defined(Py_LIMITED_API)
  return false;
}

// Releases `count` references to `op`. Must be called with the GIL held.
inline void ReleasePyObjectRefs(PyObject* op, size_t count) {
  if (op == nullptr || count == 0) {
    return;
  }
#if !defined(Py_GIL_DISABLED) && !defined(Py_LIMITED_API) && \
    !defined(Py_REF_DEBUG)
#if PY_VERSION_HEX >= 0x030C0000
  if (_Py_IsImmortal(op)) {
    return;
  }
#endif  // PY_VERSION_HEX >= 0x030C0000
  if (static_cast<size_t>(op->ob_refcnt) > count) {
    op->ob_refcnt -= count;
    return;
  }
  op->ob_refcnt -= (count - 1);
  Py_DECREF(op);
#else
  for (size_t i = 0; i < count; ++i) {
    Py_DECREF(op);
  }
#endif
}

// Releases references held by a single `PythonTraceEntry`. Must be called with
// the GIL held.
inline void ReleaseEntryRefs(const PythonTraceEntry& entry) {
  if (entry.IsPythonCall()) {
    Py_XDECREF(entry.co_filename);
    Py_XDECREF(entry.co_name);
  } else {
    Py_XDECREF(entry.m_module);
    if (entry.c_capsule != nullptr) {
      Py_DECREF(entry.c_capsule);
    }
  }
}

std::string GetEventName(PyObject* co_filename, PyObject* co_name,
                         int co_firstlineno) {
  absl::string_view filename = PyUnicodeToStringOrUnknown(co_filename);
  absl::string_view function = PyUnicodeToStringOrUnknown(co_name);

  return absl::StrCat("$", tsl::io::Basename(filename), ":", co_firstlineno,
                      " ", function);
}

std::string GetEventName(absl::string_view method_name, PyObject* module) {
  // Python stack does not have a filename/line_no for native calls.
  // Use module name and function/method name instead.
  absl::string_view filename = PyUnicodeToStringOrUnknown(module);
  if (!method_name.empty()) {
    return absl::StrCat("$", filename, " ", method_name);
  }
  return "$<unknown>";
}

void AddEventToXLine(const TraceEventInfo& event,
                     tsl::profiler::XLineBuilder* line,
                     tsl::profiler::XPlaneBuilder* plane) {
  // TODO(jiesun): maybe add full filename as event stats.
  auto xevent = line->AddEvent(*plane->GetOrCreateEventMetadata(event.name));
  xevent.SetTimestampNs(event.start_time_ns);
  xevent.SetEndTimestampNs(event.end_time_ns);
}

#if PY_VERSION_HEX < 0x030C0000
template <typename ForEachThreadFunc>
void ForEachThread(PyThreadState* curr_thread, ForEachThreadFunc&& callback) {
  // Note: PyThreadState's interp is not accessible in open source due to
  // Py_LIMITED_API definition nuances. We can not iterate all threads through
  // that PyInterpreterState.
  if constexpr (tsl::kIsDebugBuild) {
    // If debug version of python runtime is used, (e.g. monolithic binaries in
    // g3). PyGILState_Check will fail because current thread's PyThreadState
    // is not the one that holding GIL (after PyThreadState_Swap). This extra
    // check in PyEval_SetProfile is not useful, but will sporadic crash if user
    // use debug version of python runtime. In this case, we fallback to only
    // set up profile hooks in current threads.
    // In OSS, the python runtime and tensorflow profiler are built separately.
    // So this workaround doesn't apply.
    callback(curr_thread);
  } else {
    for (PyThreadState* p = curr_thread; p != nullptr; p = p->next) {
      PyThreadState_Swap(p);
      callback(p);
    }
    for (PyThreadState* p = curr_thread->prev; p != nullptr; p = p->prev) {
      PyThreadState_Swap(p);
      callback(p);
    }
  }
}

#endif  // PY_VERSION_HEX

}  // namespace

/*static*/ PythonHookContext* PythonHooks::e2e_context_ = nullptr;

/*static*/ PyObject* PythonHookContext::PyProfileCallback(PyObject* /*self*/,
                                                          PyObject* const* args,
                                                          Py_ssize_t nargs) {
  if (nargs != 3) {
    PyErr_SetString(PyExc_TypeError, "profile_callback expects 3 arguments");
    return nullptr;
  }
  PythonHooks* singleton = PythonHooks::GetSingleton();
  if (!singleton->active_context_ ||
      singleton->active_context_->stopped_.load(std::memory_order_relaxed)) {
    SysSetProfileNone();
    Py_RETURN_NONE;
  }
  const char* event = PyUnicode_AsUTF8(args[1]);
  if (event == nullptr) {
    return nullptr;
  }

  singleton->ProfileSlow(reinterpret_cast<PyFrameObject*>(args[0]), event,
                         args[2]);
  if (!SysSetProfileNone()) {
    return nullptr;
  }
  PyEval_SetProfile(&PythonHooks::ProfileFunction, nullptr);
  Py_RETURN_NONE;
}

/*static*/ PyObject* PythonHookContext::PyAtexitCallback(PyObject* /*self*/,
                                                         PyObject* /*args*/) {
  PythonHooks* singleton = PythonHooks::GetSingleton();
  auto e2e_context = singleton->Stop();
  // Serialize into internal storage before the tracked PyCodeObjects
  // went out of scope.
  if (e2e_context) {
    e2e_context->CollectData(nullptr);
    PythonHooks::set_e2e_context(e2e_context.release());
  }
  Py_RETURN_NONE;
}

std::string PythonTraceEntry::Name() const {
  if (IsPythonCall()) {
    return GetEventName(co_filename, co_name, co_firstlineno);
  }
  return GetEventName(
      method_name != nullptr ? absl::string_view(method_name) : "", m_module);
}

PythonHooks* PythonHooks::GetSingleton() {
  static PythonHooks* const singleton = new PythonHooks;
  return singleton;
}

PythonHookContext::~PythonHookContext() {
  if (!Py_IsInitialized()) {
    return;
  }
  bool has_entries = false;
  for (const auto& shard : entry_shards_) {
    if (!shard.entries.empty()) {
      has_entries = true;
      break;
    }
  }
  if (!has_entries) {
    return;
  }
  PyGILState_STATE gil_state = PyGILState_Ensure();
  for (auto& shard : entry_shards_) {
#ifdef Py_GIL_DISABLED
    absl::MutexLock lock(&shard.mu);
#endif  // Py_GIL_DISABLED
    // NOLINTNEXTLINE
    for (auto& [thread_id, thread_events] : shard.entries) {
      for (const PythonTraceEntry& event : thread_events.completed) {
        ReleaseEntryRefs(event);
      }
      while (!thread_events.active.empty()) {
        ReleaseEntryRefs(thread_events.active.top());
        thread_events.active.pop();
      }
      while (!thread_events.active_c.empty()) {
        ReleaseEntryRefs(thread_events.active_c.top());
        thread_events.active_c.pop();
      }
    }
    shard.entries.clear();
  }
  PyGILState_Release(gil_state);
}

void PythonHookContext::Start(const PythonHooksOptions& options) {
  if (!Py_IsInitialized()) {
    return;
  }

  options_ = options;
  start_timestamp_ns_ = tsl::profiler::GetCurrentTimeNanos();
  if (options_.enable_python_traceme || options_.enable_trace_python_function) {
    PyGILState_STATE gil_state = PyGILState_Ensure();
    if (options_.enable_python_traceme) {
      traceme_enabled = true;
    }
    if (options_.enable_trace_python_function) {
      if (!SetProfilerInAllThreads()) {
        LOG(ERROR) << "Can't set profiler in all threads.";
        PyErr_Print();
      }
    }
    if (options_.end_to_end_mode) {
      if (!RegisterAtexitCallback()) {
        LOG(ERROR) << "Can't install atexit handler for e2e mode.";
        PyErr_Print();
      }
    }
    PyGILState_Release(gil_state);
  }
}

void PythonHookContext::Stop() {
  stopped_.store(true, std::memory_order_release);
  if (!Py_IsInitialized()) {
    return;
  }
  if (options_.enable_python_traceme || options_.enable_trace_python_function) {
    PyGILState_STATE gil_state = PyGILState_Ensure();
    if (options_.enable_trace_python_function) {
      if (!ClearProfilerInAllThreads()) {
        LOG(ERROR) << "Can't clear profiler in all threads.";
        PyErr_Print();
      }
    }
    if (options_.enable_python_traceme) {
      traceme_enabled = false;
    }
    PyGILState_Release(gil_state);
  }
}

std::vector<PerThreadConsumeData> PythonHookContext::Consume() {
  struct HarvestedThread {
    int64_t thread_id;
    std::deque<PythonTraceEntry> completed;
    std::stack<PythonTraceEntry> active;
    std::stack<PythonTraceEntry> active_c;
  };
  std::vector<HarvestedThread> harvested;
  std::vector<HarvestedThread> discarded_incomplete;
  const bool stopped = stopped_.load(std::memory_order_acquire);

  // Phase 1: Acquire GIL only to O(threads) swap the event queues out of
  // `entry_shards_`, then immediately release the GIL.
  {
    PyGILState_STATE gil_state;
    bool has_gil = false;
    if (Py_IsInitialized()) {
      gil_state = PyGILState_Ensure();
      has_gil = true;
    }

    for (EntryShard& shard : entry_shards_) {
#ifdef Py_GIL_DISABLED
      absl::MutexLock lock(shard.mu);
#else
      DCHECK(!has_gil || PyGILState_Check());
#endif  // Py_GIL_DISABLED
      // NOLINTNEXTLINE
      for (auto& [thread_id, thread_events] : shard.entries) {
        HarvestedThread ht;
        ht.thread_id = thread_id;
        ht.completed.swap(thread_events.completed);
        if (stopped) {
          if (options_.include_incomplete_events) {
            ht.active.swap(thread_events.active);
            ht.active_c.swap(thread_events.active_c);
          } else if (!thread_events.active.empty() ||
                     !thread_events.active_c.empty()) {
            HarvestedThread disc;
            disc.thread_id = thread_id;
            disc.active.swap(thread_events.active);
            disc.active_c.swap(thread_events.active_c);
            discarded_incomplete.push_back(std::move(disc));
          }
        }
        if (!ht.completed.empty() || !ht.active.empty() ||
            !ht.active_c.empty()) {
          harvested.push_back(std::move(ht));
        }
      }
      if (stopped) {
        shard.entries.clear();
      }
    }

    if (has_gil) {
      PyGILState_Release(gil_state);
    }
  }

  if (harvested.empty() && discarded_incomplete.empty()) {
    return {};
  }

  // Phase 2 (WITHOUT GIL): Scan all harvested entries, deduplicate unique
  // functions by raw PyObject* pointer tuples, format compact-ASCII names,
  // build `consumed_data`, and free the `std::deque` buffers.
  using CodeKey = std::tuple<PyObject*, PyObject*, int>;
  using CFuncKey = std::tuple<PyObject*, PyObject*, const char*>;
  struct EventSymbol {
    std::string name;
    size_t ref_count = 0;
    int deferred_idx = -1;
  };
  struct DeferredFormatTask {
    PyObject* obj1;
    PyObject* obj2;
    const char* method_name;
    int co_firstlineno;
    bool is_python_call;
    std::string formatted_name;
  };
  struct DeferredPatch {
    size_t thread_idx;
    size_t event_idx;
    int deferred_idx;
  };

  absl::flat_hash_map<CodeKey, EventSymbol> py_symbols;
  absl::flat_hash_map<CFuncKey, EventSymbol> c_symbols;
  py_symbols.reserve(256);
  c_symbols.reserve(128);
  std::vector<DeferredFormatTask> deferred_tasks;
  std::vector<DeferredPatch> deferred_patches;

  std::vector<PerThreadConsumeData> consumed_data;
  consumed_data.reserve(harvested.size());
  const uint64_t now = (stopped && options_.include_incomplete_events)
                           ? tsl::profiler::GetCurrentTimeNanos()
                           : 0;

  auto process_entry = [&](const PythonTraceEntry& event, uint64_t end_time_ns,
                           size_t thread_idx,
                           PerThreadConsumeData* thread_data) {
    if (event.IsPythonCall()) {
      CodeKey key(event.co_filename, event.co_name, event.co_firstlineno);
      auto [it, inserted] = py_symbols.try_emplace(key);
      EventSymbol& sym = it->second;
      sym.ref_count += 1;
      if (inserted) {
        absl::string_view file_sv;
        absl::string_view name_sv;
        if (TF_PREDICT_TRUE(TryExtractWithoutGil(event.co_filename, &file_sv) &&
                            TryExtractWithoutGil(event.co_name, &name_sv))) {
          sym.name = absl::StrCat("$", tsl::io::Basename(file_sv), ":",
                                  event.co_firstlineno, " ", name_sv);
        } else {
          sym.deferred_idx = static_cast<int>(deferred_tasks.size());
          deferred_tasks.push_back({event.co_filename, event.co_name, nullptr,
                                    event.co_firstlineno, true, std::string()});
        }
      }
      if (TF_PREDICT_TRUE(sym.deferred_idx < 0)) {
        thread_data->events.push_back(
            {sym.name, event.start_time_ns, end_time_ns});
      } else {
        deferred_patches.push_back(
            {thread_idx, thread_data->events.size(), sym.deferred_idx});
        thread_data->events.push_back(
            {std::string(), event.start_time_ns, end_time_ns});
      }
    } else {
      CFuncKey key(event.m_module, event.c_capsule, event.method_name);
      auto [it, inserted] = c_symbols.try_emplace(key);
      EventSymbol& sym = it->second;
      sym.ref_count += 1;
      if (inserted) {
        absl::string_view mod_sv;
        if (TF_PREDICT_TRUE(TryExtractWithoutGil(event.m_module, &mod_sv))) {
          absl::string_view method_sv =
              event.method_name != nullptr
                  ? absl::string_view(event.method_name)
                  : "";
          sym.name = !method_sv.empty()
                         ? absl::StrCat("$", mod_sv, " ", method_sv)
                         : "$<unknown>";
        } else {
          sym.deferred_idx = static_cast<int>(deferred_tasks.size());
          deferred_tasks.push_back({event.m_module, nullptr, event.method_name,
                                    0, false, std::string()});
        }
      }
      if (TF_PREDICT_TRUE(sym.deferred_idx < 0)) {
        thread_data->events.push_back(
            {sym.name, event.start_time_ns, end_time_ns});
      } else {
        deferred_patches.push_back(
            {thread_idx, thread_data->events.size(), sym.deferred_idx});
        thread_data->events.push_back(
            {std::string(), event.start_time_ns, end_time_ns});
      }
    }
  };

  for (HarvestedThread& ht : harvested) {
    PerThreadConsumeData thread_data;
    thread_data.thread_id = ht.thread_id;
    const size_t total_events =
        ht.completed.size() + ht.active.size() + ht.active_c.size();
    thread_data.events.reserve(total_events);
    const size_t thread_idx = consumed_data.size();

    for (const PythonTraceEntry& event : ht.completed) {
      process_entry(event, event.end_time_ns, thread_idx, &thread_data);
    }
    // Free deque memory blocks outside the GIL!
    ht.completed.clear();

    while (!ht.active.empty()) {
      const PythonTraceEntry& event = ht.active.top();
      process_entry(event, now, thread_idx, &thread_data);
      ht.active.pop();
    }
    while (!ht.active_c.empty()) {
      const PythonTraceEntry& event = ht.active_c.top();
      process_entry(event, now, thread_idx, &thread_data);
      ht.active_c.pop();
    }
    consumed_data.push_back(std::move(thread_data));
  }

  // Phase 3: Acquire GIL only for O(U) batched refcount release and any rare
  // non-ASCII string formatting.
  if (Py_IsInitialized()) {
    PyGILState_STATE gil_state = PyGILState_Ensure();
    for (DeferredFormatTask& task : deferred_tasks) {
      if (task.is_python_call) {
        task.formatted_name =
            GetEventName(task.obj1, task.obj2, task.co_firstlineno);
      } else {
        task.formatted_name = GetEventName(
            task.method_name != nullptr ? absl::string_view(task.method_name)
                                        : "",
            task.obj1);
      }
    }
    // NOLINTNEXTLINE
    for (const auto& [key, sym] : py_symbols) {
      ReleasePyObjectRefs(std::get<0>(key), sym.ref_count);
      ReleasePyObjectRefs(std::get<1>(key), sym.ref_count);
    }
    // NOLINTNEXTLINE
    for (const auto& [key, sym] : c_symbols) {
      ReleasePyObjectRefs(std::get<0>(key), sym.ref_count);
      ReleasePyObjectRefs(std::get<1>(key), sym.ref_count);
    }
    for (HarvestedThread& disc : discarded_incomplete) {
      while (!disc.active.empty()) {
        ReleaseEntryRefs(disc.active.top());
        disc.active.pop();
      }
      while (!disc.active_c.empty()) {
        ReleaseEntryRefs(disc.active_c.top());
        disc.active_c.pop();
      }
    }
    PyGILState_Release(gil_state);
  }

  // Phase 4 (WITHOUT GIL): Patch any deferred non-ASCII names.
  for (const DeferredPatch& patch : deferred_patches) {
    consumed_data[patch.thread_idx].events[patch.event_idx].name =
        deferred_tasks[patch.deferred_idx].formatted_name;
  }

  return consumed_data;
}

void PythonHookContext::CollectData(tensorflow::profiler::XPlane* raw_plane) {
  if (raw_plane == nullptr) {
    end_to_end_xplane_.emplace();
    raw_plane = &*end_to_end_xplane_;
  }
  std::vector<PerThreadConsumeData> consumed_data = Consume();
  tsl::profiler::XPlaneBuilder plane(raw_plane);
  for (const PerThreadConsumeData& thread_data : consumed_data) {
    auto line = plane.GetOrCreateLine(thread_data.thread_id);
    line.SetTimestampNs(start_timestamp_ns_);
    for (const TraceEventInfo& event : thread_data.events) {
      AddEventToXLine(event, &line, &plane);
    }
  }
}

void PythonHookContext::Finalize(tensorflow::profiler::XSpace* space) {
  if (space && options_.enable_trace_python_function) {
    tensorflow::profiler::XPlane* plane =
        tsl::profiler::FindOrAddMutablePlaneWithName(
            space, tsl::profiler::kPythonTracerPlaneName);
    if (options_.end_to_end_mode) {
      if (end_to_end_xplane_) {
        end_to_end_xplane_->set_name(plane->name());
        plane->Swap(&*end_to_end_xplane_);
        end_to_end_xplane_.reset();
      }
    } else {
      CollectData(plane);
    }
  }
}

/*static*/ int PythonHooks::ProfileFunction(PyObject* obj, PyFrameObject* frame,
                                            int what, PyObject* arg) {
  GetSingleton()->ProfileFast(frame, what, arg);
  return 0;
}

void PythonHooks::ProfileSlow(PyFrameObject* frame, absl::string_view event,
                              PyObject* arg) {
  int what;

  if (absl::ConsumePrefix(&event, "c_")) {
    if (event == "call") {
      what = PyTrace_C_CALL;
    } else if (event == "return") {
      what = PyTrace_C_RETURN;
    } else if (event == "exception") {
      what = PyTrace_C_EXCEPTION;
    } else {
      return;
    }
  } else {
    if (event == "call") {
      what = PyTrace_CALL;
    } else if (event == "return") {
      what = PyTrace_RETURN;
    } else if (event == "exception") {
      what = PyTrace_EXCEPTION;
    } else {
      return;
    }
  }

  ProfileFast(frame, what, arg);
}

void PythonHookContext::ProfileFast(PyFrameObject* frame, int what,
                                    PyObject* arg) {
  if (TF_PREDICT_FALSE(stopped_.load(std::memory_order_relaxed))) {
    return;
  }
  const int64_t thread_id = tsl::Env::Default()->GetCurrentThreadId();
  uint64_t now = tsl::profiler::GetCurrentTimeNanos();
  int shard_id = thread_id % kNumEntryShards;
  EntryShard& shard = entry_shards_[shard_id];
#ifdef Py_GIL_DISABLED
  absl::MutexLock lock(shard.mu);
#else
  DCHECK(PyGILState_Check());
#endif  // Py_GIL_DISABLED
  auto& thread_traces = shard.entries[thread_id];

  switch (what) {
    case PyTrace_CALL: {
#if PY_VERSION_HEX < 0x030b0000
      PyCodeObject* f_code = frame->f_code;
      thread_traces.active.emplace(now, 0, f_code);
#else   // PY_VERSION_HEX < 0x030b0000
      PyCodeObject* f_code = PyFrame_GetCode(frame);
      thread_traces.active.emplace(now, 0, f_code);
      Py_XDECREF(f_code);
#endif  // PY_VERSION_HEX < 0x030b0000
      break;
    }
    case PyTrace_RETURN:
    case PyTrace_EXCEPTION: {
      if (!thread_traces.active.empty()) {
        auto& entry = thread_traces.active.top();
        entry.end_time_ns = now;
        thread_traces.completed.push_back(entry);
        thread_traces.active.pop();
      } else if (options_.include_incomplete_events) {
#if PY_VERSION_HEX < 0x030b0000
        PyCodeObject* f_code = frame->f_code;
        thread_traces.completed.emplace_back(start_timestamp_ns_, now, f_code);
#else   // PY_VERSION_HEX < 0x030b0000
        PyCodeObject* f_code = PyFrame_GetCode(frame);
        thread_traces.completed.emplace_back(start_timestamp_ns_, now, f_code);
        Py_XDECREF(f_code);
#endif  // PY_VERSION_HEX < 0x030b0000
      }
      break;
    }
    case PyTrace_C_CALL: {
      if (PyCFunction_Check(arg)) {
        // Python stack does not have a filename/line_no for native calls.
        auto* func = reinterpret_cast<PyCFunctionObject*>(arg);
        thread_traces.active_c.emplace(now, 0, func);
      }
      break;
    }
    case PyTrace_C_RETURN:
    case PyTrace_C_EXCEPTION: {
      if (PyCFunction_Check(arg)) {
        if (!thread_traces.active_c.empty()) {
          auto& entry = thread_traces.active_c.top();
          entry.end_time_ns = now;
          thread_traces.completed.push_back(entry);
          thread_traces.active_c.pop();
        } else if (options_.include_incomplete_events) {
          // Only the end of the events is recorded, use profiler start as
          // start timestamp of the new event.
          auto* func = reinterpret_cast<PyCFunctionObject*>(arg);
          thread_traces.completed.emplace_back(start_timestamp_ns_, now, func);
        }
      }
      break;
    }
    default:
      break;
  }
}

/*static*/ bool PythonHookContext::SetProfilerInAllThreads() {
  // We also want any new threads started to use our profiler.
  // NOTE: threading does not provide a C API equivalent to
  // `threading.setprofile` so we are forced to go via Python to setup the
  // profile when a new thread is created. After the first callback in that
  // thread we unregister the Python profile function and use
  // `PyEval_SetProfile` to register a C profiler which has significantly less
  // overhead (>2x faster).
  static PyMethodDef kProfileMethodDef = {
      "profile_callback",
      reinterpret_cast<PyCFunction>(PythonHookContext::PyProfileCallback),
      METH_FASTCALL, nullptr};
  PyObject* callback = PyCFunction_New(&kProfileMethodDef, nullptr);
  if (callback == nullptr) {
    return false;
  }
  bool ok = ThreadingSetProfile(callback);
  Py_DECREF(callback);
  if (!ok) {
    return false;
  }

  // NOTE: This must be after `threading.setprofile` otherwise we
  // end up recording that in our trace.
#if PY_VERSION_HEX < 0x030C0000
  PyThreadState* curr_thread = PyThreadState_Get();
  ForEachThread(curr_thread, [](PyThreadState* thread) {
    VLOG(1) << "Setting profiler in " << thread->thread_id;
    PyEval_SetProfile(&PythonHooks::ProfileFunction, nullptr);
  });
  PyThreadState_Swap(curr_thread);
#else   // PY_VERSION_HEX >= 0x030C0000
  PyEval_SetProfileAllThreads(&PythonHooks::ProfileFunction, nullptr);
#endif  // PY_VERSION_HEX >= 0x030C0000
  return true;
}

/*static*/ bool PythonHookContext::ClearProfilerInAllThreads() {
#if PY_VERSION_HEX < 0x030C0000
  PyThreadState* curr_thread = PyThreadState_Get();
  ForEachThread(curr_thread, [](PyThreadState* thread) {
    VLOG(1) << "Clearing profiler in " << thread->thread_id;
    PyEval_SetProfile(nullptr, nullptr);
  });
  PyThreadState_Swap(curr_thread);
#else   // PY_VERSION_HEX >= 0x030C0000
  PyEval_SetProfileAllThreads(nullptr, nullptr);
#endif  // PY_VERSION_HEX >= 0x030C0000

  // And notify the threading library that we're done.
  return ThreadingSetProfile(nullptr);
}

/*static*/ bool PythonHookContext::RegisterAtexitCallback() {
  // When end to end mode is used, Stop() and Finalize() i.e. symbolization
  // and data collection happens during C's atexit(), when Py_FinalizeEx()
  // already called.
  static PyMethodDef kAtexitMethodDef = {"e2e_atexit_callback",
                                         PythonHookContext::PyAtexitCallback,
                                         METH_NOARGS, nullptr};
  PyObject* atexit_mod = PyImport_ImportModule("atexit");
  if (atexit_mod == nullptr) {
    return false;
  }
  PyObject* py_fn = PyCFunction_New(&kAtexitMethodDef, nullptr);
  if (py_fn == nullptr) {
    Py_DECREF(atexit_mod);
    return false;
  }
  PyObject* res = PyObject_CallMethod(atexit_mod, "register", "O", py_fn);
  Py_DECREF(py_fn);
  Py_DECREF(atexit_mod);
  Py_XDECREF(res);
  return res != nullptr;
}

}  // namespace profiler
}  // namespace xla
