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
#include <optional>
#include <string>
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

const std::string& UnknownEventName() {
  static const std::string* const kUnknownEventName =
      new std::string("$<unknown>");
  return *kUnknownEventName;
}

void AddEventToXLine(const PythonTraceEntry& event,
                     tsl::profiler::XLineBuilder* line,
                     tsl::profiler::XPlaneBuilder* plane) {
  // TODO(jiesun): maybe add full filename as event stats.
  auto xevent = line->AddEvent(*plane->GetOrCreateEventMetadata(event.Name()));
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
  const char* event = PyUnicode_AsUTF8(args[1]);
  if (event == nullptr) {
    return nullptr;
  }

  PythonHooks::GetSingleton()->ProfileSlow(
      reinterpret_cast<PyFrameObject*>(args[0]), event, args[2]);
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

/*static*/ const std::string* PythonHookContext::InternName(
    EntryShard& shard, PyCodeObject* py_code_object) {
  if (py_code_object == nullptr) {
    return &UnknownEventName();
  }
  auto [it, inserted] = shard.code_names.try_emplace(py_code_object, nullptr);
  if (inserted) {
    Py_INCREF(py_code_object);
    it->second = &shard.names.emplace_back(
        GetEventName(py_code_object->co_filename, py_code_object->co_name,
                     py_code_object->co_firstlineno));
  }
  return it->second;
}

/*static*/ const std::string* PythonHookContext::InternName(
    EntryShard& shard, PyCFunctionObject* py_c_function) {
  if (py_c_function == nullptr) {
    return &UnknownEventName();
  }
  // Key on the method definition rather than the function object: bound
  // builtin methods are created per call, and holding a reference to them
  // would keep their `self` (e.g. device buffers) alive.
  PyMethodDef* method_def = py_c_function->m_ml;
  PyObject* module = py_c_function->m_module;
  auto [it, inserted] = shard.c_function_names.try_emplace(
      std::make_pair(method_def, module), nullptr);
  if (inserted) {
    Py_XINCREF(module);
    it->second = &shard.names.emplace_back(
        GetEventName((method_def != nullptr && method_def->ml_name != nullptr)
                         ? absl::string_view(method_def->ml_name)
                         : absl::string_view(),
                     module));
  }
  return it->second;
}

void PythonHookContext::ReleaseInternedObjects() {
  for (EntryShard& shard : entry_shards_) {
    absl::flat_hash_map<PyCodeObject*, const std::string*> code_names;
    absl::flat_hash_map<std::pair<PyMethodDef*, PyObject*>, const std::string*>
        c_function_names;
    {
#ifdef Py_GIL_DISABLED
      absl::MutexLock lock(shard.mu);
#endif  // Py_GIL_DISABLED
      code_names.swap(shard.code_names);
      c_function_names.swap(shard.c_function_names);
    }
    // Decrement outside of shard.mu: deallocation may run arbitrary code. The
    // order of the decrements doesn't matter.
    // NOLINTNEXTLINE
    for (const auto& [py_code_object, name] : code_names) {
      Py_DECREF(py_code_object);
    }
    // NOLINTNEXTLINE
    for (const auto& [key, name] : c_function_names) {
      Py_XDECREF(key.second);
    }
  }
}

PythonHooks* PythonHooks::GetSingleton() {
  static PythonHooks* const singleton = new PythonHooks;
  return singleton;
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
  if (Py_IsInitialized() && (options_.enable_python_traceme ||
                             options_.enable_trace_python_function)) {
    PyGILState_STATE gil_state = PyGILState_Ensure();
    if (options_.enable_trace_python_function) {
      if (!ClearProfilerInAllThreads()) {
        LOG(ERROR) << "Can't clear profiler in all threads.";
        PyErr_Print();
      }
      ReleaseInternedObjects();
    }
    if (options_.enable_python_traceme) {
      traceme_enabled = false;
    }
    PyGILState_Release(gil_state);
  }
  stopped_.store(true, std::memory_order_release);
}

std::vector<PerThreadConsumeData> PythonHookContext::Consume() {
  const bool stopped = stopped_.load(std::memory_order_acquire);
  if (!stopped && !Py_IsInitialized()) {
    return {};
  }

  const bool drain_incomplete = stopped && options_.include_incomplete_events;
  std::vector<std::pair<int64_t, PerThreadEvents>> entries;
  entries.reserve(16);

  {
#ifndef Py_GIL_DISABLED
    std::optional<PyGILState_STATE> gil_state;
    if (Py_IsInitialized()) {
      gil_state = PyGILState_Ensure();
    }
#endif  // !Py_GIL_DISABLED

    for (EntryShard& shard : entry_shards_) {
#ifdef Py_GIL_DISABLED
      absl::MutexLock lock(shard.mu);
#else
      if (gil_state.has_value()) {
        DCHECK(PyGILState_Check());
      }
#endif  // Py_GIL_DISABLED
      // NOLINTNEXTLINE
      for (auto& [thread_id, thread_events] : shard.entries) {
        if (thread_events.completed.empty() &&
            (!drain_incomplete || (thread_events.active.empty() &&
                                   thread_events.active_c.empty()))) {
          continue;
        }
        auto& [harvested_thread_id, harvested_events] = entries.emplace_back();
        harvested_thread_id = thread_id;
        std::swap(harvested_events.completed, thread_events.completed);
        if (drain_incomplete) {
          std::swap(harvested_events.active, thread_events.active);
          std::swap(harvested_events.active_c, thread_events.active_c);
        }
      }
      if (stopped) {
        shard.entries.clear();
      }
    }

#ifndef Py_GIL_DISABLED
    if (gil_state.has_value()) {
      PyGILState_Release(*gil_state);
    }
#endif  // !Py_GIL_DISABLED
  }

  const uint64_t now = tsl::profiler::GetCurrentTimeNanos();
  std::vector<PerThreadConsumeData> consumed_data;
  consumed_data.reserve(entries.size());
  for (auto& [thread_id, thread_events] : entries) {
    const size_t active_count =
        drain_incomplete
            ? (thread_events.active.size() + thread_events.active_c.size())
            : 0;

    PerThreadConsumeData thread_data;
    thread_data.thread_id = thread_id;
    thread_data.events.reserve(thread_events.completed.size() + active_count);
    for (const PythonTraceEntry& event : thread_events.completed) {
      thread_data.events.push_back(
          {std::string(event.Name()), event.start_time_ns, event.end_time_ns});
    }
    thread_events.completed.clear();

    if (drain_incomplete) {
      while (!thread_events.active.empty()) {
        const PythonTraceEntry& event = thread_events.active.top();
        thread_data.events.push_back(
            {std::string(event.Name()), event.start_time_ns, now});
        thread_events.active.pop();
      }
      while (!thread_events.active_c.empty()) {
        const PythonTraceEntry& event = thread_events.active_c.top();
        thread_data.events.push_back(
            {std::string(event.Name()), event.start_time_ns, now});
        thread_events.active_c.pop();
      }
    }
    consumed_data.push_back(std::move(thread_data));
  }

  return consumed_data;
}

void PythonHookContext::CollectData(tensorflow::profiler::XPlane* raw_plane) {
  if (raw_plane == nullptr) {
    end_to_end_xplane_.emplace();
    raw_plane = &*end_to_end_xplane_;
  }
  tsl::profiler::XPlaneBuilder plane(raw_plane);
  const uint64_t now = tsl::profiler::GetCurrentTimeNanos();
  for (EntryShard& shard : entry_shards_) {
#ifdef Py_GIL_DISABLED
    absl::MutexLock lock(shard.mu);
#endif  // Py_GIL_DISABLED
    // NOLINTNEXTLINE
    for (auto& [thread_id, thread_events] : shard.entries) {
      if (thread_events.completed.empty() &&
          (!options_.include_incomplete_events ||
           (thread_events.active.empty() && thread_events.active_c.empty()))) {
        continue;
      }
      const size_t active_count =
          options_.include_incomplete_events
              ? (thread_events.active.size() + thread_events.active_c.size())
              : 0;
      VLOG(1) << "Collecting " << thread_events.completed.size() << ":"
              << active_count << " events on thread " << thread_id;
      tsl::profiler::XLineBuilder line = plane.GetOrCreateLine(thread_id);
      if (line.NumEvents() == 0 && line.TimestampNs() == 0) {
        line.SetTimestampNs(start_timestamp_ns_);
      } else if (start_timestamp_ns_ <
                 static_cast<uint64_t>(line.TimestampNs())) {
        line.SetTimestampNsAndAdjustEventOffsets(start_timestamp_ns_);
      }
      for (const PythonTraceEntry& event : thread_events.completed) {
        AddEventToXLine(event, &line, &plane);
      }
      if (options_.include_incomplete_events) {
        while (!thread_events.active.empty()) {
          PythonTraceEntry& event = thread_events.active.top();
          event.end_time_ns = now;
          AddEventToXLine(event, &line, &plane);
          thread_events.active.pop();
        }
        while (!thread_events.active_c.empty()) {
          PythonTraceEntry& event = thread_events.active_c.top();
          event.end_time_ns = now;
          AddEventToXLine(event, &line, &plane);
          thread_events.active_c.pop();
        }
      }
    }
    shard.entries.clear();
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
      thread_traces.active.emplace(now, 0, InternName(shard, f_code));
#else   // PY_VERSION_HEX < 0x030b0000
      PyCodeObject* f_code = PyFrame_GetCode(frame);
      thread_traces.active.emplace(now, 0, InternName(shard, f_code));
      Py_XDECREF(f_code);
#endif  // PY_VERSION_HEX < 0x030b0000
      break;
    }
    case PyTrace_RETURN:
    case PyTrace_EXCEPTION: {
      if (!thread_traces.active.empty()) {
        PythonTraceEntry& entry = thread_traces.active.top();
        entry.end_time_ns = now;
        thread_traces.completed.push_back(entry);
        thread_traces.active.pop();
      } else if (options_.include_incomplete_events) {
#if PY_VERSION_HEX < 0x030b0000
        PyCodeObject* f_code = frame->f_code;
        thread_traces.completed.emplace_back(start_timestamp_ns_, now,
                                             InternName(shard, f_code));
#else   // PY_VERSION_HEX < 0x030b0000
        PyCodeObject* f_code = PyFrame_GetCode(frame);
        thread_traces.completed.emplace_back(start_timestamp_ns_, now,
                                             InternName(shard, f_code));
        Py_XDECREF(f_code);
#endif  // PY_VERSION_HEX < 0x030b0000
      }
      break;
    }
    case PyTrace_C_CALL: {
      if (PyCFunction_Check(arg)) {
        // Python stack does not have a filename/line_no for native calls.
        auto* func = reinterpret_cast<PyCFunctionObject*>(arg);
        thread_traces.active_c.emplace(now, 0, InternName(shard, func));
      }
      break;
    }
    case PyTrace_C_RETURN:
    case PyTrace_C_EXCEPTION: {
      if (PyCFunction_Check(arg)) {
        if (!thread_traces.active_c.empty()) {
          PythonTraceEntry& entry = thread_traces.active_c.top();
          entry.end_time_ns = now;
          thread_traces.completed.push_back(entry);
          thread_traces.active_c.pop();
        } else if (options_.include_incomplete_events) {
          // Only the end of the events is recorded, use profiler start as
          // start timestamp of the new event.
          auto* func = reinterpret_cast<PyCFunctionObject*>(arg);
          thread_traces.completed.emplace_back(start_timestamp_ns_, now,
                                               InternName(shard, func));
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
