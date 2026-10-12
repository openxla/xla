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

#include "xla/backends/cpu/codegen/builtin_symbols.h"

#include <cstdlib>

#ifdef _MSC_VER
#include <math.h>
#endif  // _MSC_VER

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/builtin_fp16.h"
#include "xla/backends/cpu/codegen/builtin_pow.h"

#ifdef MEMORY_SANITIZER
#include <sanitizer/msan_interface.h>
#endif  // MEMORY_SANITIZER

namespace xla::cpu {

//===----------------------------------------------------------------------===//
// A global static registry of builtin symbols available to XLA:CPU executables.
//===----------------------------------------------------------------------===//

using Registry = absl::flat_hash_map<std::string, void*>;

// Create a new registry of builtin runtime symbols by looking up the addresses
// of the symbols in the current process. Defined below.
static Registry CreateRegistry();

// Returns a global static registry of builtin runtime symbols.
static const Registry& StaticRegistry() {
  static absl::NoDestructor<Registry> registry(CreateRegistry());
  return *registry;
}

void* LookupBuiltinSymbol(absl::string_view name) {
  const Registry& registry = StaticRegistry();
  if (auto it = registry.find(name); it != registry.end()) {
    return it->second;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Create builtin runtime symbols registry for the current process.
//===----------------------------------------------------------------------===//

#if defined(PLATFORM_WINDOWS)
#ifdef __MINGW32__
#if defined(__i386__)
#undef _alloca
extern "C" void _alloca(void);
#elif defined(__x86_64__)
extern "C" void ___chkstk_ms(void);
#else
extern "C" void __chkstk(void);
#endif
#else
extern "C" void __chkstk(void);
#endif
#endif

extern "C" {
// Provided by compiler-rt and MLIR.
// Converts an F32 value to a BF16.
uint16_t __truncsfbf2(float);
// Converts an F64 value to a BF16.
uint16_t __truncdfbf2(double);

#if defined(__APPLE__) || defined(__s390x__)
// Converts an F32 value to a F16.
uint16_t __truncsfhf2(float);

float __extendhfsf2(uint16_t a);
#endif  // defined(__APPLE__) || defined(__s390x__)

}  // extern "C"

// MSVC does not have sincos[f].
#ifdef _MSC_VER

static void sincos(double x, double* sinv, double* cosv) {
  *sinv = sin(x);
  *cosv = cos(x);
}

static void sincosf(float x, float* sinv, float* cosv) {
  *sinv = sinf(x);
  *cosv = cosf(x);
}

#endif  // _MSC_VER

template <typename R, typename... Args>
static void* SymbolAddr(R (*func)(Args...)) {
  return reinterpret_cast<void*>(func);
}

// Register both the f32 (float) and f64 (double) versions of a libm symbol.
// Unfortunately the double versions are overloaded on some systems, e.g.
// Mac so we need an explicit cast. This requires passing the function signature
// for that case.
#define REGISTER_LIBM_SYMBOL(name, double_sig) \
  registry[#name "f"] = SymbolAddr(name##f);   \
  registry[#name] = SymbolAddr(static_cast<double_sig>(name));

static Registry CreateRegistry() {
  Registry registry;

  // Some platforms have overloaded memcpy, memmove, and memset, so we need to
  // specify the signature type to get the address of the specific function.
  registry["memcpy"] =
      SymbolAddr(static_cast<void* (*)(void*, const void*, size_t)>(memcpy));
  registry["memmove"] =
      SymbolAddr(static_cast<void* (*)(void*, const void*, size_t)>(memmove));
  registry["memset"] =
      SymbolAddr(static_cast<void* (*)(void*, int, size_t)>(memset));
  registry["malloc"] = SymbolAddr(static_cast<void* (*)(size_t)>(malloc));
  registry["free"] = SymbolAddr(static_cast<void (*)(void*)>(free));

  registry["__gnu_f2h_ieee"] = SymbolAddr(__gnu_f2h_ieee);
  registry["__gnu_h2f_ieee"] = SymbolAddr(__gnu_h2f_ieee);

  registry["__truncdfhf2"] = SymbolAddr(__truncdfhf2);
  registry["__truncdfbf2"] = SymbolAddr(__truncdfbf2);
  registry["__truncsfbf2"] = SymbolAddr(__truncsfbf2);

  registry["__powisf2"] = SymbolAddr(__powisf2);
  registry["__powidf2"] = SymbolAddr(__powidf2);

  REGISTER_LIBM_SYMBOL(acos, double (*)(double));
  REGISTER_LIBM_SYMBOL(acosh, double (*)(double));
  REGISTER_LIBM_SYMBOL(asin, double (*)(double));
  REGISTER_LIBM_SYMBOL(asinh, double (*)(double));
  REGISTER_LIBM_SYMBOL(atan, double (*)(double));
  REGISTER_LIBM_SYMBOL(atan2, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(atanh, double (*)(double));
  REGISTER_LIBM_SYMBOL(cbrt, double (*)(double));
  REGISTER_LIBM_SYMBOL(ceil, double (*)(double));
  REGISTER_LIBM_SYMBOL(copysign, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(cos, double (*)(double));
  REGISTER_LIBM_SYMBOL(cosh, double (*)(double));
  REGISTER_LIBM_SYMBOL(erf, double (*)(double));
  REGISTER_LIBM_SYMBOL(erfc, double (*)(double));
  REGISTER_LIBM_SYMBOL(exp, double (*)(double));
  REGISTER_LIBM_SYMBOL(exp2, double (*)(double));
  REGISTER_LIBM_SYMBOL(expm1, double (*)(double));
  REGISTER_LIBM_SYMBOL(fabs, double (*)(double));
  REGISTER_LIBM_SYMBOL(fdim, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(floor, double (*)(double));
  REGISTER_LIBM_SYMBOL(fma, double (*)(double, double, double));
  REGISTER_LIBM_SYMBOL(fmax, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(fmin, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(fmod, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(frexp, double (*)(double, int*));
  REGISTER_LIBM_SYMBOL(hypot, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(ilogb, int (*)(double));
  REGISTER_LIBM_SYMBOL(ldexp, double (*)(double, int));
  REGISTER_LIBM_SYMBOL(lgamma, double (*)(double));
  REGISTER_LIBM_SYMBOL(llrint, long long (*)(double));   // NOLINT
  REGISTER_LIBM_SYMBOL(llround, long long (*)(double));  // NOLINT
  REGISTER_LIBM_SYMBOL(log, double (*)(double));
  REGISTER_LIBM_SYMBOL(log10, double (*)(double));
  REGISTER_LIBM_SYMBOL(log1p, double (*)(double));
  REGISTER_LIBM_SYMBOL(log2, double (*)(double));
  REGISTER_LIBM_SYMBOL(logb, double (*)(double));
  REGISTER_LIBM_SYMBOL(lrint, long (*)(double));   // NOLINT
  REGISTER_LIBM_SYMBOL(lround, long (*)(double));  // NOLINT
  REGISTER_LIBM_SYMBOL(modf, double (*)(double, double*));
  REGISTER_LIBM_SYMBOL(nan, double (*)(const char*));
  REGISTER_LIBM_SYMBOL(nearbyint, double (*)(double));
  REGISTER_LIBM_SYMBOL(nextafter, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(nexttoward, double (*)(double, long double));
  REGISTER_LIBM_SYMBOL(pow, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(remainder, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(remquo, double (*)(double, double, int*));
  REGISTER_LIBM_SYMBOL(rint, double (*)(double));
  REGISTER_LIBM_SYMBOL(round, double (*)(double));
  REGISTER_LIBM_SYMBOL(scalbln, double (*)(double, long));  // NOLINT
  REGISTER_LIBM_SYMBOL(scalbn, double (*)(double, int));
  REGISTER_LIBM_SYMBOL(sin, double (*)(double));
  REGISTER_LIBM_SYMBOL(sinh, double (*)(double));
  REGISTER_LIBM_SYMBOL(sqrt, double (*)(double));
  REGISTER_LIBM_SYMBOL(tan, double (*)(double));
  REGISTER_LIBM_SYMBOL(tanh, double (*)(double));
  REGISTER_LIBM_SYMBOL(tgamma, double (*)(double));
  REGISTER_LIBM_SYMBOL(trunc, double (*)(double));

#ifdef __APPLE__
  REGISTER_LIBM_SYMBOL(__sincos, void (*)(double, double*, double*));
  registry["__sincosf_stret"] = SymbolAddr(__sincosf_stret);
  registry["__sincos_stret"] = SymbolAddr(__sincos_stret);
#else
  REGISTER_LIBM_SYMBOL(sincos, void (*)(double, double*, double*));
#endif

#undef REGISTER_LIBM_SYMBOL

#if defined(__APPLE__) || defined(__s390x__)
  registry["__truncsfhf2"] = SymbolAddr(__truncsfhf2);
  registry["__extendhfsf2"] = SymbolAddr(__extendhfsf2);
#endif

#ifdef __APPLE__
  registry["__bzero"] = SymbolAddr(bzero);
  registry["bzero"] = SymbolAddr(bzero);
  registry["memset_pattern16"] = SymbolAddr(memset_pattern16);
#endif

#if defined(PLATFORM_WINDOWS)

#ifdef __MINGW32__
#if defined(__i386__)
  registry["__chkstk"] = SymbolAddr(_alloca);
#elif defined(__x86_64__)
  registry["__chkstk"] = SymbolAddr(___chkstk_ms);
#else
  registry["__chkstk"] = SymbolAddr(__chkstk);
#endif
#else
  registry["__chkstk"] = SymbolAddr(__chkstk);
#endif

#endif

#ifdef MEMORY_SANITIZER
  registry["__msan_unpoison"] = SymbolAddr(__msan_unpoison);
#endif

  return registry;
}

}  // namespace xla::cpu
