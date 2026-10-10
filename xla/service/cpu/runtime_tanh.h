

#ifndef XLA_SERVICE_CPU_RUNTIME_TANH_H_
#define XLA_SERVICE_CPU_RUNTIME_TANH_H_

#include <stdint.h>

extern "C" {

extern void __xla_cpu_runtime_Aarch64SveHyperbolicTangent(
    void* out, const void** in, const char* opaque, size_t opaque_len,
    void* status);
}

#endif  // XLA_SERVICE_CPU_RUNTIME_TOPK_H_
