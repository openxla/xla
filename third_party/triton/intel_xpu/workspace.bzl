"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "632df690e9daa4f2042be1eb45064a3393f45cc0"
XPU_TRITON_SHA256 = "fae203dfa12cecea3ce69a53830bdd9e5307fc064e00953832b72f5aeb6f5646"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"
