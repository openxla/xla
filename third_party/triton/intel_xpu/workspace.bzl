"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "1ef21febdddab6e0f49ec942c0c2ab250b2ab8ed"
XPU_TRITON_SHA256 = "9efe50aa2ae225fdcdcbc8323fe3f8293701178558ceff0012a080fde0979009"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"
