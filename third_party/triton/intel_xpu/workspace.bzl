"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "086f96a5fa91ab878c98359181706c526802c0a4"
XPU_TRITON_SHA256 = "41a7108d6277710c8cd381dde8c2fa48bcb454232a8b9071974e9a29c6a32fd4"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"
