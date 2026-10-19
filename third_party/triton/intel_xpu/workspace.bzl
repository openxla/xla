"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "50ad141e46747150f8056751f2841b2ee378bef6"
XPU_TRITON_SHA256 = "459ef953f0e4f590338115c181bda22577a10aa38f7cebf5e3f902795400c71d"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"
