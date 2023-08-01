"""Helpers for conditional Experimental backend/runtime compilation."""

def if_experimental(then, otherwise = []):
    return select({
        "//xla/mlir/backends/experimental:enabled": then,
        "//conditions:default": otherwise,
    })

def if_not_experimental(then, otherwise = []):
    return select({
        "//xla/mlir/backends/experimental:enabled": otherwise,
        "//conditions:default": then,
    })
