# First HLO pass: eliminate integer add-zero

This example implements a small `HloModulePass` that rewrites:

```text
add(x, constant-zero) -> x
```

The pass deliberately handles only integer element types. For floating-point
values, adding positive zero can change the sign of negative zero, so the same
rewrite is not always valid under strict IEEE semantics.

The implementation demonstrates four basic pieces of an HLO rewrite pass:

1. Traverse computations and instructions in post-order.
2. Match an HLO pattern with `hlo_matchers`.
3. Check an additional semantic condition on a literal.
4. Replace the matched instruction and report whether the module changed.

Run the test in the development container:

```bash
build_tools/dev_container/run.sh \
  bazel test --config=clang_local \
  //xla/examples/first_hlo_pass:first_hlo_pass_test \
  --test_output=errors
```
