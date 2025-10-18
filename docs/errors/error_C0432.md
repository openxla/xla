# Error code: C0432 - MismatchedRanks

The ranks of the two operands for the operation are not compatible.

## Erroneous code example

```c
p0 = f32[4,4] parameter(0)
p1 = f32[4,4,4] parameter(1)

f32[4,4] add(p0, p1)
```

## How to solve it

Ensure that both operands are the same rank.
