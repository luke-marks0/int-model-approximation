# Current Strategy

The repo supports two student paths:

- `codebook`: FP8 checkpoint linears run one Freivalds-checkable integer GEMM over exact FP8-codebook values, then deterministically rescale the product. This is the default path.
- `hawkeye`: FP8 checkpoint linears run direct Hawkeye integer replay of Hopper FP8 QGMMA accumulation. This can exactly match the Hopper FP8 teacher, but it is not cheaply Freivalds-checkable because it is not one matrix product.

Non-FP8 linears use the baseline per-row/per-token int32 GEMM path in both modes.
