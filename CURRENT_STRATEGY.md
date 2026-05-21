# Current Strategy

The repo supports two student paths:

- `codebook`: FP8 checkpoint linears run one Freivalds-checkable integer GEMM over exact FP8-codebook values, then deterministically rescale the product. This is the default path.
- `hawkeye`: FP8 checkpoint linears run direct Hawkeye integer replay of Hopper FP8 QGMMA accumulation. This can exactly match the Hopper FP8 teacher, but it is not cheaply Freivalds-checkable because it is not one matrix product.

Non-FP8 linears use the baseline per-row/per-token int32 GEMM path in both modes.

## Branch Notes

- `try/grouped-codebook-accumulator` was pruned. It split each FP8 codebook
  linear into Hawkeye-sized K=32 exact integer products and truncated the integer
  accumulator after each group. The natural 14-bit setting was much worse than
  `codebook` on the 128-token Hopper-teacher probe (top1 0.0234, top5 0.0750).
  Wider truncation only became usable when it was effectively the original
  codebook sum; the best sweep point kept top1 at 0.9141 with no meaningful top5
  improvement while increasing FP8-linears checks from 168 to 7,680.
- `try/hybrid-hawkeye-codebook` was kept. The useful selector was Hawkeye for
  the first 11 transformer layers and codebook for later FP8 linears. On the
  main Hopper-teacher prompt this moved codebook top1/top5 from 0.9285/0.9215 to
  0.9431/0.9436 and reduced mean logit L2 from 92.04 to 70.51. The cost tradeoff
  is not the final target: 91 FP8 linears stay single-product codebook checks,
  while 77 FP8 linears still use direct Hawkeye replay.
- `try/codebook-hawkeye-correction` was kept as the first legitimate combined
  method. Each FP8 linear runs the normal codebook product plus Hawkeye-derived
  K=32 grouped correction products, then applies a fixed blend. On the
  2048-token Hopper-teacher probe, codebook top1/top5 0.9458/0.9291 moved to
  0.9482/0.9341, mean logit L2 fell from 94.37 to 93.30, total logit L2 from
  4592.71 to 4475.65, and DiFR mean from 0.0113 to 0.0100. P99 L2 regressed
  slightly, 206.59 to 210.59. The check cost is 7,849 exact integer products per
  forward on Qwen2.5-0.5B.
