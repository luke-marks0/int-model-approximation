# Current Strategy

The repo supports three student paths:

- `codebook`: FP8 checkpoint linears run one Freivalds-checkable integer GEMM over exact FP8-codebook values, then deterministically rescale the product. This is the default path.
- `hawkeye`: FP8 checkpoint linears run direct Hawkeye integer replay of Hopper FP8 QGMMA accumulation. This can exactly match the Hopper FP8 teacher, but it is not cheaply Freivalds-checkable because it is not one matrix product.
- `hawkeye-class-counts`: FP8 checkpoint linears run exact Hawkeye replay from Freivalds-checkable class-count products. With `IMA_HAWKEYE_CLASS_CHUNK=512` and `IMA_HAWKEYE_PACKED_COUNT_LANES=6`, each K=32 Hawkeye group is one count product, giving exact Hopper QGMMA logits on the FP8-linears-only probe at 7,680 checkable products per Qwen2.5-0.5B forward.

Non-FP8 linears use the baseline per-row/per-token int32 GEMM path in all modes.

## Branch Notes

- `try/grouped-codebook-accumulator` was pruned. It split each FP8 codebook
  linear into Hawkeye-sized K=32 exact integer products and truncated the integer
  accumulator after each group. The natural 14-bit setting was much worse than
  `codebook` on the 128-token Hopper-teacher probe (top1 0.0234, top5 0.0750).
  Wider truncation only became usable when it was effectively the original
  codebook sum; the best sweep point kept top1 at 0.9141 with no meaningful top5
  improvement while increasing FP8-linears checks from 168 to 7,680.
- `try/hybrid-hawkeye-codebook` was pruned after the target was clarified to be
  one combined method rather than different methods for different layers. It did
  improve the main Hopper-teacher prompt from codebook top1/top5 0.9285/0.9215
  to 0.9431/0.9436, but it kept 77 FP8 linears on direct Hawkeye replay and only
  left 91 FP8 linears on the codebook proof shape.
- `try/codebook-hawkeye-correction` was kept as the first legitimate combined
  method. Each FP8 linear runs the normal codebook product plus Hawkeye-derived
  K=32 grouped correction products, then applies a fixed blend. On the
  2048-token Hopper-teacher probe, codebook top1/top5 0.9458/0.9291 moved to
  0.9482/0.9341, mean logit L2 fell from 94.37 to 93.30, total logit L2 from
  4592.71 to 4475.65, and DiFR mean from 0.0113 to 0.0100. P99 L2 regressed
  slightly, 206.59 to 210.59. The check cost is 7,849 exact integer products per
  forward on Qwen2.5-0.5B.
- `try/exact-hawkeye-class-counts` was kept as the high-accuracy construction.
  It composes every FP8 class pair inside a Hawkeye K=32 group into
  Freivalds-checkable class-count products, then deterministically replays the
  Hawkeye alignment and normalization logic from those counts. K=64 grouping was
  not viable because crossing Hawkeye's K=32 accumulator boundary broke exact
  agreement. Class chunk 512 with 6 packed weight-class lanes reaches the
  expected floor of one product per K=32 group: 7,680 checkable FP8-linear
  products on Qwen2.5-0.5B. On the 16-token FP8-linears-only Hopper probe,
  top1/top5 are 1.0000/1.0000 and logit L2 is 0.0, but runtime is high
  (about 208s for the student forward on H100).
- `try/global-hawkeye-aligned-codebook` was pruned. It kept the normal codebook
  product and added one globally exponent-aligned Hawkeye-style product per FP8
  linear, so verification cost was 336 products. The correction reduced some L2
  statistics but moved argmax decisions the wrong way on the 128-token Hopper
  FP8-only probe: best top1 was 0.9063 versus the codebook baseline 0.9141.
- `try/codebook-output-gain` was pruned. It kept exactly the codebook proof cost
  of 168 FP8-linear products and applied a deterministic scalar gain after each
  codebook GEMM. A sweep from 0.90 to 1.10 found no top1 improvement on the
  128-token Hopper FP8-only probe; gain 1.00 remained best/tied at 0.9141.
- `try/coarse-codebook-accumulation` was pruned. It split the codebook GEMM into
  coarse K chunks and rounded the accumulated partial output to bf16 after each
  chunk, hoping to approximate Hawkeye's periodic normalization at far below
  K=32 cost. Chunk sizes 2048, 1024, 512, 256, 128, and 64 all regressed top1 on
  the 128-token Hopper FP8-only probe; the best was 0.8984 at 3,840 products.
- `try/tail-class-counts` was pruned. It used codebook for most FP8 linears and
  exact class-count reconstruction in only the final transformer layer. The
  packed class-count replay hit a CUDA illegal-memory fault at 128 tokens, and
  the unpacked fallback was both expensive and worse: 15,059 products, top1
  0.9063 on the 128-token Hopper FP8-only probe.
- `try/codebook-dequant-blend` was kept as the best low-cost gain found so far.
  It blends the normal codebook product with one dequantized-weight int32
  product per FP8 linear. On an 11-row, 2048-token Hopper FP8-only probe,
  codebook top1/top5 0.9341/0.9342 moved to 0.9429/0.9377 at blend 0.05; mean
  logit L2 fell from 93.70 to 91.22 and p99 from 211.21 to 196.88. Check cost
  is 336 exact integer products, 2x codebook and far below class-counts.
