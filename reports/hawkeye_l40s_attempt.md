# Hawkeye/L40S FP8 accumulation attempt

Date: 2026-05-20. Hardware: NVIDIA L40S, SM89.

## Goal

Use Hawkeye's Tensor Core accumulation model as a concrete hypothesis for the
remaining `torch._scaled_mm` mismatch on L40S/Ada, then decide whether that
should revise the integer-kernel strategy.

The experiment is in [`scripts/probe_hawkeye_l40s_fp8.py`](../scripts/probe_hawkeye_l40s_fp8.py).
It compares real `torch._scaled_mm(..., out_dtype=bf16)` against:

- exact FP8-codebook integer summation, the current `alpha=32/32` endpoint
- Hawkeye-style grouped fixed-point summation with configurable product group
  size and internal significand width

## Paper signal used

Hawkeye reports:

- Lovelace follows Ampere's accumulation structure for the characterized
  formats.
- The public simulator exposes architecture constants: Ampere-style group
  size is 9 elements (`acc + 8 products`), Hopper-style FP8 E4M3 is 33
  elements (`acc + 32 products`) with 14-bit internal significand.
- Integration into model architectures is left as future work.

Because this repo targets L40S, the probe sweeps both Ada/Ampere-shaped
grouping (`8` products per group) and Hopper-FP8-shaped grouping (`32`
products per group), with internal widths around the 14-bit public FP8 model
and wider values.

## Random K=896 probe

Shape: `M=16, N=32, K=896`, random BF16 activations and random FP8 weights.
Ten seeds.

| candidate | mean bf16 bit-match | min | max | mean abs error |
|---|---:|---:|---:|---:|
| exact codebook / width 24 | 0.956250 | 0.929688 | 0.968750 | 0.001873 |
| group 8, width 17 | 0.958203 | 0.941406 | 0.968750 | 0.001688 |
| group 32, width 17 | 0.958203 | 0.935547 | 0.970703 | 0.001814 |
| group 8, width 14 | 0.871289 | 0.837891 | 0.894531 | 0.009136 |
| group 32, width 14 | 0.878516 | 0.853516 | 0.902344 | 0.012677 |

The 14-bit Hopper-FP8 constant is too aggressive on L40S. A 17-bit internal
width is the only consistent improvement in this probe, and the improvement is
small: about +0.20 percentage points of bf16 bit-match.

## Real Qwen FP8 layer probe

Layer: `model.layers.0.self_attn.q_proj` from
`RedHatAI/Qwen2.5-0.5B-FP8-dynamic`, shape `M=16, N=896, K=896`, random BF16
activations. Five seeds.

| candidate | seed bit-matches | mean |
|---|---:|---:|
| exact codebook / width 24 | 0.95466, 0.94985, 0.95159, 0.94845, 0.95299 | 0.95151 |
| group 32, width 17 | 0.95550, 0.95215, 0.95096, 0.94992, 0.95571 | 0.95285 |

Mean absolute error also improved slightly for `group 32, width 17` on every
seed in this five-seed real-layer probe.

`group 8, width 17` sometimes improved bit-match, but also produced rare huge
outliers in the current prototype. That makes it a bad promotion candidate
until the scalar model is audited against Hawkeye's C++ simulator line by line.

## Freivalds impact

This does not directly preserve the current proof shape.

The active codebook path is checkable because it is one exact integer matrix
product:

```text
(a_fp8 * 512) @ (b_fp8 * 512).T
```

Hawkeye-style accumulation chooses a max exponent per output element and group,
then shifts/truncates each product relative to that max. That alignment depends
on both operands jointly:

```text
max_k exponent(a_i,k * b_j,k)
```

So the aligned terms are not simply `A' @ B'` for committed matrices `A'` and
`B'`. A verifier could still check such a computation, but not with the repo's
current single Freivalds matrix-product abstraction. It would need either:

- a new proof shape for per-output product alignment, or
- a verifier-defined non-cuBLAS teacher such as `int_fp8_codebook`.

## Recommendation

Do not change `CURRENT_STRATEGY.md` yet.

The best L40S Hawkeye-inspired candidate found here is:

```text
products_per_group = 32
internal_significand_width = 17
zero_exponent = -139
rounding = truncation / toward zero during alignment
```

It is a real signal, but too small and too structurally awkward to promote as
the active integer-kernel strategy. The right next engineering step, if we want
to chase cuBLAS FP8 bytes on L40S, is a focused implementation branch:

1. Validate this Python model against Hawkeye's C++ simulator on single-tile
   FP8 cases.
2. Add scalar golden tests for the `group 32, width 17` path.
3. Prototype a Triton output-tile kernel for the grouped accumulator.
4. Measure full-corpus top1/logit L2 against the FP8 hardware teacher.
5. Decide whether the proof system can tolerate a non-matrix-product alignment
   check.

Until then, this remains an experimental lead, not a strategy replacement.
