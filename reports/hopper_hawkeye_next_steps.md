# Hopper Hawkeye Next Steps

Date: 2026-05-20.

## Goal

Use a Hopper instance to test the version of the Hawkeye idea that should be
most favorable:

```text
simple FP8 Tensor-Core teacher
  -> public Hopper FP8 Hawkeye model
  -> Freivalds-checkable bucket/moment reconstruction
```

The L40S experiments in this branch showed that Hawkeye-style structure is
Freivalds-compatible, but our SM89/L40S scalar parameters are not a strong
match for the actual FP8 path. Hopper is different because Hawkeye's public
`gpu-simulator` repo includes a Hopper FP8 E4M3 model.

## Hypothesis

On Hopper, a custom teacher built directly from fixed FP8 QGMMA tiles should be
much better matched by the public Hawkeye FP8 model than either:

- L40S Triton `tl.dot`, or
- opaque `torch._scaled_mm` / cuBLAS.

Public Hawkeye Hopper FP8 constants from `badasherez/gpu-simulator`:

```text
zero_exponent = -139
internal_significand_width = 14
accumulator group = acc + 32 products
rounding = truncation / toward zero during alignment
```

## Experiments To Run On Hopper

### 1. Verify The Public Hawkeye FP8 Simulator

Clone/build Hawkeye's simulator on the Hopper box and run its FP8 test:

```bash
git clone https://github.com/badasherez/gpu-simulator /tmp/gpu-simulator
cd /tmp/gpu-simulator
python setup.py build_ext --inplace
pytest tests/ -k fp8 --hardware H100 -v -s
```

If the test passes, the public simulator is bit-exact for the raw Hopper FP8
QGMMA tile.

### 2. Re-run This Branch's L40S Probes On Hopper

From this repo:

```bash
uv run python scripts/probe_hawkeye_l40s_fp8.py \
  --m 16 --n 32 --k 896 \
  --groups 32 \
  --widths 14,15,16,17,18,24 \
  --seed 0

uv run python scripts/probe_hawkeye_bucket_products.py \
  --m 16 --n 32 --k 896 \
  --group 32 --width 14 \
  --seed 0
```

If Hopper behaves like the public model, `width=14` should be competitive or
best. On L40S, `width=14` was clearly too aggressive and `width=17` was the
weak empirical winner.

### 3. Build A Direct QGMMA Teacher

The clean teacher should not be cuBLAS. It should be a custom CUDA/PTX kernel
that directly invokes:

```text
wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3
```

Desired properties:

- real `torch.float8_e4m3fn` operands
- fixed `K=32` QGMMA steps
- fixed left-to-right K-tile order for `K=896` (`28` tiles)
- incoming accumulator passed as the QGMMA `C/D` operand
- scale multiplication and bf16 cast outside the QGMMA loop

The Hawkeye repo already contains a useful starting point:

```text
experiments/wgmma_e4m3/fp8_e4m3_wgmma.cu
```

### 4. Compare Four Teachers/Models

For random tensors and the first real Qwen FP8 layer:

```text
teacher A: torch._scaled_mm / cuBLAS
teacher B: direct fixed-QGMMA Hopper teacher
model C: Hawkeye scalar model, width=14
model D: Freivalds moment buckets, width=14
```

Expected outcomes:

- `B` should match `C` closely if the public Hopper model applies.
- `D` should match `C` closely if moment buckets remain a good compression.
- `A` may still differ from `B` because cuBLAS macro-kernel behavior is not
  necessarily the same as a fixed-QGMMA loop.

### 5. Promotion Criterion

Do not promote a strategy based only on random tile probes.

Promotion requires running the same corpus-style comparison used in the earlier
reports:

```text
0.5B 10-prompt corpus
top1/top5/logit_l2 against the selected teacher
```

If the target remains cuBLAS `_scaled_mm`, the direct-QGMMA teacher only helps
if it also tracks cuBLAS better. If the target can be verifier-defined, then
the direct-QGMMA teacher plus Freivalds bucket reconstruction may be a cleaner
reference than cuBLAS.

## Why This Matters

The L40S branch established the proof-shape result:

```text
Hawkeye-style FP8 accumulation can be represented by many ordinary integer
matmuls over exponent/significand buckets.
```

The open question is whether we can choose a Tensor-Core FP8 teacher that
Hawkeye models accurately enough to reduce the remaining FP8 error at the model
level. Hopper is the right place to test that because Hawkeye's public FP8 model
is Hopper-specific.
