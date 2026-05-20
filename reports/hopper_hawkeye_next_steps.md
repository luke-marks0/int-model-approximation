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

Repo implementation:

```bash
uv run python scripts/probe_hopper_qgmma_hawkeye.py \
  --m 16 --n 32 --k 896 \
  --group 32 --width 14 \
  --seed 0

uv run python scripts/probe_hopper_qgmma_hawkeye.py \
  --m 16 \
  --group 32 --width 14 \
  --seed 0 \
  --real-weight
```

The direct-QGMMA teacher is implemented as a JIT-built PyTorch CUDA extension in
`scripts/hopper_qgmma/`. The end-to-end command runner for steps 1-4 is:

```bash
uv run python scripts/run_hopper_hawkeye_experiment.py
```

The model entrypoint can also use the direct-QGMMA teacher instead of cuBLAS:

```bash
IMA_TEACHER_KERNEL=hopper_qgmma uv run python -m int_model_approximation
```

This is intentionally non-default. The current teacher composes the public
Hawkeye 64x128x32 tile through Python loops, so it is useful for correctness
experiments but not yet a fast full-model teacher.

### 4a. Integer Error Against The QGMMA Teacher

To check whether the selected teacher lowers the integerized-layer error:

```bash
uv run python scripts/probe_hopper_qgmma_integer_error.py \
  --real-weight \
  --m 16 \
  --seed 0 \
  --alphas 0,10/32,17/32,32/32
```

First Qwen FP8 layer result on H100, shape `M=16, N=896, K=896`:

| alpha | mean abs vs cuBLAS | mean abs vs QGMMA |
|---:|---:|---:|
| 10/32 | 0.0258826 | 0.0258765 |
| 32/32 | 0.0026531 | 0.0027501 |

Changing only the teacher from cuBLAS to direct QGMMA barely moves the current
`10/32` layer error. The lower-error setting in this isolated probe is the
codebook endpoint (`32/32`), so any promotion should evaluate
`IMA_TEACHER_KERNEL=hopper_qgmma IMA_CODEBOOK_NUM=32` on the full corpus once
the QGMMA teacher path is fast enough.

### 4b. Exact Freivalds-Checkable Hawkeye Student

Implementation:

```bash
uv run python scripts/probe_hopper_hawkeye_freivalds_student.py \
  --m 16 --n 32 --k 896 \
  --group 32 --width 14 \
  --class-chunk 32 \
  --seed 0

uv run python scripts/probe_hopper_hawkeye_freivalds_student.py \
  --real-weight \
  --m 16 \
  --group 32 --width 14 \
  --class-chunk 32 \
  --seed 0
```

This is the exact class-pair construction, but class pairs are batched into
larger ordinary integer products:

```text
(class_chunk * M) x K_group  @  K_group x (class_chunk * N)
```

Each product is an exact `int32 x int32 -> int64` matrix product and can be
checked with Freivalds. The post-matmul work is deterministic Hawkeye replay.

Results on H100:

| case | checkable products | student vs Hawkeye scalar | student vs QGMMA |
|---|---:|---:|---:|
| random `16x32x896` | 896 | 1.0000 bit-match | 1.0000 bit-match |
| Qwen q_proj `16x896x896` | 1,554 | 1.0000 bit-match | 0.99986 bit-match |

This answers the proof-shape question directly: a Hawkeye FP8 Tensor-Core
teacher can be reconstructed by a Freivalds-checkable integer student, using
many ordinary checked matmuls plus deterministic side logic. The remaining
`0.99986` real-layer gap is the scalar Hawkeye-vs-direct-QGMMA gap for that
input, not a Freivalds reconstruction error.

The exact reconstruction is also available as a slow, non-default model student:

```bash
IMA_TEACHER_KERNEL=hopper_qgmma \
IMA_STUDENT_KERNEL=hawkeye_exact \
IMA_HAWKEYE_CLASS_CHUNK=32 \
  uv run python -m int_model_approximation
```

The active Freivalds products in this path are the same ordinary integer
matmuls used in the probe; non-FP8 linears keep the existing int32 path.

One full-model failure mode showed up in the first end-to-end pass: a
zero-significand Hawkeye accumulator used the synthetic zero exponent `-139`,
and the first `_gfloat_to_float32` helper masked that negative exponent field
with `& 0xFF`. That decoded exact zero as a huge finite BF16 value, which then
overflowed when the model cast the linear output back to FP16. The package and
script helpers now decode zero-significand gfloats as exact zero and clamp the
constructed exponent field instead of wrapping it.

Post-fix one-prompt smoke result on Dolly row 0, 16 tokens:

| metric | value |
|---|---:|
| QGMMA teacher forward | 2.62 s |
| Hawkeye exact student forward | 288.36 s |
| checkable integer products | 252,299 |
| finite teacher logits | 100% |
| finite student logits | 100% |
| top1 similarity | 0.9375 |
| top5 similarity | 0.9250 |
| logit L2 mean | 99.9468 |

The first working implementation computed the per-group max exponent with a
first pass over class-count products. That was exact but redundant: the max
product exponent is deterministic from committed FP8 exponent/nonzero fields.
The current helper computes that max directly and leaves only the replay
class-count products as Freivalds-checkable integer matmuls. On the same
16-token smoke run, this dropped student time from 530.59 s to 288.36 s and
checked products from 504,597 to 252,299, without changing logits.

Packed class-count products are also implemented behind
`IMA_HAWKEYE_PACKED_COUNT_LANES`. Packing uses base-64 lanes, so up to six
weight-side class counts can share one int32 product column without carry
because each K group contributes at most 32 products per lane. This is still
Freivalds-checkable: the packed output is one ordinary integer matrix product,
and unpacking is deterministic replay.

On the random `16x32x896` probe, `--packed-count-lanes 6` reduced checked
products from 448 to 111 and still matched direct QGMMA bit-for-bit. On the real
Qwen q_proj probe it reduced checked products to 222 while preserving the same
student-vs-Hawkeye bit match. In full-model timing, the current PyTorch
unpacking implementation is mixed:

| prompt tokens | lanes | student time | checked products |
|---:|---:|---:|---:|
| 1 | 1 | 19.30 s | 53,469 |
| 1 | 6 | 14.95 s | 14,017 |
| 16 | 1 | 288.36 s | 252,299 |
| 16 | 6 | 392.94 s | 66,208 |

The model default remains `IMA_HAWKEYE_PACKED_COUNT_LANES=1` because the packed
path uses much larger intermediate unpack tensors during prompt prefill. It is
worth revisiting with a fused unpack-and-accumulate Triton kernel; the product
count reduction is real, but the current tensor replay pays it back in memory
traffic for multi-token prompts.

An opt-in first fused replay kernel is available through
`IMA_HAWKEYE_FUSED_REPLAY=1` / `--fused-replay`. It fuses the lane-1
`counts * aligned_value` contribution into one Triton kernel per class-count
product while leaving the Freivalds-checkable count products unchanged. It is
bit-exact on the random QGMMA probe, but it is slower in full-model timing:

| prompt tokens | replay | student time | checked products |
|---:|---|---:|---:|
| 1 | default tensor replay | 19.30 s | 53,469 |
| 1 | fused contribution replay | 151.97 s | 53,469 |

The reason is launch granularity. This fused kernel adds one Triton launch per
class-count product, so it saves tensor materialization but loses badly to many
small launches. A useful fused implementation needs to combine the count-product
kernel and replay, or fuse replay across many count products/groups per launch.

### 4c. Kernel-Design Handoff Notes

Several follow-up kernel designs were tried after the first fused replay result.
All student paths described here keep the expensive work Freivalds-checkable:
each checked object is still an ordinary integer matrix product over committed
mask operands, followed by deterministic Hawkeye replay.

The most useful positive result was an opt-in int8 count-product backend:

```bash
uv run python scripts/eval_hopper_hawkeye_one_prompt.py \
  --teacher-kernel hopper_qgmma \
  --student-kernel hawkeye_exact \
  --max-tokens 1 \
  --packed-count-lanes 1 \
  --count-matmul int8
```

This encodes the class masks as `int8` 0/1 matrices and uses Triton `tl.dot`.
For `K_group=32`, the count is exact in int32 because the largest lane-1 count
is 32. The product is still Freivalds-checkable as an integer matmul; the only
change is the operand dtype and tensor-core implementation.

Warm one-token results on Dolly row 0:

| backend | extra env | student time | checked products | logits |
|---|---|---:|---:|---|
| scalar int32->int64 count | default | 19.09 s | 53,469 | same as exact baseline |
| int8 `tl.dot`, block N=64 | `--count-matmul int8` | 18.48 s | 53,469 | same as exact baseline |
| int8 `tl.dot`, block N=128 | `IMA_INT8_COUNT_BLOCK_N=128 --count-matmul int8` | 18.21 s | 53,469 | same as exact baseline |

The first run for a new tile shape is misleading because Triton compiles many
specialized kernels across the model's projection shapes. For example, block
N=128 took 200.12 s cold and 18.21 s warm. A block N=256 cold run was started
but killed before completion because the instance had to shut down; there is no
valid block-N=256 timing yet.

Negative results from the same pass:

| design | one-token result | conclusion |
|---|---:|---|
| weight-side mask cache inside each K group | 75.38 s, 53,469 products | Correct but slower; cached `B` flats add memory pressure and do not pay off at one token. It is gated by `IMA_HAWKEYE_CACHE_WEIGHT_CHUNKS=1`. |
| group size 64 | 196.88 s, 56,747 products, top1 mismatch | Larger groups reduce random-probe class products but break the exact QGMMA grouping assumption and are much slower in full model. |
| class chunk 64 | 154.91 s, 29,377 products | Fewer products, but much larger replay/count tensors dominate. |
| custom int32-output count kernel | 159.67 s, 53,469 products | Correct but far slower than the existing scalar int64 count kernel after warmup. |
| fused count+replay with atomics | 186.21 s | Correct on probes, but atomics and launch overhead dominate. The probe counter undercounts this path because it bypasses `_Int32KernelProbe`. |

The main observation for the next developer: launch count is not the only
problem. Designs that reduce checked product count often increase tensor shape,
memory traffic, atomics, or compilation cost enough to lose badly. The best
near-term path looks like keeping the exact `K_group=32` Hawkeye grouping and
making the count products use true int8 tensor-core MMA, then fusing replay at a
larger scheduling granularity than "one Triton replay launch per class product".
The int8 backend is small but real progress and should be re-tested on 16 or 32
tokens before any default change.

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
