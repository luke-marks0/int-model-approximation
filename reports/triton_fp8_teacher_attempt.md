# Simplified FP8 Tensor-Core Teacher Attempt

Date: 2026-05-20. Hardware: NVIDIA L40S, SM89.

## Question

Can we modify the teacher so Hawkeye has a cleaner target, while still requiring
FP8 matmuls to run on Tensor Cores?

The idea was to replace cuBLAS `torch._scaled_mm` with a simpler teacher whose
inner matmul is still FP8 Tensor Core work, but whose tiling and K-order are
fixed and visible.

Implementation: [`scripts/probe_triton_fp8_teacher.py`](../scripts/probe_triton_fp8_teacher.py).

## Teacher Tried

Custom Triton FP8 teacher:

```text
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 32
for k0 in 0..K step 32:
    acc = tl.dot(a_fp8[:, k0:k0+32], b_fp8[k0:k0+32, :], acc)
out = bf16(acc * scale_a * scale_b)
```

This uses `tl.dot` over `torch.float8_e4m3fn` operands, so the FP8 matmul work
is still on Tensor Cores. The purpose is to remove cuBLAS dispatcher and
macro-kernel ambiguity.

## Results

### K=32 Sanity

Shape: `M=16, N=16, K=32`, random tensors.

The Triton teacher exactly matched cuBLAS:

```text
triton_vs_cublas_teacher: bit_match = 1.0000
```

This confirms the custom teacher is using a compatible single-K-tile FP8 path.

Against that single tile, exact codebook / Hawkeye / moment buckets all had the
same result:

```text
bit_match = 0.9921875
```

So even in the clean K=32 case, the current Hawkeye scalar constants are not a
perfect L40S MMA model.

### K=896

Shape: `M=16, N=32, K=896`, random tensors.

```text
triton_vs_cublas_teacher: bit_match = 0.8671875
```

Against the Triton teacher:

| candidate | bf16 bit-match |
|---|---:|
| exact codebook | 0.861328 |
| Hawkeye scalar | 0.863281 |
| moment buckets | 0.863281 |

Against cuBLAS on the same inputs:

| candidate | bf16 bit-match |
|---|---:|
| exact codebook | 0.955078 |
| Hawkeye scalar | 0.962891 |
| moment buckets | 0.962891 |

So the simplified Triton teacher did **not** make Hawkeye fit better. It made
the teacher move away from both cuBLAS and the Hawkeye/codebook approximations.

## Chunked `_scaled_mm` Attempt

Also tried a teacher that splits K into 32-wide chunks:

```text
for each K chunk:
    partial = torch._scaled_mm(chunk, out_dtype=bf16)
sum partials in fp32
cast final sum to bf16
```

This still uses Tensor Cores for every FP8 chunk, but `_scaled_mm` only exposes
bf16 output for row-wise scaling on this stack. The per-chunk bf16 rounding is
too destructive:

```text
chunked_bf16_mm vs cuBLAS full K=896: bit_match = 0.533203
```

Not a viable teacher.

## Interpretation

The teacher-modification idea is valid, but these two simple versions do not
solve it.

What we learned:

1. For a single K=32 tile, Triton and cuBLAS agree, but our current Hawkeye
   scalar model is still not bit-exact on L40S.
2. For K=896, Triton's explicit left-to-right `tl.dot` loop is a different
   macro-kernel from cuBLAS and is not better matched by the Hawkeye scalar
   model.
3. Splitting into public `_scaled_mm` chunks fails because bf16 partial outputs
   round too early.

The remaining promising path is lower level: use a custom CUDA/PTX teacher that
directly invokes SM89 `mma.sync` FP8 instructions with the exact accumulator
operand pattern Hawkeye characterizes, or first run Hawkeye's characterization
suite on L40S FP8 to recover the true SM89 constants. Triton is too high-level
to guarantee the exact MMA accumulator semantics we want to study.
