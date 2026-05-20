# Hawkeye-informed Freivalds design

Date: 2026-05-20.

## Short answer

Yes. If the budget allows an arbitrary number of Freivalds-checkable integer
matmuls, Hawkeye can define a checkable FP8 Tensor Core emulator.

The earlier limitation was only about the current proof shape:

```text
one or two ordinary A @ B products + deterministic blend
```

Hawkeye-style accumulation is not one ordinary product, because each output
element and K-group chooses a max exponent from the products in that output
cell. But that joint dependence can be exposed by bucketing the FP8 operands
and checking many ordinary integer products.

## Exact bucketed construction

For a K-group, e.g. 32 FP8 products, decode each nonzero FP8 operand into:

```text
class = (effective_exponent, signed_significand)
```

For e4m3fn this gives at most 254 nonzero classes:

- effective exponent `1..15`
- signed significand values from the raw e4m3 significand, with subnormals
  left denormalized, matching Hawkeye's public FP8 simulator

For every activation class `u` and weight class `v`, build indicator matrices:

```text
A_u[i,k] = 1 if activation a[i,k] is class u else 0
B_v[j,k] = 1 if weight     b[j,k] is class v else 0
```

Then Freivalds-check:

```text
C_{u,v} = A_u @ B_v.T
```

Each `C_{u,v}[i,j]` is the count of K positions where output `(i,j)` saw that
exact FP8 class pair.

Given all checked `C_{u,v}`, the verifier can replay Hawkeye's grouped
accumulation deterministically:

```text
product_exp(u,v) = exp(u) + exp(v) - 14
product_sig(u,v) = signed_sig(u) * signed_sig(v)

E[i,j] = max(
  incoming_accumulator_exp[i,j],
  max product_exp(u,v) where C_{u,v}[i,j] > 0
)

aligned_sum[i,j] =
  aligned_incoming_accumulator[i,j]
  + sum_{u,v} C_{u,v}[i,j] *
      trunc_towards_zero((product_sig(u,v) << base_shift)
                         >> (E[i,j] - product_exp(u,v)))
```

Then normalize the integer sum into the next Hawkeye accumulator state and
continue to the next K-group. Final scale multiplication and bf16 cast are
ordinary deterministic postprocessing.

This is exact for the chosen Hawkeye scalar model. Every expensive product is
an integer matrix product, hence Freivalds-checkable. The max, shifts,
truncation, normalization, and bf16 cast are side logic, not hidden GEMM.

## Cost

The naive exact construction is expensive:

```text
254 * 254 = 64,516 checked count-products per K-group
896 / 32 = 28 groups
~1.8M checked products per FP8 layer
```

That is not a practical first implementation, but it proves the key point:
Hawkeye's joint max-exponent behavior does not make Freivalds impossible. It
just changes the product basis from "one dense product" to "many bucket-count
products plus deterministic replay."

## Compression routes

The exact count table is overkill. Hawkeye gives structure we can exploit.

### 1. Exponent-pair products

First compute product-exponent occupancy with only exponent buckets:

```text
Count_{ea,eb} = A_exp_ea @ B_exp_eb.T
```

There are at most `15 * 15 = 225` such products per K-group. These identify
the group max exponent `E` for each output, except for the incoming accumulator
comparison.

### 2. Signed significand moment products

For contribution values, compute signed significand moments:

```text
Moment_{ea,eb} =
  (signed_sig(A) where exp=ea) @ (signed_sig(B) where exp=eb).T
```

Another `225` products. This is exact whenever the Hawkeye shift does not drop
bits from individual products before summing. It becomes an approximation when
per-product truncation happens before summation.

This is the closest analogue to the current two-matmul strategy: it is still a
small set of normal integer products whose outputs feed deterministic
postprocessing.

### 3. Low-rank lookup decomposition

For exact per-product truncation, the coefficient table

```text
f_E(sig_a, sig_b) =
  trunc((sig_a * sig_b) << base_shift >> (E - product_exp))
```

is only a tiny `16 x 16` integer lookup table for each `(E, ea, eb)`. Any such
table can be decomposed into a sum of rank-1 integer/rational factors:

```text
f_E(sig_a, sig_b) = sum_r U_r(sig_a) * V_r(sig_b)
```

Each rank-1 term is one Freivalds-checkable matmul:

```text
(U_r(sig(A)) where exp=ea) @ (V_r(sig(B)) where exp=eb).T
```

This trades the huge 254-class count table for many fewer structured products.
The exact rank and coefficient growth need to be measured, but the upper bound
is small because the table is only `16 x 16`.

### 4. Hawkeye-informed correction basis

The most practical near-term experiment is not an exact emulator. It is a
correction basis derived from the L40S Hawkeye residual:

```text
Y = Y_high
  + alpha * (Y_codebook - Y_high)
  + sum_t beta_t * BucketProduct_t
```

Useful `BucketProduct_t` candidates:

- high-product-exponent occupancy
- signed significand moments by product exponent
- cancellation indicators: high absolute product mass but small signed moment
- low-bit residual moments for the `group=32, width=17` L40S candidate

Every `BucketProduct_t` is still an ordinary integer matmul. The coefficients
can be fixed dyadic constants as in the current strategy.

## Recommendation

The next implementation branch should not try to drop a full Hawkeye emulator
into `Int32Linear` immediately. A better staged plan:

1. Build a single-group scalar golden model for `group=32, width=17`.
2. Add a bucket-product probe that computes exponent occupancy and signed
   significand moments via ordinary integer products.
3. Compare:

```text
exact codebook
Hawkeye scalar candidate
exponent/moment bucket reconstruction
dyadic bucket correction on top of current alpha=10/32
```

4. Promote only if the bucket correction beats the current `alpha=10/32`
   strategy on corpus, not just on random layer probes.

If exact cuBLAS-byte matching becomes mandatory, use the full class-pair or
low-rank lookup construction. If the goal remains a cheap ZKP-friendly model,
start with the compressed correction basis.

## Experiments Run

Implementation: [`scripts/probe_hawkeye_bucket_products.py`](../scripts/probe_hawkeye_bucket_products.py).

Candidate used:

```text
products_per_group = 32
internal_significand_width = 17
zero_exponent = -139
```

This is the best L40S candidate from
[`hawkeye_l40s_attempt.md`](hawkeye_l40s_attempt.md).

### Exact Class-Pair Sanity

Small random shape: `M=8, N=16, K=64`, two K-groups.

The exact class-pair construction matched the scalar Hawkeye candidate
byte-for-byte:

```text
class_pair_exact vs hawkeye_scalar: bit_match = 1.0000
```

Dynamic checked products in this small run:

```text
15,129 class-pair count products/group
30,258 total count products
405,000 static upper-bound products
```

This confirms the exact decomposition is viable in principle.

### Moment-Bucket Probe, Random K=896

Shape: `M=16, N=32, K=896`, ten random seeds.

Products per K-group:

```text
225 exponent-pair count products
225 signed-significand moment products
450 total products/group
28 groups
12,600 Freivalds-checkable products/layer
```

Mean bf16 bit-match against `torch._scaled_mm`:

| candidate | mean bit-match |
|---|---:|
| exact codebook | 0.95625 |
| Hawkeye scalar | 0.95820 |
| moment buckets | 0.95879 |

Moment buckets matched the Hawkeye scalar candidate at about `0.99883`
mean bf16 bit-match. In most seeds they were byte-identical; the rare
differences are from the moment compression moving truncation after summing
within an exponent pair.

### Moment-Bucket Probe, Real Qwen Layer

Layer: `model.layers.0.self_attn.q_proj`,
`RedHatAI/Qwen2.5-0.5B-FP8-dynamic`, shape `M=16, N=896, K=896`, five random
activation seeds.

Mean bf16 bit-match against `torch._scaled_mm`:

| candidate | mean bit-match |
|---|---:|
| exact codebook | 0.95151 |
| Hawkeye scalar | 0.95285 |
| moment buckets | 0.95280 |

Moment buckets matched the Hawkeye scalar candidate at about `0.99869`
mean bf16 bit-match.

### Interpretation

The Freivalds-compatible moment basis successfully captures almost all of the
Hawkeye scalar candidate while using `12,600` ordinary checked products per
layer for Qwen-style `K=896`.

That is a real proof-shape result: Hawkeye can inform a Freivalds-checkable
integer kernel. It is not yet a strategy replacement, because the measured
layer-level gain is small and has not been validated end-to-end on corpus
top1/logit L2. The next promotion-grade experiment is to implement the
moment-bucket outputs in the model path behind an env flag and run the same
10-prompt corpus comparison used by the earlier reports.
