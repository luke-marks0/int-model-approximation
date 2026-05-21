# int-model-approximation

DiFR layer-error measurement for a real quantized Hugging Face model.

The ultimate goal is to build a low-error integer version of a production FP8
teacher whose matmuls are cheap to check. Proving the production float/FP8
computation directly is expensive: the matmuls do not give the clean algebraic
product that Freivalds-style checks need, and trying to prove float behavior
tends to pull in costly range checks, rounding semantics, tolerance windows, and
implementation-specific kernel state.

This repo studies two useful endpoints. The `codebook` path keeps the cheap
Freivalds proof shape, but has high error against the genuine FP8 teacher. The
`hawkeye` path is a perfect recreation of the Hopper FP8 teacher, but is not
cheaply checkable. The research target is to unify the benefits of both ideas:
the low error of Hawkeye with the cheap checkability of a single integer product.
The current experimental `hawkeye-class-counts` path sits between those
endpoints: it reconstructs Hawkeye exactly from Freivalds-checkable class-count
products, but still needs many products per model forward.

The supported run loads `RedHatAI/Qwen2.5-0.5B-FP8-dynamic`, executes a real FP8
reference forward, builds an integerized copy, and writes layer and logit error
metrics. There are no fake-quant, training, sweep, or alternate model paths.

## Supported student paths

Choose the student with `IMA_STUDENT_KERNEL`:

- `codebook` (default): FP8 checkpoint linears are replaced by one Triton
  `int32 x int32 -> int64` product over exact FP8-codebook integer values. The
  activation side is dynamically quantized to FP8 per token, converted to the
  integer FP8 codebook, multiplied by the committed integer FP8 weights, and
  deterministically rescaled. This is the proof-friendly path: each FP8 linear
  is a single ordinary integer matrix product, so Freivalds' check applies
  directly. Its downside is accuracy: because it verifies a codebook-defined
  product rather than NVIDIA's FP8 Tensor Core accumulator, its error against the
  genuine FP8 teacher is high.

- `hawkeye`: FP8 checkpoint linears are replaced by direct Hawkeye integer replay
  of Hopper FP8 QGMMA accumulation, using K=32 groups by default. This is the
  hardware-faithful path: with the Hopper QGMMA teacher it exactly matches the
  genuine FP8 matmul output. Its downside is verification cost: it is not cheaply
  Freivalds-checkable because it is not one algebraic matrix product. Every
  output element runs
  per-group max-exponent alignment, signed shifts, normalization, and bf16
  conversion logic.

- `hawkeye-class-counts`: FP8 checkpoint linears are reconstructed from exact
  class-count products for each Hawkeye K=32 group. For a group, the activation
  and weight FP8 values are bucketed by exponent/significand class; one integer
  product computes all counts for a chunk of activation classes against packed
  weight classes, and deterministic replay converts those counts back into the
  same aligned integer accumulation Hawkeye uses. With `IMA_HAWKEYE_CLASS_CHUNK=512`
  and `IMA_HAWKEYE_PACKED_COUNT_LANES=6`, each K=32 Hawkeye group becomes one
  checkable count product. On Qwen2.5-0.5B this is 7,680 checkable FP8-linear
  products per forward, versus 168 for `codebook`; accuracy is exact against the
  Hopper QGMMA teacher on the FP8-linears-only probe.

In short, `codebook` gives cheap checking with high teacher error, `hawkeye`
gives perfect teacher reconstruction without cheap checking, and
`hawkeye-class-counts` gives perfect teacher reconstruction with checkable
products that are still too numerous to be the final cheap path.

## Development target

The primary job for a developer is to reduce teacher error while preserving, or
recovering, the cheap proof shape:

```text
integer operands -> exact integer matrix product -> deterministic postprocessing
```

The codebook integer GEMM shows why this proof shape is attractive: it can be
checked cheaply with Freivalds because it is an ordinary exact matrix product
over integers. Any low-error replacement should preserve that property where
possible. In particular:

- the GEMM itself must remain a real integer GPU operation, not a float GEMM,
  fake-quantized operation, CPU fallback, or emulated path
- the matmul check must remain exact; do not introduce approximate-equality
  checks, tolerance windows, or prover-chosen corrections
- integer Freivalds checks of the matmuls can never have range checks, so a
  proposal that needs range checks for the matmul verification is out of scope
- any correction after the GEMM must be deterministic from fixed, committed, or
  reproducibly derived data

The `hawkeye` path shows what the low-error target should look like: exact
agreement with the Hopper QGMMA teacher. Its current direct replay is a reference
for accuracy, not a final proof-friendly construction.

Common sources of integer GEMM error include operand quantization, scale-field
mismatch, cancellation in poorly conditioned dot products, clipping, output
requantization, and kernel-specific accumulation or reduction behavior. Report
results per layer or matmul family, not only as pooled averages; one bad layer
can dominate downstream behavior.

Keep [CURRENT_STRATEGY.md](CURRENT_STRATEGY.md) as the concise record of the
active paths. When a strategy is removed from the active surface, delete it from
that record.

## Requirements

- CUDA GPU with SM_89+ support
- Python managed through `uv`
- Network/HF access to download `RedHatAI/Qwen2.5-0.5B-FP8-dynamic`

## Run

```bash
uv run python -m int_model_approximation
```

or:

```bash
uv run int-model-approximation
```

To run the Hawkeye student against the Hopper QGMMA teacher:

```bash
IMA_TEACHER_KERNEL=hopper_qgmma IMA_STUDENT_KERNEL=hawkeye \
  uv run python -m int_model_approximation
```

To run the exact class-count reconstruction:

```bash
IMA_TEACHER_KERNEL=hopper_qgmma IMA_STUDENT_KERNEL=hawkeye-class-counts \
  IMA_HAWKEYE_CLASS_CHUNK=512 IMA_HAWKEYE_PACKED_COUNT_LANES=6 \
  uv run python -m int_model_approximation
```

For a capped prompt sample:

```bash
uv run python scripts/eval_hopper_hawkeye_one_prompt.py --max-tokens 128
```

The result is written to:

```text
results/difr_layer_errors.json
```

## Output

The JSON includes:

- per-layer isolated L2 error: one integerized layer run on cached FP8-reference inputs
- per-layer cumulative L2 error: full integerized model output at each layer vs reference
- total logit L2 error
- DiFR score
- top-1 similarity
- top-5 similarity
- kernel launch counts for the FP8 reference and checkable int32 products

Use the isolated layer error to judge whether a GEMM-level change improved the
local approximation. Use cumulative layer and logit metrics to catch changes that
look good locally but destabilize the full model.

## Checks

```bash
uv run --extra dev pytest
uv run --extra dev ruff check
```
