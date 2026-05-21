# int-model-approximation

DiFR layer-error measurement for a real quantized Hugging Face model.

The goal is to build a cheap, provable integer proxy for a production quantized
model. Proving the production float/FP8 computation directly is expensive: the
matmuls do not give the clean algebraic product that Freivalds-style checks need,
and trying to prove float behavior tends to pull in costly range checks,
rounding semantics, tolerance windows, and implementation-specific kernel state.

This repo therefore measures a narrower question: what do we get from integer
students that either preserve a cheap Freivalds proof shape or exactly replay the
hardware FP8 accumulator?

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
  directly. It perfectly reconstructs the repo's codebook-defined FP8 product,
  but it is not a faithful replay of NVIDIA's FP8 Tensor Core accumulator.

- `hawkeye`: FP8 checkpoint linears are replaced by direct Hawkeye integer replay
  of Hopper FP8 QGMMA accumulation, using K=32 groups by default. This is the
  hardware-faithful path: with the Hopper QGMMA teacher it can exactly match the
  genuine FP8 matmul output. It is not cheaply Freivalds-checkable because the
  computation is not one algebraic matrix product; every output element runs
  per-group max-exponent alignment, signed shifts, normalization, and bf16
  conversion logic.

In short, use `codebook` when the main requirement is one cheap, checkable
integer matmul per FP8 linear. Use `hawkeye` when the main requirement is exact
agreement with the genuine Hopper FP8 teacher and the proof cost is not the
deciding constraint.

## Development target

The primary job for a developer working on the `codebook` path is to reduce the
error introduced by the integer GEMMs while preserving the proof shape:

```text
integer operands -> exact integer matrix product -> deterministic postprocessing
```

The codebook integer GEMM can be checked cheaply with Freivalds because it is an
ordinary exact matrix product over integers. Any improvement to that path must
keep that property intact. In particular:

- the GEMM itself must remain a real integer GPU operation, not a float GEMM,
  fake-quantized operation, CPU fallback, or emulated path
- the matmul check must remain exact; do not introduce approximate-equality
  checks, tolerance windows, or prover-chosen corrections
- integer Freivalds checks of the matmuls can never have range checks, so a
  proposal that needs range checks for the matmul verification is out of scope
- any correction after the GEMM must be deterministic from fixed, committed, or
  reproducibly derived data

The `hawkeye` path has a different goal: it is allowed to replay the FP8
accumulator directly, and should be judged against the Hopper QGMMA teacher
rather than against the Freivalds proof shape.

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
