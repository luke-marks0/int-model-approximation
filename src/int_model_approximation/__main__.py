"""Single DiFR experiment entrypoint.

Run:
    python -m int_model_approximation

This is intentionally not configurable. It evaluates the current HF quantized
model against an integerized copy using real GPU kernels only:

* reference linears with FP8 weights run through torch._scaled_mm
* codebook student linears run through a Triton int32 x int32 -> int64 CUDA kernel
* hawkeye student linears replay the Hopper FP8 accumulator with integer logic
* hawkeye-class-counts student linears replay Hawkeye from checkable class-count products
* no CPU fallback is allowed
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoTokenizer

from int_model_approximation.hawkeye import hawkeye_fp8_sum
from int_model_approximation.hawkeye_freivalds import exact_hawkeye_fp8_sum
from int_model_approximation.metrics import logit_l2, post_gumbel_margin, top1_match, topk_overlap


MODEL_ID = "RedHatAI/Qwen2.5-0.5B-FP8-dynamic"
OUTPUT_PATH = Path("results/difr_layer_errors.json")
TEACHER_KERNEL = os.environ.get("IMA_TEACHER_KERNEL", "fp8_scaled_mm")
HOPPER_QGMMA_TEACHER_KERNELS = {"hopper_qgmma", "hawkeye_qgmma"}
STUDENT_KERNEL = os.environ.get("IMA_STUDENT_KERNEL", "codebook")
SUPPORTED_STUDENT_KERNELS = {"codebook", "hawkeye", "hawkeye-class-counts"}
HAWKEYE_PRODUCTS_PER_GROUP = int(os.environ.get("IMA_HAWKEYE_GROUP", "32"))
HAWKEYE_INTERNAL_WIDTH = int(os.environ.get("IMA_HAWKEYE_WIDTH", "14"))
HAWKEYE_CLASS_CHUNK = int(os.environ.get("IMA_HAWKEYE_CLASS_CHUNK", "32"))
HAWKEYE_PACKED_COUNT_LANES = int(os.environ.get("IMA_HAWKEYE_PACKED_COUNT_LANES", "6"))
HAWKEYE_PACKED_GROUP_LANES = int(os.environ.get("IMA_HAWKEYE_PACKED_GROUP_LANES", "1"))
HAWKEYE_ZERO_EXPONENT = -139
HAWKEYE_BLOCK_M = int(os.environ.get("IMA_HAWKEYE_BLOCK_M", "4"))
HAWKEYE_BLOCK_N = int(os.environ.get("IMA_HAWKEYE_BLOCK_N", "32"))
PROMPT = (
    "Layer-wise error measurement matters because a quantized language model can "
    "preserve final-token behavior while still accumulating hidden-state drift. "
    "This run compares a production quantized checkpoint with an integerized copy "
    "using one deterministic forward pass through the same prompt.\n\n"
    "The teacher is RedHatAI's FP8-dynamic Qwen2.5-0.5B. The default codebook "
    "student replaces FP8 Linear layers with one Triton int32-by-int32 matmul "
    "over exact FP8-codebook integer values. That product can be checked cheaply "
    "by Freivalds' algorithm: a verifier picks a random sign vector r, computes "
    "B r and then A (B r), and accepts when (A B) r equals the claimed product "
    "times r.\n\n"
    "Consider the canonical SwiGLU block: y = silu(x @ W_gate.T) * (x @ W_up.T), "
    "then z = y @ W_down.T. Three linears, three integer matmuls per token. The "
    "intermediate hidden size at 0.5B is 4864; at 7B it is 18944. A naive int8 "
    "tile lands well below an int32 tile in expressivity but well above it in "
    "tensor-core throughput. Choosing between them is, in effect, choosing where "
    "to spend a few bits of accuracy in exchange for a Freivalds proof that "
    "remains cheap to verify on chain.\n\n"
    "def softmax(x):\n"
    "    m = x.max(axis=-1, keepdims=True)\n"
    "    e = (x - m).exp()\n"
    "    return e / e.sum(axis=-1, keepdims=True)\n\n"
    "RMSNorm is similar in spirit: y = x * gamma / sqrt(mean(x ** 2) + eps), "
    "where the bracketed mean is the only nonlinear piece. The interesting "
    "question for an integer proxy is not whether the nonlinearity can be made "
    "exact, but whether the deterministic post-processing applied to the matmul "
    "outputs absorbs enough of the rounding error introduced upstream that the "
    "logit distribution stays close to the FP8 reference.\n\n"
    "In ancient mythology, the river Styx separated the world of the living from "
    "the underworld, and Charon ferried the dead across it for a single coin. "
    "Numerical analysts have their own Styx: catastrophic cancellation in "
    "ill-conditioned dot products, where two large quantities of opposite sign "
    "nearly cancel and the tiny residue carries all of the answer's information. "
    "When you build a fully integer pipeline, you don't escape this — you choose "
    "where to confront it, and you pay your coin in extra bits of accumulator.\n\n"
    "Pourquoi quantifier? Parce que la mémoire est chère et la bande passante "
    "encore plus. \"The proof is in the pudding,\" she said, opening her laptop. "
    "Wenn der Beweis günstig sein soll, muss das Produkt exakt sein. 量化的关键 "
    "在于把误差留在可以被验证者捕获的地方。\n\n"
    "Ultimately, the practical question for this repository is narrower than the "
    "philosophy suggests. Per-layer isolated error tells us whether a particular "
    "integer linear path got closer to the FP8 reference; per-layer cumulative "
    "error tells us whether that local improvement actually propagates through "
    "the residual stream; and the post-Gumbel margin on the final logits tells us "
    "whether any of it would change a downstream sampler's decision. A change "
    "that improves isolated error but regresses cumulative or logit error is "
    "not a real improvement."
)

FP8_E4M3_MAX = 448.0
FP8_E4M3_CODE_SCALE = 512.0
INT32_MAX = (1 << 31) - 1
INT64_ACCUM_LIMIT = (1 << 62) - 1
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 32
_ACTIVE_INT32_PROBE = None


def _require_cuda_tensor(x: torch.Tensor, label: str) -> None:
    if x.device.type != "cuda":
        raise RuntimeError(f"{label} must be on CUDA; this repo has no CPU execution path")


def _require_gpu() -> str:
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required; CPU execution is unsupported.")
    if STUDENT_KERNEL not in SUPPORTED_STUDENT_KERNELS:
        supported = ", ".join(sorted(SUPPORTED_STUDENT_KERNELS))
        raise SystemExit(f"unknown IMA_STUDENT_KERNEL={STUDENT_KERNEL!r}; choose one of: {supported}")
    if TEACHER_KERNEL == "fp8_scaled_mm" and not hasattr(torch, "_scaled_mm"):
        raise SystemExit("torch._scaled_mm is unavailable; real FP8 GEMM cannot run.")
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) < (8, 9):
        raise SystemExit(f"SM_89+ is required for this run; found SM_{major}{minor}.")
    return "cuda"


def _freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _disable_compressed_tensor_hooks(model: nn.Module) -> None:
    """Keep HF compressed-tensors checkpoints from replacing FP8 weights."""
    for module in [model, *model.modules()]:
        hooks = getattr(module, "_forward_pre_hooks", None)
        if not hooks:
            continue
        for hook_id, fn in list(hooks.items()):
            qname = f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', '')}"
            if "compressed_tensors" in qname or "decompress" in qname.lower():
                del hooks[hook_id]


def _per_token_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _require_cuda_tensor(x, "FP8 activation")
    rows = x.reshape(-1, x.shape[-1])
    scale = rows.detach().abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scale = (scale / FP8_E4M3_MAX).clamp_min(1e-12)
    q = (rows.to(torch.float32) / scale).to(torch.float8_e4m3fn)
    return q.contiguous(), scale


def _fp8_e4m3_to_int32(x: torch.Tensor) -> torch.Tensor:
    return (x.to(torch.float32) * FP8_E4M3_CODE_SCALE).round().to(torch.int32).contiguous()


def _per_token_fp8_int32(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_fp8, scale = _per_token_fp8(x)
    return _fp8_e4m3_to_int32(q_fp8), scale / FP8_E4M3_CODE_SCALE


def _int32_qmax(k: int) -> float:
    # Keep worst-case sum_k(qmax*qmax) inside signed int64 with headroom.
    return float(min(INT32_MAX, int(math.sqrt(INT64_ACCUM_LIMIT / max(k, 1)))))


def _per_token_int32(x: torch.Tensor, qmax: float) -> tuple[torch.Tensor, torch.Tensor]:
    _require_cuda_tensor(x, "int32 activation")
    rows = x.reshape(-1, x.shape[-1])
    scale = rows.detach().abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scale = (scale / qmax).clamp_min(1e-12)
    q = (rows.to(torch.float32) / scale).round().clamp(-qmax, qmax)
    return q.to(torch.int32).contiguous(), scale


def _per_row_int32(weight: torch.Tensor, qmax: float) -> tuple[torch.Tensor, torch.Tensor]:
    _require_cuda_tensor(weight, "int32 weight source")
    w = weight.detach().to(torch.float32)
    scale = (w.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-12)
    q = (w / scale).round().clamp(-qmax, qmax)
    return q.to(torch.int32).contiguous(), scale.to(torch.float32)


@triton.jit
def _int32_raw_matmul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    m_size: tl.constexpr,
    n_size: tl.constexpr,
    k_size: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    accum = tl.zeros((block_m, block_n), dtype=tl.int64)

    for k0 in range(0, k_size, block_k):
        for kk in range(0, block_k):
            k = k0 + kk
            a = tl.load(
                a_ptr + offs_m * k_size + k,
                mask=(offs_m < m_size) & (k < k_size),
                other=0,
            ).to(tl.int64)
            b = tl.load(
                b_ptr + k * n_size + offs_n,
                mask=(k < k_size) & (offs_n < n_size),
                other=0,
            ).to(tl.int64)
            accum += a[:, None] * b[None, :]

    tl.store(
        out_ptr + offs_m[:, None] * n_size + offs_n[None, :],
        accum,
        mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size),
    )


def _int32_raw_matmul(activations: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
    global _ACTIVE_INT32_PROBE
    _require_cuda_tensor(activations, "int32 activation matrix")
    _require_cuda_tensor(weight_t, "int32 weight matrix")
    if activations.dtype != torch.int32 or weight_t.dtype != torch.int32:
        raise RuntimeError("int32 raw matmul received non-int32 operands")
    m_size, k_size = activations.shape
    k2, n_size = weight_t.shape
    if k_size != k2:
        raise RuntimeError(f"int32 raw matmul shape mismatch: {activations.shape} @ {weight_t.shape}")
    out = torch.empty((m_size, n_size), device=activations.device, dtype=torch.int64)
    if _ACTIVE_INT32_PROBE is not None:
        _ACTIVE_INT32_PROBE.count += 1
    _int32_raw_matmul_kernel[(triton.cdiv(m_size, BLOCK_M), triton.cdiv(n_size, BLOCK_N))](
        activations,
        weight_t,
        out,
        m_size,
        n_size,
        k_size,
        block_m=BLOCK_M,
        block_n=BLOCK_N,
        block_k=BLOCK_K,
        num_warps=4,
    )
    return out


@triton.jit
def _int32_matmul_kernel(
    a_ptr,
    b_ptr,
    x_scale_ptr,
    w_scale_ptr,
    out_ptr,
    m_size: tl.constexpr,
    n_size: tl.constexpr,
    k_size: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    accum = tl.zeros((block_m, block_n), dtype=tl.int64)

    for k0 in range(0, k_size, block_k):
        for kk in range(0, block_k):
            k = k0 + kk
            a = tl.load(
                a_ptr + offs_m * k_size + k,
                mask=(offs_m < m_size) & (k < k_size),
                other=0,
            ).to(tl.int64)
            b = tl.load(
                b_ptr + k * n_size + offs_n,
                mask=(k < k_size) & (offs_n < n_size),
                other=0,
            ).to(tl.int64)
            accum += a[:, None] * b[None, :]

    x_scale = tl.load(x_scale_ptr + offs_m, mask=offs_m < m_size, other=0.0).to(tl.float32)
    w_scale = tl.load(w_scale_ptr + offs_n, mask=offs_n < n_size, other=0.0).to(tl.float32)
    out = accum.to(tl.float32) * x_scale[:, None] * w_scale[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * n_size + offs_n[None, :],
        out,
        mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size),
    )


def _int32_matmul(
    activations: torch.Tensor,
    weight_t: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    global _ACTIVE_INT32_PROBE
    _require_cuda_tensor(activations, "int32 activation matrix")
    _require_cuda_tensor(weight_t, "int32 weight matrix")
    if activations.dtype != torch.int32 or weight_t.dtype != torch.int32:
        raise RuntimeError("int32 matmul received non-int32 operands")
    m_size, k_size = activations.shape
    k2, n_size = weight_t.shape
    if k_size != k2:
        raise RuntimeError(f"int32 matmul shape mismatch: {activations.shape} @ {weight_t.shape}")
    out = torch.empty((m_size, n_size), device=activations.device, dtype=torch.float32)
    if _ACTIVE_INT32_PROBE is not None:
        _ACTIVE_INT32_PROBE.count += 1
    _int32_matmul_kernel[(triton.cdiv(m_size, BLOCK_M), triton.cdiv(n_size, BLOCK_N))](
        activations,
        weight_t,
        x_scale.reshape(-1),
        w_scale.reshape(-1),
        out,
        m_size,
        n_size,
        k_size,
        block_m=BLOCK_M,
        block_n=BLOCK_N,
        block_k=BLOCK_K,
        num_warps=4,
    )
    return out


def _hopper_qgmma_fp8_scaled_mm(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    try:
        from scripts.hopper_qgmma_teacher import qgmma_fp8_scaled_mm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "IMA_TEACHER_KERNEL=hopper_qgmma requires scripts/hopper_qgmma_teacher.py "
            "from this repository checkout."
        ) from exc
    return qgmma_fp8_scaled_mm(x_fp8, x_scale, w_fp8, w_scale)


class FP8Linear(nn.Module):
    """Linear backed by the selected FP8 teacher GEMM."""

    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        if weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError(f"FP8Linear needs float8_e4m3fn weights, got {weight.dtype}")
        _require_cuda_tensor(weight, "FP8 weight")
        self.register_buffer("weight", weight.detach().contiguous(), persistent=False)
        self.register_buffer("weight_scale", weight_scale.detach().to(torch.float32), persistent=False)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach(), persistent=False)
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _require_cuda_tensor(x, "FP8 input")
        in_shape = x.shape
        x_fp8, x_scale = _per_token_fp8(x)
        if TEACHER_KERNEL == "fp8_scaled_mm":
            y = torch._scaled_mm(
                x_fp8,
                self.weight.t(),
                scale_a=x_scale,
                scale_b=self.weight_scale.reshape(1, -1),
                out_dtype=torch.bfloat16,
            )
        elif TEACHER_KERNEL in HOPPER_QGMMA_TEACHER_KERNELS:
            y = _hopper_qgmma_fp8_scaled_mm(x_fp8, x_scale, self.weight, self.weight_scale)
        else:
            raise RuntimeError(f"unknown IMA_TEACHER_KERNEL={TEACHER_KERNEL!r}")
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.to(x.dtype).reshape(*in_shape[:-1], self.out_features)


class Int32Linear(nn.Module):
    """Linear backed by a real int32 GPU GEMM implemented as a Triton kernel."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ):
        super().__init__()
        _require_cuda_tensor(weight, "int32 weight source")
        self.qmax = _int32_qmax(weight.shape[1])
        w_i32, w_scale = _per_row_int32(weight, self.qmax)
        self.register_buffer("weight_t", w_i32.t().contiguous(), persistent=False)
        self.register_buffer("weight_scale", w_scale.reshape(1, -1), persistent=False)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach(), persistent=False)
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _require_cuda_tensor(x, "int32 input")
        in_shape = x.shape
        x_i32, x_scale = _per_token_int32(x, self.qmax)
        y = _int32_matmul(x_i32, self.weight_t, x_scale, self.weight_scale)
        if self.bias is not None:
            y = y + self.bias.to(torch.float32)
        return y.to(x.dtype).reshape(*in_shape[:-1], self.out_features)


class CodebookLinear(nn.Module):
    """FP8 linear reconstructed by one Freivalds-checkable codebook integer GEMM."""

    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        if weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError(f"CodebookLinear needs FP8 weights, got {weight.dtype}")
        _require_cuda_tensor(weight, "FP8 codebook weight")
        codebook_i32 = _fp8_e4m3_to_int32(weight.detach())
        codebook_scale = weight_scale.detach().to(torch.float32) / FP8_E4M3_CODE_SCALE
        self.register_buffer("weight_t", codebook_i32.t().contiguous(), persistent=False)
        self.register_buffer("weight_scale", codebook_scale.reshape(1, -1), persistent=False)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach(), persistent=False)
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _require_cuda_tensor(x, "FP8 codebook input")
        in_shape = x.shape
        x_i32, x_scale = _per_token_fp8_int32(x)
        y = _int32_matmul(x_i32, self.weight_t, x_scale, self.weight_scale)
        if self.bias is not None:
            y = y + self.bias.to(torch.float32)
        return y.to(x.dtype).reshape(*in_shape[:-1], self.out_features)


class HawkeyeLinear(nn.Module):
    """FP8 linear reconstructed by direct Hawkeye integer replay."""

    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        if weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError(f"HawkeyeLinear needs FP8 weights, got {weight.dtype}")
        _require_cuda_tensor(weight, "Hawkeye FP8 weight")
        self.register_buffer("weight", weight.detach().contiguous(), persistent=False)
        self.register_buffer("weight_scale", weight_scale.detach().to(torch.float32), persistent=False)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach(), persistent=False)
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _require_cuda_tensor(x, "Hawkeye input")
        in_shape = x.shape
        x_fp8, x_scale = _per_token_fp8(x)
        y, _stats = hawkeye_fp8_sum(
            x_fp8,
            x_scale,
            self.weight,
            self.weight_scale,
            products_per_group=HAWKEYE_PRODUCTS_PER_GROUP,
            internal_width=HAWKEYE_INTERNAL_WIDTH,
            zero_exponent=HAWKEYE_ZERO_EXPONENT,
            block_m=HAWKEYE_BLOCK_M,
            block_n=HAWKEYE_BLOCK_N,
        )
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.to(x.dtype).reshape(*in_shape[:-1], self.out_features)


class HawkeyeClassCountsLinear(nn.Module):
    """FP8 linear reconstructed exactly from checkable Hawkeye class-count products."""

    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        if weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError(f"HawkeyeClassCountsLinear needs FP8 weights, got {weight.dtype}")
        _require_cuda_tensor(weight, "Hawkeye exact FP8 weight")
        self.register_buffer("weight", weight.detach().contiguous(), persistent=False)
        self.register_buffer("weight_scale", weight_scale.detach().to(torch.float32), persistent=False)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach(), persistent=False)
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _require_cuda_tensor(x, "Hawkeye exact input")
        in_shape = x.shape
        x_fp8, x_scale = _per_token_fp8(x)
        y, _stats = exact_hawkeye_fp8_sum(
            x_fp8,
            x_scale,
            self.weight,
            self.weight_scale,
            _int32_raw_matmul,
            products_per_group=HAWKEYE_PRODUCTS_PER_GROUP,
            internal_width=HAWKEYE_INTERNAL_WIDTH,
            zero_exponent=HAWKEYE_ZERO_EXPONENT,
            class_chunk=HAWKEYE_CLASS_CHUNK,
            packed_count_lanes=HAWKEYE_PACKED_COUNT_LANES,
            packed_group_lanes=HAWKEYE_PACKED_GROUP_LANES,
        )
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.to(x.dtype).reshape(*in_shape[:-1], self.out_features)


def _expanded_block_scale(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if scale.shape == weight.shape:
        return scale
    row_block = math.ceil(weight.shape[0] / scale.shape[0])
    col_block = math.ceil(weight.shape[1] / scale.shape[1])
    return scale.repeat_interleave(row_block, 0).repeat_interleave(col_block, 1)[
        : weight.shape[0], : weight.shape[1]
    ]


def _dequantized_weight(module: nn.Linear) -> torch.Tensor:
    weight = module.weight.detach()
    if weight.dtype == torch.float8_e4m3fn:
        if hasattr(module, "weight_scale"):
            return weight.to(torch.float32) * module.weight_scale.detach().to(torch.float32)
        if hasattr(module, "weight_scale_inv"):
            scale = module.weight_scale_inv.detach().to(torch.float32)
            return weight.to(torch.float32) * _expanded_block_scale(weight, scale)
        raise RuntimeError("FP8 Linear is missing a recognized weight scale")
    return weight.to(torch.float32)


def _replace_fp8_linears(model: nn.Module) -> list[str]:
    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.dtype == torch.float8_e4m3fn:
            replacements.append((name, module))
    for name, module in replacements:
        if not hasattr(module, "weight_scale"):
            raise RuntimeError(f"{name} has FP8 weights but no per-row weight_scale")
        replacement = FP8Linear(module.weight, module.weight_scale, module.bias)
        _set_submodule(model, name, replacement)
    if not replacements:
        raise RuntimeError("The HF checkpoint did not expose any FP8 Linear weights.")
    return [name for name, _ in replacements]


def _replace_int32_linears(model: nn.Module) -> list[str]:
    if STUDENT_KERNEL not in SUPPORTED_STUDENT_KERNELS:
        supported = ", ".join(sorted(SUPPORTED_STUDENT_KERNELS))
        raise RuntimeError(f"unknown IMA_STUDENT_KERNEL={STUDENT_KERNEL!r}; choose one of: {supported}")
    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            replacements.append((name, module))
    for name, module in replacements:
        if module.weight.dtype == torch.float8_e4m3fn and hasattr(module, "weight_scale"):
            if STUDENT_KERNEL == "hawkeye":
                replacement = HawkeyeLinear(module.weight, module.weight_scale, module.bias)
            elif STUDENT_KERNEL == "hawkeye-class-counts":
                replacement = HawkeyeClassCountsLinear(
                    module.weight,
                    module.weight_scale,
                    module.bias,
                )
            else:
                replacement = CodebookLinear(module.weight, module.weight_scale, module.bias)
        else:
            replacement = Int32Linear(_dequantized_weight(module), module.bias)
        _set_submodule(model, name, replacement)
    if not replacements:
        raise RuntimeError("No Linear modules were found to integerize.")
    return [name for name, _ in replacements]


def _set_submodule(model: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, replacement)


@dataclass
class Capture:
    input: torch.Tensor
    output: torch.Tensor


def _install_hooks(model: nn.Module, layer_names: set[str]) -> tuple[dict[str, Capture], list[Any]]:
    captures: dict[str, Capture] = {}
    handles: list[Any] = []
    for name, module in model.named_modules():
        if name not in layer_names:
            continue

        def hook(_module: nn.Module, args: tuple[Any, ...], output: Any, name: str = name) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                return
            if not isinstance(output, torch.Tensor):
                return
            captures[name] = Capture(
                input=args[0].detach().clone(),
                output=output.detach().clone(),
            )

        handles.append(module.register_forward_hook(hook))
    return captures, handles


def _remove_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def _l2_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise RuntimeError(f"shape mismatch: {tuple(reference.shape)} vs {tuple(candidate.shape)}")
    diff = reference.float() - candidate.float()
    flat_l2 = diff.norm().item()
    if diff.dim() <= 1:
        per_position = diff.reshape(1, -1).norm(dim=-1)
    else:
        per_position = diff.norm(dim=-1).flatten()
    return {
        "l2": flat_l2,
        "per_position_mean": per_position.mean().item(),
        "per_position_p50": per_position.median().item(),
        "per_position_p99": per_position.quantile(0.99).item(),
        "per_position_max": per_position.max().item(),
    }


class _KernelProbe:
    def __init__(self, attr: str):
        self.attr = attr
        self.count = 0
        self._original = None

    def __enter__(self) -> "_KernelProbe":
        self._original = getattr(torch, self.attr)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return self._original(*args, **kwargs)

        setattr(torch, self.attr, wrapper)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        setattr(torch, self.attr, self._original)


class _Int32KernelProbe:
    def __init__(self):
        self.count = 0

    def __enter__(self) -> "_Int32KernelProbe":
        global _ACTIVE_INT32_PROBE
        _ACTIVE_INT32_PROBE = self
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        global _ACTIVE_INT32_PROBE
        _ACTIVE_INT32_PROBE = None


def _extract_logits(output: Any) -> torch.Tensor:
    if hasattr(output, "logits"):
        return output.logits
    return output[0]


def _logit_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(0)
    noise = torch.empty(reference.shape, device=reference.device, dtype=torch.float32)
    noise.exponential_(generator=generator).log_().neg_()
    difr = post_gumbel_margin(reference, candidate, noise)
    l2 = logit_l2(reference, candidate)
    return {
        "logit_l2": l2.norm().item(),
        "logit_l2_mean": l2.float().mean().item(),
        "logit_l2_p99": l2.float().flatten().quantile(0.99).item(),
        "difr_score_mean": difr.float().mean().item(),
        "difr_score_p99": difr.float().flatten().quantile(0.99).item(),
        "top1_similarity": top1_match(reference, candidate).float().mean().item(),
        "top5_similarity": topk_overlap(reference, candidate, k=5).float().mean().item(),
    }


def _load_reference(device: str) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    _disable_compressed_tensor_hooks(model)
    replaced = _replace_fp8_linears(model)
    print(f"[difr] reference: {MODEL_ID}, FP8 linears={len(replaced)}")
    return _freeze(model)


def _load_integerized(device: str) -> tuple[nn.Module, list[str]]:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    _disable_compressed_tensor_hooks(model)
    replaced = _replace_int32_linears(model)
    print(f"[difr] integerized linears={len(replaced)}")
    return _freeze(model), replaced


def main() -> None:
    device = _require_gpu()
    torch.manual_seed(0)
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    input_ids = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    reference = _load_reference(device)
    integerized, layer_names = _load_integerized(device)
    layer_name_set = set(layer_names)

    ref_captures, ref_handles = _install_hooks(reference, layer_name_set)
    fp8_probe_count = 0
    if TEACHER_KERNEL == "fp8_scaled_mm":
        with torch.inference_mode(), _KernelProbe("_scaled_mm") as fp8_probe:
            reference_output = reference(input_ids)
        fp8_probe_count = fp8_probe.count
        if fp8_probe_count == 0:
            raise RuntimeError("Reference forward did not call torch._scaled_mm.")
    else:
        with torch.inference_mode():
            reference_output = reference(input_ids)
    _remove_hooks(ref_handles)

    int_captures, int_handles = _install_hooks(integerized, layer_name_set)
    with torch.inference_mode(), _Int32KernelProbe() as int_probe:
        integerized_output = integerized(input_ids)
    _remove_hooks(int_handles)
    if STUDENT_KERNEL in {"codebook", "hawkeye-class-counts"} and int_probe.count == 0:
        raise RuntimeError("Integerized forward did not launch the int32 CUDA kernel.")

    rows = []
    isolated_int32_kernel_calls = 0
    for name in layer_names:
        if name not in ref_captures or name not in int_captures:
            continue
        int_layer = integerized.get_submodule(name)
        with torch.inference_mode(), _Int32KernelProbe() as iso_probe:
            isolated_output = int_layer(ref_captures[name].input)
        isolated_int32_kernel_calls += iso_probe.count
        if not isinstance(int_layer, HawkeyeLinear) and iso_probe.count == 0:
            raise RuntimeError(f"Isolated layer {name} did not launch the int32 CUDA kernel.")
        rows.append(
            {
                "name": name,
                "shape": list(ref_captures[name].output.shape),
                "integer_qmax": getattr(int_layer, "qmax", None),
                "isolated_error": _l2_stats(ref_captures[name].output, isolated_output),
                "cumulative_error": _l2_stats(
                    ref_captures[name].output, int_captures[name].output
                ),
            }
        )

    reference_logits = _extract_logits(reference_output)
    integerized_logits = _extract_logits(integerized_output)
    result = {
        "model": MODEL_ID,
        "prompt_tokens": int(input_ids.shape[1]),
        "device": torch.cuda.get_device_name(0),
        "integerized_kernel": STUDENT_KERNEL,
        "teacher_kernel": TEACHER_KERNEL,
        "integerized_path": (
            {
                "codebook": "single FP8-codebook integer GEMM",
                "hawkeye": "direct Hawkeye FP8 accumulator replay",
                "hawkeye-class-counts": (
                    "exact Hawkeye replay from Freivalds-checkable class-count products"
                ),
            }[STUDENT_KERNEL]
        ),
        "hawkeye_class_counts": (
            {
                "class_chunk": HAWKEYE_CLASS_CHUNK,
                "packed_count_lanes": HAWKEYE_PACKED_COUNT_LANES,
                "packed_group_lanes": HAWKEYE_PACKED_GROUP_LANES,
                "products_per_group": HAWKEYE_PRODUCTS_PER_GROUP,
                "internal_width": HAWKEYE_INTERNAL_WIDTH,
            }
            if STUDENT_KERNEL == "hawkeye-class-counts"
            else None
        ),
        "integerized_qmax": "per-layer floor(sqrt((2^62 - 1) / in_features))",
        "runtime_s": time.time() - started,
        "kernel_calls": {
            "reference_scaled_mm": fp8_probe_count,
            "integerized_checkable_int32_kernel": int_probe.count,
            "isolated_int32_kernel": isolated_int32_kernel_calls,
        },
        "total_error": _l2_stats(reference_logits, integerized_logits),
        "logit_error": _logit_metrics(reference_logits, integerized_logits),
        "layers": rows,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    print(
        "[difr] wrote "
        f"{OUTPUT_PATH} | layers={len(rows)} | "
        f"logit_l2_mean={result['logit_error']['logit_l2_mean']:.4g} | "
        f"top1={result['logit_error']['top1_similarity']:.4f} | "
        f"top5={result['logit_error']['top5_similarity']:.4f}"
    )


if __name__ == "__main__":
    main()
