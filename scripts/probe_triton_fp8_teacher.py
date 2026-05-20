"""Compare Hawkeye models against a simplified Tensor-Core FP8 teacher.

The default teacher in this repo is `torch._scaled_mm`, which dispatches into a
cuBLAS/CUTLASS kernel.  That full GEMM path can differ from a single Hawkeye
MMA-tile model because of macro-kernel scheduling, split-K policy, epilogue
details, and CUDA-version-specific accumulation choices.

This probe defines a narrower teacher:

  * inputs are still real `torch.float8_e4m3fn`
  * the matmul is still performed by Tensor Cores through Triton `tl.dot`
  * program tiles are fixed to `BLOCK_M=16`, `BLOCK_N=16`, `BLOCK_K=32`
  * K tiles are accumulated in a fixed left-to-right loop

That should make each inner dot map cleanly to a small fixed set of SM89 FP8
MMA tiles and make Hawkeye-style tile models a better fit, if the remaining gap
was mostly cuBLAS macro-kernel behavior.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch
import triton
import triton.language as tl

from probe_hawkeye_bucket_products import moment_bucket_sum
from probe_hawkeye_l40s_fp8 import (
    Candidate,
    exact_codebook_sum,
    fp8_scaled_mm,
    hawkeye_grouped_sum,
    metrics,
    per_token_fp8,
)


@triton.jit
def _triton_fp8_scaled_mm_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    out_ptr,
    m_size: tl.constexpr,
    n_size: tl.constexpr,
    k_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k0 in range(0, k_size, block_k):
        k = k0 + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(offs_m[:, None] < m_size) & (k[None, :] < k_size),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < k_size) & (offs_n[None, :] < n_size),
            other=0.0,
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    scale_a = tl.load(scale_a_ptr + offs_m, mask=offs_m < m_size, other=0.0).to(tl.float32)
    scale_b = tl.load(scale_b_ptr + offs_n, mask=offs_n < n_size, other=0.0).to(tl.float32)
    y = acc * scale_a[:, None] * scale_b[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * n_size + offs_n[None, :],
        y,
        mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size),
    )


def triton_fp8_scaled_mm(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 32,
) -> torch.Tensor:
    if x_fp8.dtype != torch.float8_e4m3fn or w_fp8.dtype != torch.float8_e4m3fn:
        raise RuntimeError("triton teacher expects float8_e4m3fn operands")
    if x_fp8.shape[1] != w_fp8.shape[1]:
        raise RuntimeError(f"shape mismatch: {x_fp8.shape} vs {w_fp8.shape}")
    if block_k != 32:
        raise RuntimeError("this probe intentionally fixes block_k=32")

    m_size, k_size = x_fp8.shape
    n_size = w_fp8.shape[0]
    # B is K x N so the dot layout is A[M,K] @ B[K,N].  The copy avoids any
    # hidden transpose/stride behavior in the teacher kernel.
    b = w_fp8.t().contiguous()
    out = torch.empty((m_size, n_size), device=x_fp8.device, dtype=torch.bfloat16)
    _triton_fp8_scaled_mm_kernel[(triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))](
        x_fp8,
        b,
        x_scale.reshape(-1),
        w_scale.reshape(-1).to(torch.float32),
        out,
        m_size,
        n_size,
        k_size,
        stride_am=x_fp8.stride(0),
        stride_ak=x_fp8.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=4,
    )
    return out


def _random_case(
    m_size: int,
    n_size: int,
    k_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m_size, k_size), generator=generator, device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n_size, k_size), generator=generator, device="cuda", dtype=torch.float32)
    w_scale = (w.abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12)
    w_fp8 = (w / w_scale).to(torch.float8_e4m3fn).contiguous()
    x_fp8, x_scale = per_token_fp8(x)
    return x_fp8, x_scale, w_fp8, w_scale, {}


def _real_weight_case(
    m_size: int,
    seed: int,
    model_id: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    for name, module in model.named_modules():
        if (
            hasattr(module, "weight")
            and hasattr(module, "weight_scale")
            and module.weight.dtype == torch.float8_e4m3fn
        ):
            w_fp8 = module.weight.detach().contiguous()
            w_scale = module.weight_scale.detach().to(torch.float32)
            layer_name = name
            break
    else:
        raise RuntimeError("no FP8 weight with weight_scale found")

    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m_size, w_fp8.shape[1]), generator=generator, device="cuda", dtype=torch.bfloat16)
    x_fp8, x_scale = per_token_fp8(x)
    return x_fp8, x_scale, w_fp8, w_scale, {"model": model_id, "layer": layer_name}


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    candidate = Candidate(
        name=f"g{args.group}_w{args.width}",
        products_per_group=args.group,
        internal_width=args.width,
    )
    if args.real_weight:
        x_fp8, x_scale, w_fp8, w_scale, meta = _real_weight_case(args.m, args.seed, args.model_id)
    else:
        x_fp8, x_scale, w_fp8, w_scale, meta = _random_case(args.m, args.n, args.k, args.seed)

    cublas_teacher = fp8_scaled_mm(x_fp8, x_scale, w_fp8, w_scale)
    triton_teacher = triton_fp8_scaled_mm(x_fp8, x_scale, w_fp8, w_scale)
    codebook = exact_codebook_sum(x_fp8, x_scale, w_fp8, w_scale)
    hawkeye = hawkeye_grouped_sum(x_fp8, x_scale, w_fp8, w_scale, candidate)
    moment, moment_stats = moment_bucket_sum(x_fp8, x_scale, w_fp8, w_scale, candidate)

    return {
        **meta,
        "shape": [int(x_fp8.shape[0]), int(w_fp8.shape[0]), int(x_fp8.shape[1])],
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "teacher": {
            "block_m": 16,
            "block_n": 16,
            "block_k": 32,
            "uses_tensor_cores": "Triton tl.dot over float8_e4m3fn operands",
        },
        "candidate": asdict(candidate),
        "triton_vs_cublas_teacher": metrics(cublas_teacher, triton_teacher),
        "vs_triton_teacher": {
            "exact_codebook": metrics(triton_teacher, codebook),
            "hawkeye_scalar": metrics(triton_teacher, hawkeye),
            "moment_buckets": metrics(triton_teacher, moment),
        },
        "vs_cublas_teacher": {
            "exact_codebook": metrics(cublas_teacher, codebook),
            "hawkeye_scalar": metrics(cublas_teacher, hawkeye),
            "moment_buckets": metrics(cublas_teacher, moment),
        },
        "bucket_products": {
            "moment_buckets": asdict(moment_stats),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--k", type=int, default=896)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group", type=int, default=32)
    parser.add_argument("--width", type=int, default=17)
    parser.add_argument("--real-weight", action="store_true")
    parser.add_argument("--model-id", default="RedHatAI/Qwen2.5-0.5B-FP8-dynamic")
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    result = run_experiment(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
