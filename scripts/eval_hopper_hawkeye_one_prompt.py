"""Compare the Hopper QGMMA teacher and supported integer students on one prompt."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import int_model_approximation.__main__ as entry  # noqa: E402


DEFAULT_DATASET_URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
    "resolve/main/databricks-dolly-15k.jsonl"
)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _seconds_since(start: float) -> float:
    _sync()
    return time.perf_counter() - start


def _load_dolly_prompt(url: str, index: int) -> tuple[str, dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=60) as response:
        for row_index, raw_line in enumerate(response):
            if row_index != index:
                continue
            record = json.loads(raw_line)
            instruction = record.get("instruction", "").strip()
            context = record.get("context", "").strip()
            response_text = record.get("response", "").strip()
            parts = [instruction]
            if context:
                parts.append(context)
            if response_text:
                parts.append(response_text)
            prompt = "\n\n".join(part for part in parts if part)
            if not prompt:
                raise RuntimeError(f"HF dataset row {index} did not contain usable text")
            return prompt, record
    raise RuntimeError(f"HF dataset has fewer than {index + 1} rows")


def _tokenize_capped(tokenizer: Any, prompt: str, max_tokens: int, device: str) -> torch.Tensor:
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    if input_ids.shape[1] > max_tokens:
        input_ids = input_ids[:, :max_tokens]
    return input_ids.to(device)


def _load_teacher(model_id: str, device: str, teacher_kernel: str) -> tuple[torch.nn.Module, int]:
    entry.TEACHER_KERNEL = teacher_kernel
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    entry._disable_compressed_tensor_hooks(model)
    replaced = entry._replace_fp8_linears(model)
    return entry._freeze(model), len(replaced)


def _replace_student_fp8_linears_only(model: torch.nn.Module) -> list[str]:
    replacements: list[tuple[str, torch.nn.Linear]] = []
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.Linear)
            and module.weight.dtype == torch.float8_e4m3fn
            and hasattr(module, "weight_scale")
        ):
            replacements.append((name, module))

    for name, module in replacements:
        if entry.STUDENT_KERNEL == "hawkeye":
            replacement = entry.HawkeyeLinear(module.weight, module.weight_scale, module.bias)
        elif entry.STUDENT_KERNEL == "hawkeye-class-counts":
            replacement = entry.HawkeyeClassCountsLinear(
                module.weight,
                module.weight_scale,
                module.bias,
            )
        elif entry.STUDENT_KERNEL == "codebook":
            replacement = entry.CodebookLinear(module.weight, module.weight_scale, module.bias)
        else:
            supported = ", ".join(sorted(entry.SUPPORTED_STUDENT_KERNELS))
            raise RuntimeError(
                f"unknown student kernel {entry.STUDENT_KERNEL!r}; choose one of: {supported}"
            )
        entry._set_submodule(model, name, replacement)

    if not replacements:
        raise RuntimeError("No FP8 Linear modules with weight_scale were found.")
    return [name for name, _ in replacements]


def _load_student(
    model_id: str,
    device: str,
    student_kernel: str,
    *,
    fp8_only: bool,
) -> tuple[torch.nn.Module, int]:
    entry.STUDENT_KERNEL = student_kernel
    if entry.STUDENT_KERNEL not in entry.SUPPORTED_STUDENT_KERNELS:
        supported = ", ".join(sorted(entry.SUPPORTED_STUDENT_KERNELS))
        raise RuntimeError(
            f"unknown student kernel {entry.STUDENT_KERNEL!r}; choose one of: {supported}"
        )
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    entry._disable_compressed_tensor_hooks(model)
    if fp8_only:
        replaced = _replace_student_fp8_linears_only(model)
    else:
        replaced = entry._replace_int32_linears(model)
    return entry._freeze(model), len(replaced)


def _forward_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return entry._extract_logits(model(input_ids)).detach().cpu()


def _unload(model: torch.nn.Module) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _finite_stats(x: torch.Tensor) -> dict[str, int | bool]:
    finite = torch.isfinite(x)
    nan = torch.isnan(x)
    posinf = x == float("inf")
    neginf = x == float("-inf")
    return {
        "all_finite": bool(finite.all().item()),
        "finite": int(finite.sum().item()),
        "nan": int(nan.sum().item()),
        "posinf": int(posinf.sum().item()),
        "neginf": int(neginf.sum().item()),
        "total": int(x.numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=entry.MODEL_ID)
    parser.add_argument("--teacher-kernel", default="hopper_qgmma")
    parser.add_argument(
        "--student-kernel",
        choices=sorted(entry.SUPPORTED_STUDENT_KERNELS),
        default="hawkeye",
    )
    parser.add_argument("--dataset-url", default=DEFAULT_DATASET_URL)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--hawkeye-group", type=int, default=entry.HAWKEYE_PRODUCTS_PER_GROUP)
    parser.add_argument("--hawkeye-width", type=int, default=entry.HAWKEYE_INTERNAL_WIDTH)
    parser.add_argument("--hawkeye-class-chunk", type=int, default=entry.HAWKEYE_CLASS_CHUNK)
    parser.add_argument(
        "--hawkeye-packed-count-lanes",
        type=int,
        default=entry.HAWKEYE_PACKED_COUNT_LANES,
    )
    parser.add_argument("--hawkeye-block-m", type=int, default=entry.HAWKEYE_BLOCK_M)
    parser.add_argument("--hawkeye-block-n", type=int, default=entry.HAWKEYE_BLOCK_N)
    parser.add_argument("--student-fp8-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    entry.TEACHER_KERNEL = args.teacher_kernel
    entry.STUDENT_KERNEL = args.student_kernel
    device = entry._require_gpu()
    torch.manual_seed(0)
    entry.HAWKEYE_PRODUCTS_PER_GROUP = args.hawkeye_group
    entry.HAWKEYE_INTERNAL_WIDTH = args.hawkeye_width
    entry.HAWKEYE_CLASS_CHUNK = args.hawkeye_class_chunk
    entry.HAWKEYE_PACKED_COUNT_LANES = args.hawkeye_packed_count_lanes
    entry.HAWKEYE_BLOCK_M = args.hawkeye_block_m
    entry.HAWKEYE_BLOCK_N = args.hawkeye_block_n

    prompt_start = time.perf_counter()
    prompt, record = _load_dolly_prompt(args.dataset_url, args.dataset_index)
    prompt_s = time.perf_counter() - prompt_start

    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    input_ids = _tokenize_capped(tokenizer, prompt, args.max_tokens, device)
    tokenizer_s = _seconds_since(tokenizer_start)

    extension_s = 0.0
    if args.teacher_kernel in entry.HOPPER_QGMMA_TEACHER_KERNELS:
        extension_start = time.perf_counter()
        from scripts.hopper_qgmma_teacher import load_hopper_qgmma_extension

        load_hopper_qgmma_extension()
        extension_s = _seconds_since(extension_start)

    teacher_load_start = time.perf_counter()
    teacher, fp8_linears = _load_teacher(args.model_id, device, args.teacher_kernel)
    teacher_load_s = _seconds_since(teacher_load_start)

    teacher_forward_start = time.perf_counter()
    teacher_logits = _forward_logits(teacher, input_ids)
    teacher_forward_s = _seconds_since(teacher_forward_start)
    _unload(teacher)

    student_load_start = time.perf_counter()
    student, student_linears = _load_student(
        args.model_id,
        device,
        args.student_kernel,
        fp8_only=args.student_fp8_only,
    )
    student_load_s = _seconds_since(student_load_start)

    student_forward_start = time.perf_counter()
    with entry._Int32KernelProbe() as int_probe:
        student_logits = _forward_logits(student, input_ids)
    student_forward_s = _seconds_since(student_forward_start)
    student_int32_calls = int_probe.count
    _unload(student)

    total_error = entry._l2_stats(teacher_logits, student_logits)
    logit_error = entry._logit_metrics(teacher_logits, student_logits)
    top1_match = teacher_logits.argmax(-1) == student_logits.argmax(-1)

    result = {
        "model": args.model_id,
        "teacher_kernel": args.teacher_kernel,
        "student_kernel": args.student_kernel,
        "student_fp8_only": args.student_fp8_only,
        "dataset_url": args.dataset_url,
        "dataset_index": args.dataset_index,
        "dataset_category": record.get("category"),
        "prompt_chars": len(prompt),
        "prompt_tokens": int(input_ids.shape[1]),
        "device": torch.cuda.get_device_name(0),
        "fp8_linears": fp8_linears,
        "student_linears": student_linears,
        "hawkeye": {
            "products_per_group": args.hawkeye_group,
            "internal_width": args.hawkeye_width,
            "class_chunk": args.hawkeye_class_chunk,
            "packed_count_lanes": args.hawkeye_packed_count_lanes,
            "block_m": args.hawkeye_block_m,
            "block_n": args.hawkeye_block_n,
        },
        "timing_s": {
            "hf_prompt_fetch": prompt_s,
            "tokenizer_load_and_encode": tokenizer_s,
            "qgmma_extension_load": extension_s,
            "teacher_load_and_replace": teacher_load_s,
            "teacher_forward": teacher_forward_s,
            "student_load_and_replace": student_load_s,
            "student_forward": student_forward_s,
        },
        "kernel_calls": {
            "student_int32": student_int32_calls,
        },
        "finite": {
            "teacher_logits": _finite_stats(teacher_logits),
            "student_logits": _finite_stats(student_logits),
        },
        "total_error": total_error,
        "logit_error": logit_error,
        "top1_matches": int(top1_match.sum().item()),
        "top1_positions": int(top1_match.numel()),
        "logits_shape": list(teacher_logits.shape),
    }

    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
