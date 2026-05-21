"""Time one Qwen forward pass with the experimental Hopper QGMMA FP8 teacher."""

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


def _prepare_model(model_id: str, device: str, teacher_kernel: str) -> tuple[torch.nn.Module, int]:
    entry.TEACHER_KERNEL = teacher_kernel
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    entry._disable_compressed_tensor_hooks(model)
    replaced = entry._replace_fp8_linears(model)
    return entry._freeze(model), len(replaced)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=entry.MODEL_ID)
    parser.add_argument("--teacher-kernel", default="hopper_qgmma")
    parser.add_argument("--dataset-url", default=DEFAULT_DATASET_URL)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    entry.TEACHER_KERNEL = args.teacher_kernel
    device = entry._require_gpu()
    torch.manual_seed(0)

    prompt_start = time.perf_counter()
    prompt, record = _load_dolly_prompt(args.dataset_url, args.dataset_index)
    prompt_s = time.perf_counter() - prompt_start

    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    input_ids = _tokenize_capped(tokenizer, prompt, args.max_tokens, device)
    tokenizer_s = _seconds_since(tokenizer_start)

    build_s = 0.0
    if args.teacher_kernel in entry.HOPPER_QGMMA_TEACHER_KERNELS:
        build_start = time.perf_counter()
        from scripts.hopper_qgmma_teacher import load_hopper_qgmma_extension

        load_hopper_qgmma_extension()
        build_s = _seconds_since(build_start)

    model_start = time.perf_counter()
    model, fp8_linears = _prepare_model(args.model_id, device, args.teacher_kernel)
    model_s = _seconds_since(model_start)

    forward_times: list[float] = []
    last_logits: torch.Tensor | None = None
    for _ in range(args.repeat):
        start = time.perf_counter()
        with torch.inference_mode():
            output = model(input_ids)
            last_logits = entry._extract_logits(output)
        forward_times.append(_seconds_since(start))

    if last_logits is None:
        raise RuntimeError("forward did not produce logits")

    result = {
        "model": args.model_id,
        "teacher_kernel": args.teacher_kernel,
        "dataset_url": args.dataset_url,
        "dataset_index": args.dataset_index,
        "dataset_category": record.get("category"),
        "prompt_chars": len(prompt),
        "prompt_tokens": int(input_ids.shape[1]),
        "device": torch.cuda.get_device_name(0),
        "fp8_linears": fp8_linears,
        "timing_s": {
            "hf_prompt_fetch": prompt_s,
            "tokenizer_load_and_encode": tokenizer_s,
            "qgmma_extension_load": build_s,
            "model_load_and_replace": model_s,
            "forward": forward_times,
        },
        "logits_shape": list(last_logits.shape),
        "top1_first_token": int(last_logits[0, 0].argmax().item()),
        "top1_last_token": int(last_logits[0, -1].argmax().item()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
