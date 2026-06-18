"""Binary-answer logit error sweep for direct int32 model approximations.

The primary metric is the error on the teacher's top binary answer token:

    student_logit[teacher_binary_argmax] - teacher_logit[teacher_binary_argmax]

This intentionally ignores whole-vocabulary logit L2 and intermediate states.
Prompts are constrained to a two-token answer set such as " Yes"/" No".
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import int_model_approximation.__main__ as entry  # noqa: E402


FP8_MODELS = [
    "RedHatAI/Qwen2.5-0.5B-FP8-dynamic",
    "RedHatAI/Qwen2.5-1.5B-FP8-dynamic",
    "RedHatAI/Qwen2.5-3B-FP8-dynamic",
]

FP4_MODELS = [
    "pebeto/Qwen2.5-1.5B-Instruct-w4a4-fp4",
    "JongYeop/Qwen2.5-3B-Instruct-FP4-W4A4",
]


@dataclass(frozen=True)
class BinaryPrompt:
    prompt_id: str
    family: str
    answer_pair: tuple[str, str]
    expected: str
    prompt: str
    statement: str


@dataclass
class ForwardResult:
    logits: torch.Tensor
    seconds: float
    int32_kernel_calls: int = 0
    reference_scaled_mm_calls: int = 0


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _elapsed(start: float) -> float:
    _sync()
    return time.perf_counter() - start


def _require_device() -> str:
    device = entry._require_gpu()
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) < (8, 9):
        raise RuntimeError(f"SM89+ is required for this evaluator; found SM{major}{minor}")
    return device


def _freeze(model: torch.nn.Module) -> torch.nn.Module:
    return entry._freeze(model)


def _clear_model(model: torch.nn.Module | None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _normalize_pair(pair: tuple[str, str]) -> str:
    return f"{pair[0].strip()}/{pair[1].strip()}"


def _yes_no_prompt(prompt_id: str, family: str, statement: str, expected_yes: bool) -> BinaryPrompt:
    prompt = (
        "Answer with exactly one token: Yes or No.\n"
        f"Question: {statement}\n"
        "Answer:"
    )
    return BinaryPrompt(
        prompt_id=prompt_id,
        family=family,
        answer_pair=(" Yes", " No"),
        expected=" Yes" if expected_yes else " No",
        prompt=prompt,
        statement=statement,
    )


def _true_false_prompt(prompt_id: str, family: str, statement: str, expected_true: bool) -> BinaryPrompt:
    prompt = (
        "Respond with exactly one token: true or false.\n"
        f"Statement: {statement}\n"
        "Answer:"
    )
    return BinaryPrompt(
        prompt_id=prompt_id,
        family=family,
        answer_pair=(" true", " false"),
        expected=" true" if expected_true else " false",
        prompt=prompt,
        statement=statement,
    )


def _choice_prompt(
    prompt_id: str,
    family: str,
    question: str,
    option_a: str,
    option_b: str,
    expected: str,
) -> BinaryPrompt:
    prompt = (
        "Choose the correct option. Answer with exactly one token: A or B.\n"
        f"Question: {question}\n"
        f"A: {option_a}\n"
        f"B: {option_b}\n"
        "Answer:"
    )
    return BinaryPrompt(
        prompt_id=prompt_id,
        family=family,
        answer_pair=(" A", " B"),
        expected=f" {expected}",
        prompt=prompt,
        statement=f"{question} A={option_a!r} B={option_b!r}",
    )


def _fewshot_yes_no(prompt_id: str, family: str, statement: str, expected_yes: bool) -> BinaryPrompt:
    prompt = (
        "Answer each question with exactly one token: Yes or No.\n"
        "Question: Is 2 + 2 equal to 4?\n"
        "Answer: Yes\n"
        "Question: Is 9 less than 3?\n"
        "Answer: No\n"
        f"Question: {statement}\n"
        "Answer:"
    )
    return BinaryPrompt(
        prompt_id=prompt_id,
        family=family,
        answer_pair=(" Yes", " No"),
        expected=" Yes" if expected_yes else " No",
        prompt=prompt,
        statement=statement,
    )


def _private_reason_prompt(
    prompt_id: str, family: str, statement: str, expected_yes: bool
) -> BinaryPrompt:
    prompt = (
        "Decide privately, then output only the final binary answer token.\n"
        "Allowed answers: Yes or No.\n"
        f"Question: {statement}\n"
        "Final answer:"
    )
    return BinaryPrompt(
        prompt_id=prompt_id,
        family=family,
        answer_pair=(" Yes", " No"),
        expected=" Yes" if expected_yes else " No",
        prompt=prompt,
        statement=statement,
    )


def _build_prompts() -> list[BinaryPrompt]:
    prompts: list[BinaryPrompt] = []

    yes_no_cases = [
        ("math_yes_01", "math_direct", "Is 17 + 24 equal to 41?", True),
        ("math_no_01", "math_direct", "Is 18 * 7 equal to 125?", False),
        ("math_yes_02", "math_direct", "Is 144 divided by 12 equal to 12?", True),
        ("math_no_02", "math_direct", "Is 31 a multiple of 6?", False),
        ("math_yes_03", "math_direct", "Is 2 to the fifth power equal to 32?", True),
        ("math_no_03", "math_direct", "Is 15 squared equal to 215?", False),
        ("logic_yes_01", "logic_direct", "If P is true and Q is false, is P OR Q true?", True),
        ("logic_no_01", "logic_direct", "If P is true and Q is false, is P AND Q true?", False),
        ("logic_yes_02", "logic_direct", "Is the negation of false true?", True),
        ("logic_no_02", "logic_direct", "Can a statement and its negation both be true?", False),
        ("fact_yes_01", "fact_direct", "Is Paris the capital of France?", True),
        ("fact_no_01", "fact_direct", "Is the Pacific Ocean smaller than Lake Erie?", False),
        ("fact_yes_02", "fact_direct", "Does water contain hydrogen?", True),
        ("fact_no_02", "fact_direct", "Is the chemical symbol for gold Ag?", False),
        ("fact_yes_03", "fact_direct", "Does a triangle have three sides?", True),
        ("fact_no_03", "fact_direct", "Is the Earth the fourth planet from the Sun?", False),
    ]
    for prompt_id, family, statement, expected in yes_no_cases:
        prompts.append(_yes_no_prompt(prompt_id, family, statement, expected))

    for prompt_id, base_family, statement, expected in yes_no_cases[:10]:
        prompts.append(_fewshot_yes_no(f"fewshot_{prompt_id}", f"fewshot_{base_family}", statement, expected))
        prompts.append(
            _private_reason_prompt(
                f"private_{prompt_id}",
                f"private_{base_family}",
                statement,
                expected,
            )
        )

    tf_cases = [
        ("tf_math_01", "true_false_math", "The integer 91 is divisible by 7.", True),
        ("tf_math_02", "true_false_math", "The integer 91 is divisible by 9.", False),
        ("tf_math_03", "true_false_math", "The product 13 * 11 is 143.", True),
        ("tf_math_04", "true_false_math", "The sum 58 + 67 is 116.", False),
        ("tf_logic_01", "true_false_logic", "If all squares are rectangles, then a square is a rectangle.", True),
        ("tf_logic_02", "true_false_logic", "If all cats are mammals, then all mammals are cats.", False),
        ("tf_fact_01", "true_false_fact", "The freezing point of pure water at sea level is about 0 degrees Celsius.", True),
        ("tf_fact_02", "true_false_fact", "The Moon is larger than the Earth.", False),
        ("tf_fact_03", "true_false_fact", "A byte is commonly eight bits.", True),
        ("tf_fact_04", "true_false_fact", "The Roman numeral X represents five.", False),
    ]
    for prompt_id, family, statement, expected in tf_cases:
        prompts.append(_true_false_prompt(prompt_id, family, statement, expected))

    choice_cases = [
        ("choice_math_01", "choice_math", "Which number is larger?", "19", "23", "B"),
        ("choice_math_02", "choice_math", "Which expression equals 42?", "6 * 7", "8 * 6", "A"),
        ("choice_math_03", "choice_math", "Which number is even?", "37", "58", "B"),
        ("choice_math_04", "choice_math", "Which fraction is greater?", "3/4", "2/3", "A"),
        ("choice_logic_01", "choice_logic", "Which value is the result of true AND false?", "true", "false", "B"),
        ("choice_logic_02", "choice_logic", "Which value is the result of false OR true?", "true", "false", "A"),
        ("choice_fact_01", "choice_fact", "Which planet is known as the Red Planet?", "Mars", "Venus", "A"),
        ("choice_fact_02", "choice_fact", "Which animal is a mammal?", "salmon", "dolphin", "B"),
        ("choice_fact_03", "choice_fact", "Which unit measures electric current?", "ampere", "pascal", "A"),
        ("choice_fact_04", "choice_fact", "Which language is primarily written with the Greek alphabet?", "Greek", "Thai", "A"),
        ("choice_reading_01", "choice_reading", "In the phrase 'red square', which word names a color?", "red", "square", "A"),
        ("choice_reading_02", "choice_reading", "In the phrase 'cold water', which word names a substance?", "cold", "water", "B"),
    ]
    for prompt_id, family, question, option_a, option_b, expected in choice_cases:
        prompts.append(_choice_prompt(prompt_id, family, question, option_a, option_b, expected))

    return prompts


def _select_prompts(prompts: list[BinaryPrompt], limit: int | None) -> list[BinaryPrompt]:
    if limit is None or limit >= len(prompts):
        return prompts
    return prompts[:limit]


def _candidate_token_ids(tokenizer: Any, prompts: Iterable[BinaryPrompt]) -> dict[str, int]:
    values = sorted({token for prompt in prompts for token in prompt.answer_pair} | {p.expected for p in prompts})
    mapping: dict[str, int] = {}
    bad: list[tuple[str, list[int]]] = []
    for value in values:
        ids = tokenizer.encode(value, add_special_tokens=False)
        if len(ids) != 1:
            bad.append((value, ids))
        else:
            mapping[value] = ids[0]
    if bad:
        details = ", ".join(f"{value!r}->{ids}" for value, ids in bad)
        raise RuntimeError(f"answer tokens must be single tokenizer tokens: {details}")
    return mapping


def _batched(items: list[BinaryPrompt], batch_size: int) -> Iterable[list[BinaryPrompt]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _extract_final_logits(output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    logits = entry._extract_logits(output)
    last_indices = attention_mask.sum(dim=1).to(torch.long) - 1
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, last_indices, :].detach().to(torch.float32).cpu()


def _forward_prompt_logits(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[BinaryPrompt],
    device: str,
    batch_size: int,
    *,
    count_int32: bool,
    count_scaled_mm: bool,
) -> ForwardResult:
    all_logits: list[torch.Tensor] = []
    int32_calls = 0
    scaled_mm_calls = 0
    start = time.perf_counter()

    with torch.inference_mode():
        for batch in _batched(prompts, batch_size):
            encoded = tokenizer(
                [item.prompt for item in batch],
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            kwargs = {"attention_mask": attention_mask, "use_cache": False}
            if count_int32:
                with entry._Int32KernelProbe() as probe:
                    output = model(input_ids, **kwargs)
                int32_calls += probe.count
            elif count_scaled_mm:
                with entry._KernelProbe("_scaled_mm") as probe:
                    output = model(input_ids, **kwargs)
                scaled_mm_calls += probe.count
            else:
                output = model(input_ids, **kwargs)
            all_logits.append(_extract_final_logits(output, attention_mask))

    return ForwardResult(
        logits=torch.cat(all_logits, dim=0),
        seconds=_elapsed(start),
        int32_kernel_calls=int32_calls,
        reference_scaled_mm_calls=scaled_mm_calls,
    )


def _load_fp8_teacher(model_id: str, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    entry.MODEL_ID = model_id
    entry.TEACHER_KERNEL = "fp8_scaled_mm"
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    entry._disable_compressed_tensor_hooks(model)
    replaced = entry._replace_fp8_linears(model)
    return _freeze(model), {
        "load_seconds": _elapsed(load_start),
        "fp8_linears": len(replaced),
        "linears_replaced": len(replaced),
    }


def _load_direct_int32_student(model_id: str, device: str, *, compressed_checkpoint: bool) -> tuple[torch.nn.Module, dict[str, Any]]:
    entry.MODEL_ID = model_id
    entry.STUDENT_KERNEL = "codebook"
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        low_cpu_mem_usage=not compressed_checkpoint,
    ).to(device)
    if compressed_checkpoint:
        _force_decompress_if_needed(model, device)
    else:
        entry._disable_compressed_tensor_hooks(model)
    replaced = entry._replace_int32_linears(model)
    return _freeze(model), {
        "load_seconds": _elapsed(load_start),
        "linears_replaced": len(replaced),
    }


def _force_decompress_if_needed(model: torch.nn.Module, device: str) -> None:
    """Trigger compressed-tensors' one-shot decompression hook if present."""
    if not hasattr(model, "ct_decompress_hook"):
        return
    input_ids = torch.tensor([[0]], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        model(input_ids, attention_mask=attention_mask, use_cache=False)
    if hasattr(model, "ct_decompress_hook"):
        raise RuntimeError("compressed-tensors decompression hook did not remove itself")


def _load_decompressed_teacher(model_id: str, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        low_cpu_mem_usage=False,
    ).to(device)
    _force_decompress_if_needed(model, device)
    linears = sum(1 for module in model.modules() if isinstance(module, torch.nn.Linear))
    return _freeze(model), {
        "load_seconds": _elapsed(load_start),
        "decompressed_linears": linears,
    }


def _score_prompts(
    prompts: list[BinaryPrompt],
    token_ids: dict[str, int],
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        pair_ids = [token_ids[prompt.answer_pair[0]], token_ids[prompt.answer_pair[1]]]
        expected_id = token_ids[prompt.expected]
        teacher_pair = teacher_logits[idx, pair_ids]
        student_pair = student_logits[idx, pair_ids]
        teacher_idx = int(torch.argmax(teacher_pair).item())
        student_idx = int(torch.argmax(student_pair).item())
        teacher_token = prompt.answer_pair[teacher_idx]
        student_token = prompt.answer_pair[student_idx]
        top_token_id = pair_ids[teacher_idx]
        teacher_top_logit = float(teacher_logits[idx, top_token_id].item())
        student_top_logit = float(student_logits[idx, top_token_id].item())
        signed_error = student_top_logit - teacher_top_logit
        teacher_margin = float((teacher_pair[teacher_idx] - teacher_pair[1 - teacher_idx]).item())
        student_margin_same_orientation = float(
            (student_pair[teacher_idx] - student_pair[1 - teacher_idx]).item()
        )
        expected_pair_idx = pair_ids.index(expected_id)
        rows.append(
            {
                "prompt_id": prompt.prompt_id,
                "family": prompt.family,
                "answer_pair": _normalize_pair(prompt.answer_pair),
                "expected": prompt.expected.strip(),
                "teacher_binary_top1": teacher_token.strip(),
                "student_binary_top1": student_token.strip(),
                "teacher_matches_expected": teacher_idx == expected_pair_idx,
                "student_matches_expected": student_idx == expected_pair_idx,
                "binary_top1_agreement": teacher_idx == student_idx,
                "teacher_top1_token": teacher_token,
                "teacher_top1_token_id": int(top_token_id),
                "teacher_top1_logit": teacher_top_logit,
                "student_logit_on_teacher_top1": student_top_logit,
                "signed_top1_logit_error": signed_error,
                "abs_top1_logit_error": abs(signed_error),
                "teacher_binary_margin": teacher_margin,
                "student_margin_on_teacher_orientation": student_margin_same_orientation,
                "signed_margin_error": student_margin_same_orientation - teacher_margin,
                "abs_margin_error": abs(student_margin_same_orientation - teacher_margin),
                "prompt_tokens": None,
                "statement": prompt.statement,
                "prompt": prompt.prompt,
            }
        )
    return rows


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    values_sorted = sorted(values)
    pos = (len(values_sorted) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values_sorted[lo]
    return values_sorted[lo] * (hi - pos) + values_sorted[hi] * (pos - lo)


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [float(r["abs_top1_logit_error"]) for r in rows]
    signed = [float(r["signed_top1_logit_error"]) for r in rows]
    margins = [float(r["teacher_binary_margin"]) for r in rows]
    margin_errors = [float(r["abs_margin_error"]) for r in rows]
    agreements = [bool(r["binary_top1_agreement"]) for r in rows]
    teacher_correct = [bool(r["teacher_matches_expected"]) for r in rows]
    student_correct = [bool(r["student_matches_expected"]) for r in rows]
    return {
        "n": len(rows),
        "binary_top1_agreement": sum(agreements) / len(rows) if rows else float("nan"),
        "teacher_expected_accuracy": sum(teacher_correct) / len(rows) if rows else float("nan"),
        "student_expected_accuracy": sum(student_correct) / len(rows) if rows else float("nan"),
        "abs_top1_logit_error_mean": statistics.fmean(errors) if errors else float("nan"),
        "abs_top1_logit_error_median": statistics.median(errors) if errors else float("nan"),
        "abs_top1_logit_error_p90": _quantile(errors, 0.90),
        "abs_top1_logit_error_p99": _quantile(errors, 0.99),
        "abs_top1_logit_error_max": max(errors) if errors else float("nan"),
        "signed_top1_logit_error_mean": statistics.fmean(signed) if signed else float("nan"),
        "teacher_binary_margin_mean": statistics.fmean(margins) if margins else float("nan"),
        "teacher_binary_margin_median": statistics.median(margins) if margins else float("nan"),
        "abs_margin_error_mean": statistics.fmean(margin_errors) if margin_errors else float("nan"),
        "abs_margin_error_p90": _quantile(margin_errors, 0.90),
    }


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return {name: _summarize_rows(group_rows) for name, group_rows in sorted(grouped.items())}


def _margin_bucket_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets = {
        "low_margin_<1": [],
        "medium_margin_1_to_5": [],
        "high_margin_>=5": [],
    }
    for row in rows:
        margin = float(row["teacher_binary_margin"])
        if margin < 1.0:
            buckets["low_margin_<1"].append(row)
        elif margin < 5.0:
            buckets["medium_margin_1_to_5"].append(row)
        else:
            buckets["high_margin_>=5"].append(row)
    return {name: _summarize_rows(bucket_rows) for name, bucket_rows in buckets.items() if bucket_rows}


def _write_markdown_report(result: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Binary Answer Int32 Error Report")
    lines.append("")
    lines.append(f"- Device: {result['device']}")
    lines.append(f"- Prompt count: {result['prompt_count']}")
    lines.append("- Metric: student minus teacher logit on the teacher's top binary answer token")
    lines.append("- FP4 runs use the decompressed FP4 checkpoint as teacher on this SM89 GPU.")
    lines.append(f"- Started UTC: {result['started_utc']}")
    lines.append(f"- Finished UTC: {result['finished_utc']}")
    failed_runs = [run for run in result["runs"] if "summary" not in run]
    if failed_runs:
        lines.append("")
        lines.append("## Failed Runs")
        for run in failed_runs:
            lines.append(f"- `{run['model_id']}` ({run['experiment']}): {run.get('error', 'unknown error')}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(
        "| experiment | model | n | agree | mean abs err | p90 | p99 | max | signed mean | teacher margin | teacher acc | student acc |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run in result["runs"]:
        s = run["summary"]
        lines.append(
            "| "
            f"{run['experiment']} | `{run['model_id']}` | {s['n']} | "
            f"{s['binary_top1_agreement']:.3f} | {s['abs_top1_logit_error_mean']:.4g} | "
            f"{s['abs_top1_logit_error_p90']:.4g} | {s['abs_top1_logit_error_p99']:.4g} | "
            f"{s['abs_top1_logit_error_max']:.4g} | {s['signed_top1_logit_error_mean']:.4g} | "
            f"{s['teacher_binary_margin_mean']:.4g} | {s['teacher_expected_accuracy']:.3f} | "
            f"{s['student_expected_accuracy']:.3f} |"
        )
    lines.append("")
    lines.append("## By Answer Token Pair")
    for run in result["runs"]:
        lines.append("")
        lines.append(f"### {run['experiment']} `{run['model_id']}`")
        lines.append("")
        lines.append("| pair | n | agree | mean abs err | p90 | signed mean | teacher margin |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for pair, s in run["by_answer_pair"].items():
            lines.append(
                f"| {pair} | {s['n']} | {s['binary_top1_agreement']:.3f} | "
                f"{s['abs_top1_logit_error_mean']:.4g} | {s['abs_top1_logit_error_p90']:.4g} | "
                f"{s['signed_top1_logit_error_mean']:.4g} | {s['teacher_binary_margin_mean']:.4g} |"
            )
    lines.append("")
    lines.append("## By Teacher Confidence")
    for run in result["runs"]:
        lines.append("")
        lines.append(f"### {run['experiment']} `{run['model_id']}`")
        lines.append("")
        lines.append("| margin bucket | n | agree | mean abs err | p90 | max |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for bucket, s in run["by_margin_bucket"].items():
            lines.append(
                f"| {bucket} | {s['n']} | {s['binary_top1_agreement']:.3f} | "
                f"{s['abs_top1_logit_error_mean']:.4g} | {s['abs_top1_logit_error_p90']:.4g} | "
                f"{s['abs_top1_logit_error_max']:.4g} |"
            )
    lines.append("")
    lines.append("## Worst Cases")
    for run in result["runs"]:
        lines.append("")
        lines.append(f"### {run['experiment']} `{run['model_id']}`")
        for row in run["worst_cases"][:8]:
            lines.append(
                "- "
                f"{row['prompt_id']} ({row['answer_pair']}): abs_err={row['abs_top1_logit_error']:.4g}, "
                f"teacher={row['teacher_binary_top1']}, student={row['student_binary_top1']}, "
                f"teacher_margin={row['teacher_binary_margin']:.4g}, statement={row['statement']}"
            )
    lines.append("")
    path.write_text("\n".join(lines))


def _run_one(
    experiment: str,
    model_id: str,
    prompts: list[BinaryPrompt],
    tokenizer: Any,
    token_ids: dict[str, int],
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    compressed_checkpoint = experiment == "fp4_decompressed_checkpoint_vs_int32"
    teacher: torch.nn.Module | None = None
    student: torch.nn.Module | None = None
    teacher_meta: dict[str, Any] = {}
    student_meta: dict[str, Any] = {}
    try:
        if experiment == "fp8_scaled_mm_vs_int32":
            teacher, teacher_meta = _load_fp8_teacher(model_id, device)
            teacher_result = _forward_prompt_logits(
                teacher,
                tokenizer,
                prompts,
                device,
                batch_size,
                count_int32=False,
                count_scaled_mm=True,
            )
        elif experiment == "fp4_decompressed_checkpoint_vs_int32":
            teacher, teacher_meta = _load_decompressed_teacher(model_id, device)
            teacher_result = _forward_prompt_logits(
                teacher,
                tokenizer,
                prompts,
                device,
                batch_size,
                count_int32=False,
                count_scaled_mm=False,
            )
        else:
            raise RuntimeError(f"unknown experiment {experiment!r}")
    finally:
        _clear_model(teacher)

    try:
        student, student_meta = _load_direct_int32_student(
            model_id,
            device,
            compressed_checkpoint=compressed_checkpoint,
        )
        student_result = _forward_prompt_logits(
            student,
            tokenizer,
            prompts,
            device,
            batch_size,
            count_int32=True,
            count_scaled_mm=False,
        )
    finally:
        _clear_model(student)

    rows = _score_prompts(prompts, token_ids, teacher_result.logits, student_result.logits)
    for row, prompt in zip(rows, prompts, strict=True):
        row["prompt_tokens"] = len(tokenizer.encode(prompt.prompt, add_special_tokens=True))

    summary = _summarize_rows(rows)
    worst = sorted(rows, key=lambda row: float(row["abs_top1_logit_error"]), reverse=True)
    teacher_forward_meta = {
        "forward_seconds": teacher_result.seconds,
        "int32_kernel_calls": teacher_result.int32_kernel_calls,
        "reference_scaled_mm_calls": teacher_result.reference_scaled_mm_calls,
    }
    student_forward_meta = {
        "forward_seconds": student_result.seconds,
        "int32_kernel_calls": student_result.int32_kernel_calls,
        "reference_scaled_mm_calls": student_result.reference_scaled_mm_calls,
    }
    return {
        "experiment": experiment,
        "model_id": model_id,
        "teacher": teacher_meta | teacher_forward_meta,
        "student": student_meta | student_forward_meta,
        "summary": summary,
        "by_answer_pair": _group_summary(rows, "answer_pair"),
        "by_family": _group_summary(rows, "family"),
        "by_margin_bucket": _margin_bucket_summary(rows),
        "worst_cases": worst[:20],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp8-model", action="append", default=[])
    parser.add_argument("--fp4-model", action="append", default=[])
    parser.add_argument("--skip-fp8", action="store_true")
    parser.add_argument("--skip-fp4", action="store_true")
    parser.add_argument("--prompt-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("results/binary_answer_int32_error.json"))
    parser.add_argument("--report", type=Path, default=Path("results/binary_answer_int32_error.md"))
    args = parser.parse_args()

    device = _require_device()
    torch.manual_seed(0)
    prompts = _select_prompts(_build_prompts(), args.prompt_limit)
    if not prompts:
        raise RuntimeError("no prompts selected")

    fp8_models = args.fp8_model or FP8_MODELS
    fp4_models = args.fp4_model or FP4_MODELS
    tokenizer_model = (fp8_models or fp4_models)[0]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    token_ids = _candidate_token_ids(tokenizer, prompts)

    result: dict[str, Any] = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished_utc": None,
        "device": torch.cuda.get_device_name(0),
        "prompt_count": len(prompts),
        "answer_token_ids": token_ids,
        "prompts": [asdict(prompt) for prompt in prompts],
        "runs": [],
    }

    for model_id in fp8_models:
        if args.skip_fp8:
            break
        print(f"[binary-eval] fp8 teacher vs int32 student: {model_id}")
        try:
            run = _run_one(
                "fp8_scaled_mm_vs_int32",
                model_id,
                prompts,
                tokenizer,
                token_ids,
                device,
                args.batch_size,
            )
        except Exception as exc:
            run = {
                "experiment": "fp8_scaled_mm_vs_int32",
                "model_id": model_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[binary-eval] fp8 run failed for {model_id}: {run['error']}")
        result["runs"].append(run)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))

    for model_id in fp4_models:
        if args.skip_fp4:
            break
        print(f"[binary-eval] fp4 checkpoint/decompressed teacher vs int32 student: {model_id}")
        try:
            model_tokenizer = AutoTokenizer.from_pretrained(model_id)
            if model_tokenizer.pad_token is None:
                model_tokenizer.pad_token = model_tokenizer.eos_token
            model_tokenizer.padding_side = "right"
            model_token_ids = _candidate_token_ids(model_tokenizer, prompts)
            run = _run_one(
                "fp4_decompressed_checkpoint_vs_int32",
                model_id,
                prompts,
                model_tokenizer,
                model_token_ids,
                device,
                args.batch_size,
            )
        except Exception as exc:
            run = {
                "experiment": "fp4_decompressed_checkpoint_vs_int32",
                "model_id": model_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[binary-eval] fp4 run failed for {model_id}: {run['error']}")
        result["runs"].append(run)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))

    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    reportable = {**result, "runs": [run for run in result["runs"] if "summary" in run]}
    _write_markdown_report(reportable, args.report)
    print(f"[binary-eval] wrote {args.output}")
    print(f"[binary-eval] wrote {args.report}")


if __name__ == "__main__":
    main()
