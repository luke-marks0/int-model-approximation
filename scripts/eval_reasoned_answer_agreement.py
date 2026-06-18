"""Evaluate FP-vs-int agreement after free-form reasoning and parsed final answers."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.eval_binary_answer_int32 as binary_eval  # noqa: E402


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
class ReasonedPrompt:
    prompt_id: str
    dataset: str
    family: str
    expected: str
    allowed: tuple[str, str]
    option_values: dict[str, str]
    statement: str
    prompt: str


def _load_jsonl(path: Path, dataset: str, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if limit is not None and len(rows) >= limit:
                break
            row = json.loads(line)
            row["dataset"] = dataset
            rows.append(row)
    return rows


def _extract_question_block(original_prompt: str) -> str:
    lines = []
    for raw_line in original_prompt.splitlines():
        line = raw_line.rstrip()
        if (
            line.startswith("Question:")
            or line.startswith("Statement:")
            or line.startswith("A:")
            or line.startswith("B:")
        ):
            lines.append(line)
    if not lines:
        raise RuntimeError(f"could not extract question block from prompt: {original_prompt!r}")
    return "\n".join(lines)


def _reasoned_prompt(row: dict[str, Any]) -> ReasonedPrompt:
    allowed = tuple(token.strip() for token in row["answer_pair"])
    allowed_text = " or ".join(allowed)
    final_examples = " or ".join(f"Final answer: {label}" for label in allowed)
    question_block = _extract_question_block(row["prompt"])
    option_values: dict[str, str] = {}
    for line in question_block.splitlines():
        if re.match(r"^[AB]:", line):
            label, value = line.split(":", 1)
            option_values[label.strip()] = value.strip()
    prompt = (
        "Think briefly before answering, using at most two short sentences. "
        "Then put the final answer on a new line using only the requested label.\n"
        f"Allowed final answers: {allowed_text}.\n"
        f"End with exactly one of these forms: {final_examples}.\n\n"
        f"{question_block}\n\n"
        "Solution:"
    )
    return ReasonedPrompt(
        prompt_id=row["prompt_id"],
        dataset=row["dataset"],
        family=row["family"],
        expected=row["expected"].strip(),
        allowed=allowed,  # type: ignore[arg-type]
        option_values=option_values,
        statement=row["statement"],
        prompt=prompt,
    )


def _candidate_regex(allowed: tuple[str, str]) -> str:
    if set(allowed) == {"A", "B"}:
        return r"(A|B)"
    if {a.lower() for a in allowed} == {"yes", "no"}:
        return r"(Yes|No|yes|no)"
    if {a.lower() for a in allowed} == {"true", "false"}:
        return r"(true|false|True|False)"
    escaped = "|".join(re.escape(a) for a in allowed)
    return rf"({escaped})"


def _normalize_answer(value: str, allowed: tuple[str, str]) -> str:
    for label in allowed:
        if value.lower() == label.lower():
            return label
    return value


def _parse_answer(
    text: str,
    allowed: tuple[str, str],
    option_values: dict[str, str],
) -> dict[str, Any]:
    candidate = _candidate_regex(allowed)
    patterns = [
        rf"final\s+answer\s*[:\-]\s*(?:option\s+)?{candidate}\b",
        rf"answer\s*[:\-]\s*(?:option\s+)?{candidate}\b",
        rf"(?:correct|final)\s+answer\s+is\s+(?:option\s+)?{candidate}\b",
        rf"therefore[^.\n]*\b(?:option\s+)?{candidate}\b",
        rf"\boption\s+{candidate}\b",
        rf"\b(?:option\s+)?{candidate}\s*[:\)]",
        rf"\b{candidate}\b",
    ]
    matches: list[tuple[int, int, str, str]] = []
    for pattern_index, pattern in enumerate(patterns):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = match.group(1)
            matches.append((pattern_index, match.start(), value, match.group(0)))
    if not matches:
        value_match = _parse_option_value_answer(text, option_values)
        if value_match is not None:
            return value_match
        return {"answer": None, "method": "unparsed", "matched_text": None}
    explicit = [item for item in matches if item[0] <= 3]
    source = explicit[-1] if explicit else matches[-1]
    return {
        "answer": _normalize_answer(source[2], allowed),
        "method": f"pattern_{source[0]}",
        "matched_text": source[3],
    }


def _parse_option_value_answer(
    text: str,
    option_values: dict[str, str],
) -> dict[str, Any] | None:
    if not option_values:
        return None
    matches: list[tuple[int, str, str]] = []
    for label, value in option_values.items():
        aliases = _option_value_aliases(value)
        if not aliases:
            continue
        for alias in aliases:
            escaped = re.escape(alias)
            patterns = [
                rf"final\s+answer\s*[:\-]\s*{escaped}\b",
                rf"answer\s+is\s+{escaped}\b",
                rf"(?:target|correct)\s+(?:option|answer)\s+is\s+['\"]?{escaped}\b",
                rf"matches[^.\n]*\b{escaped}\b",
                rf"\b{escaped}\b",
            ]
            for pattern in patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    matches.append((match.start(), label, match.group(0)))
    if not matches:
        return None
    source = sorted(matches, key=lambda item: item[0])[-1]
    return {
        "answer": source[1],
        "method": "option_value",
        "matched_text": source[2],
    }


def _option_value_aliases(value: str) -> list[str]:
    normalized = value.strip()
    if not normalized:
        return []
    aliases = [normalized]
    lowered = normalized.lower()
    for prefix in ["target option ", "distractor option "]:
        if lowered.startswith(prefix):
            aliases.append(normalized[len(prefix) :].strip())
    return [alias for alias in dict.fromkeys(aliases) if alias]


def _clear_model(model: torch.nn.Module | None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _batched(items: list[ReasonedPrompt], batch_size: int) -> list[list[ReasonedPrompt]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _generate_texts(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[ReasonedPrompt],
    device: str,
    batch_size: int,
    max_new_tokens: int,
) -> tuple[list[str], float]:
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    texts: list[str] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in _batched(prompts, batch_size):
            encoded = tokenizer(
                [item.prompt for item in batch],
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            ).to(device)
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            continuation = output[:, encoded.input_ids.shape[1] :]
            for row in continuation:
                texts.append(tokenizer.decode(row, skip_special_tokens=True).strip())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    tokenizer.padding_side = old_padding_side
    return texts, time.perf_counter() - start


def _load_teacher(experiment: str, model_id: str, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    if experiment == "fp8_scaled_mm_vs_int32":
        return binary_eval._load_fp8_teacher(model_id, device)
    if experiment == "fp4_decompressed_checkpoint_vs_int32":
        return binary_eval._load_decompressed_teacher(model_id, device)
    raise RuntimeError(f"unknown experiment {experiment}")


def _run_one(
    experiment: str,
    model_id: str,
    prompts: list[ReasonedPrompt],
    device: str,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher: torch.nn.Module | None = None
    student: torch.nn.Module | None = None
    try:
        teacher, teacher_meta = _load_teacher(experiment, model_id, device)
        teacher_texts, teacher_seconds = _generate_texts(
            teacher,
            tokenizer,
            prompts,
            device,
            batch_size,
            max_new_tokens,
        )
    finally:
        _clear_model(teacher)

    try:
        student, student_meta = binary_eval._load_direct_int32_student(
            model_id,
            device,
            compressed_checkpoint=experiment == "fp4_decompressed_checkpoint_vs_int32",
        )
        student_texts, student_seconds = _generate_texts(
            student,
            tokenizer,
            prompts,
            device,
            batch_size,
            max_new_tokens,
        )
    finally:
        _clear_model(student)

    rows = []
    for prompt, teacher_text, student_text in zip(prompts, teacher_texts, student_texts, strict=True):
        teacher_parse = _parse_answer(teacher_text, prompt.allowed, prompt.option_values)
        student_parse = _parse_answer(student_text, prompt.allowed, prompt.option_values)
        teacher_answer = teacher_parse["answer"]
        student_answer = student_parse["answer"]
        rows.append(
            {
                "prompt_id": prompt.prompt_id,
                "dataset": prompt.dataset,
                "family": prompt.family,
                "expected": prompt.expected,
                "allowed": list(prompt.allowed),
                "statement": prompt.statement,
                "teacher_answer": teacher_answer,
                "student_answer": student_answer,
                "teacher_parse_method": teacher_parse["method"],
                "student_parse_method": student_parse["method"],
                "teacher_matched_text": teacher_parse["matched_text"],
                "student_matched_text": student_parse["matched_text"],
                "teacher_correct": teacher_answer == prompt.expected,
                "student_correct": student_answer == prompt.expected,
                "answer_agreement": teacher_answer is not None and teacher_answer == student_answer,
                "both_parsed": teacher_answer is not None and student_answer is not None,
                "teacher_text": teacher_text,
                "student_text": student_text,
            }
        )

    return {
        "experiment": experiment,
        "model_id": model_id,
        "teacher": teacher_meta | {"generation_seconds": teacher_seconds},
        "student": student_meta | {"generation_seconds": student_seconds},
        "summary_by_dataset": {
            name: _summarize([row for row in rows if row["dataset"] == name])
            for name in ["easy", "hard"]
        },
        "rows": rows,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    both = [row for row in rows if row["both_parsed"]]
    return {
        "n": n,
        "both_parsed": len(both) / n if n else 0.0,
        "teacher_parse_rate": sum(row["teacher_answer"] is not None for row in rows) / n if n else 0.0,
        "student_parse_rate": sum(row["student_answer"] is not None for row in rows) / n if n else 0.0,
        "answer_agreement_all": sum(row["answer_agreement"] for row in rows) / n if n else 0.0,
        "answer_agreement_when_both_parsed": (
            sum(row["answer_agreement"] for row in both) / len(both) if both else 0.0
        ),
        "teacher_accuracy_all": sum(row["teacher_correct"] for row in rows) / n if n else 0.0,
        "student_accuracy_all": sum(row["student_correct"] for row in rows) / n if n else 0.0,
        "teacher_accuracy_when_parsed": (
            sum(row["teacher_correct"] for row in rows if row["teacher_answer"] is not None)
            / max(1, sum(row["teacher_answer"] is not None for row in rows))
        ),
        "student_accuracy_when_parsed": (
            sum(row["student_correct"] for row in rows if row["student_answer"] is not None)
            / max(1, sum(row["student_answer"] is not None for row in rows))
        ),
    }


def _write_report(result: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Reasoned Final-Answer Agreement Report")
    lines.append("")
    lines.append(f"- Easy prompts evaluated: {result['dataset_counts']['easy']}")
    lines.append(f"- Hard prompts evaluated: {result['dataset_counts']['hard']}")
    lines.append(f"- Generation batch size: {result['batch_size']}")
    lines.append(f"- Max new tokens: {result['max_new_tokens']}")
    lines.append("- Metric: parse only the final answer label from generated text; ignore reasoning text.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| dataset | experiment | model | both parsed | answer agreement | teacher acc | student acc | teacher gen s | student gen s |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for dataset in ["easy", "hard"]:
        for run in result["runs"]:
            summary = run["summary_by_dataset"][dataset]
            lines.append(
                f"| {dataset} | {run['experiment']} | `{run['model_id']}` | "
                f"{summary['both_parsed']:.3f} | {summary['answer_agreement_when_both_parsed']:.3f} | "
                f"{summary['teacher_accuracy_all']:.3f} | {summary['student_accuracy_all']:.3f} | "
                f"{run['teacher']['generation_seconds']:.1f} | {run['student']['generation_seconds']:.1f} |"
            )
    lines.append("")
    lines.append("## Disagreement Examples")
    for run in result["runs"]:
        disagreements = [
            row for row in run["rows"]
            if row["both_parsed"] and not row["answer_agreement"]
        ]
        if not disagreements:
            continue
        lines.append("")
        lines.append(f"### `{run['model_id']}`")
        for row in disagreements[:8]:
            lines.append(
                f"- {row['dataset']} `{row['prompt_id']}` expected={row['expected']} "
                f"teacher={row['teacher_answer']} student={row['student_answer']} "
                f"statement={row['statement']}"
            )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--easy-dataset", type=Path, default=Path("results/easy_binary_dataset.jsonl"))
    parser.add_argument("--hard-dataset", type=Path, default=Path("results/hard_binary_dataset.jsonl"))
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output", type=Path, default=Path("results/reasoned_answer_agreement.json"))
    parser.add_argument("--report", type=Path, default=Path("results/reasoned_answer_agreement.md"))
    parser.add_argument("--fp8-model", action="append", default=[])
    parser.add_argument("--fp4-model", action="append", default=[])
    parser.add_argument("--skip-fp4", action="store_true")
    args = parser.parse_args()

    device = binary_eval._require_device()
    easy_rows = _load_jsonl(args.easy_dataset, "easy", args.limit_per_dataset)
    hard_rows = _load_jsonl(args.hard_dataset, "hard", args.limit_per_dataset)
    prompts = [_reasoned_prompt(row) for row in [*easy_rows, *hard_rows]]
    fp8_models = args.fp8_model or FP8_MODELS
    fp4_models = [] if args.skip_fp4 else (args.fp4_model or FP4_MODELS)
    models = [
        *[("fp8_scaled_mm_vs_int32", model_id) for model_id in fp8_models],
        *[("fp4_decompressed_checkpoint_vs_int32", model_id) for model_id in fp4_models],
    ]

    result: dict[str, Any] = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished_utc": None,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "dataset_counts": {"easy": len(easy_rows), "hard": len(hard_rows)},
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for experiment, model_id in models:
        print(f"[reasoned] evaluating {model_id}", flush=True)
        run = _run_one(
            experiment,
            model_id,
            prompts,
            device,
            args.batch_size,
            args.max_new_tokens,
        )
        result["runs"].append(run)
        args.output.write_text(json.dumps(result, indent=2))

    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.output.write_text(json.dumps(result, indent=2))
    _write_report(result, args.report)
    print(f"[reasoned] wrote {args.output}")
    print(f"[reasoned] wrote {args.report}")


if __name__ == "__main__":
    main()
