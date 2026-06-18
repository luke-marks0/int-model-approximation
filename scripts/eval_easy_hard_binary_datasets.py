"""Generate easy/hard binary datasets and evaluate direct-int32 logit error."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict
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


def _choice_prompt(
    prompt_id: str,
    family: str,
    question: str,
    option_a: str,
    option_b: str,
    expected: str,
) -> binary_eval.BinaryPrompt:
    return binary_eval.BinaryPrompt(
        prompt_id=prompt_id,
        family=family,
        answer_pair=(" A", " B"),
        expected=f" {expected}",
        prompt=(
            "Choose the correct option. Answer with exactly one token: A or B.\n"
            f"Question: {question}\n"
            f"A: {option_a}\n"
            f"B: {option_b}\n"
            "Answer:"
        ),
        statement=f"{question} A={option_a!r} B={option_b!r}",
    )


def _yes_no_prompt(
    prompt_id: str,
    family: str,
    statement: str,
    expected_yes: bool,
) -> binary_eval.BinaryPrompt:
    return binary_eval.BinaryPrompt(
        prompt_id=prompt_id,
        family=family,
        answer_pair=(" Yes", " No"),
        expected=" Yes" if expected_yes else " No",
        prompt=(
            "Answer with exactly one token: Yes or No.\n"
            f"Question: {statement}\n"
            "Answer:"
        ),
        statement=statement,
    )


def _true_false_prompt(
    prompt_id: str,
    family: str,
    statement: str,
    expected_true: bool,
) -> binary_eval.BinaryPrompt:
    return binary_eval.BinaryPrompt(
        prompt_id=prompt_id,
        family=family,
        answer_pair=(" true", " false"),
        expected=" true" if expected_true else " false",
        prompt=(
            "Respond with exactly one token: true or false.\n"
            f"Statement: {statement}\n"
            "Answer:"
        ),
        statement=statement,
    )


def _generate_easy_candidates() -> list[binary_eval.BinaryPrompt]:
    words = [
        "red",
        "blue",
        "green",
        "yellow",
        "black",
        "white",
        "silver",
        "orange",
        "circle",
        "square",
        "triangle",
        "line",
        "point",
        "angle",
        "table",
        "chair",
        "window",
        "pencil",
        "paper",
        "clock",
        "river",
        "mountain",
        "forest",
        "desert",
        "apple",
        "banana",
        "grape",
        "lemon",
        "peach",
        "melon",
        "north",
        "south",
        "east",
        "west",
        "spring",
        "summer",
        "autumn",
        "winter",
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ]
    prompts: list[binary_eval.BinaryPrompt] = []
    idx = 0
    for i, target in enumerate(words):
        for j in range(1, min(9, len(words))):
            foil = words[(i + j * 5) % len(words)]
            if foil == target:
                continue
            expected = "A" if (i + j) % 2 == 0 else "B"
            a, b = (target, foil) if expected == "A" else (foil, target)
            prompts.append(
                _choice_prompt(
                    f"easy_exact_word_{idx:04d}",
                    "easy_exact_match_word",
                    f"Which option exactly matches the target word '{target}'?",
                    a,
                    b,
                    expected,
                )
            )
            idx += 1

    numbers = [str(n) for n in range(0, 200)]
    for i, target in enumerate(numbers):
        foil = str((i + 37) % 200)
        if foil == target:
            foil = str((i + 38) % 200)
        expected = "A" if i % 2 == 0 else "B"
        a, b = (target, foil) if expected == "A" else (foil, target)
        prompts.append(
            _choice_prompt(
                f"easy_exact_number_{i:04d}",
                "easy_exact_match_number",
                f"Which option exactly matches the target number {target}?",
                a,
                b,
                expected,
            )
        )

    for i in range(160):
        expected_true = i % 2 == 0
        left = "same" if expected_true else "different"
        right = "same"
        prompts.append(
            _true_false_prompt(
                f"easy_literal_tf_{i:04d}",
                "easy_literal_true_false",
                f"The word '{left}' is exactly the same word as '{right}'.",
                expected_true,
            )
        )

    for i, word in enumerate(words * 4):
        expected_yes = i % 2 == 0
        shown = word if expected_yes else words[(i + 13) % len(words)]
        prompts.append(
            _yes_no_prompt(
                f"easy_literal_yesno_{i:04d}",
                "easy_literal_yes_no",
                f"Is the displayed word '{shown}' exactly '{word}'?",
                expected_yes,
            )
        )

    return prompts


def _generate_hard_prompts(target_count: int) -> list[binary_eval.BinaryPrompt]:
    prompts: list[binary_eval.BinaryPrompt] = []
    idx = 0

    for i in range(90):
        a = 137 + 17 * i
        b = 246 + 19 * i
        correct = a + b
        foil = correct + (7 if i % 2 == 0 else -9)
        expected = "A" if i % 2 == 0 else "B"
        oa, ob = (str(correct), str(foil)) if expected == "A" else (str(foil), str(correct))
        prompts.append(
            _choice_prompt(
                f"hard_addition_{idx:04d}",
                "hard_arithmetic_addition",
                f"What is {a} + {b}?",
                oa,
                ob,
                expected,
            )
        )
        idx += 1

    for i in range(70):
        a = 23 + (i * 7) % 67
        b = 14 + (i * 11) % 83
        correct = a * b
        foil = correct + (a if i % 2 == 0 else -b)
        expected = "A" if i % 3 != 0 else "B"
        oa, ob = (str(correct), str(foil)) if expected == "A" else (str(foil), str(correct))
        prompts.append(
            _choice_prompt(
                f"hard_multiplication_{idx:04d}",
                "hard_arithmetic_multiplication",
                f"What is {a} multiplied by {b}?",
                oa,
                ob,
                expected,
            )
        )
        idx += 1

    for i in range(70):
        n = 10_000 + 137 * i + (i % 11) * 19
        d = [7, 9, 11, 13, 17][i % 5]
        expected = n % d == 0
        if i % 4 == 0:
            n = (n // d) * d
            expected = True
        prompts.append(
            _yes_no_prompt(
                f"hard_divisibility_{idx:04d}",
                "hard_divisibility",
                f"Is {n} divisible by {d} with no remainder?",
                expected,
            )
        )
        idx += 1

    for i in range(70):
        p = i % 2 == 0
        q = i % 3 == 0
        r = i % 5 == 0
        value = (p and not q) or (r and q)
        prompts.append(
            _true_false_prompt(
                f"hard_logic_{idx:04d}",
                "hard_symbolic_logic",
                (
                    f"Let P be {str(p).lower()}, Q be {str(q).lower()}, and R be "
                    f"{str(r).lower()}. The expression (P and not Q) or (R and Q) is true."
                ),
                value,
            )
        )
        idx += 1

    for i in range(70):
        red = 11 + (i * 7) % 41
        blue = 9 + (i * 5) % 37
        remove_red = i % 6
        add_blue = (i * 3) % 8
        final_red = red - remove_red
        final_blue = blue + add_blue
        expected = final_blue > final_red
        prompts.append(
            _yes_no_prompt(
                f"hard_word_problem_{idx:04d}",
                "hard_word_problem",
                (
                    f"A box starts with {red} red tokens and {blue} blue tokens. "
                    f"Then {remove_red} red tokens are removed and {add_blue} blue tokens are added. "
                    "Are there more blue tokens than red tokens at the end?"
                ),
                expected,
            )
        )
        idx += 1

    for i in range(70):
        values = [
            (37 + 13 * i) % 101,
            (71 + 17 * i) % 101,
            (19 + 23 * i) % 101,
            (53 + 29 * i) % 101,
        ]
        ordered = sorted(values)
        correct = ordered[-2]
        foil = ordered[1] if ordered[1] != correct else ordered[0]
        expected = "A" if i % 2 == 0 else "B"
        oa, ob = (str(correct), str(foil)) if expected == "A" else (str(foil), str(correct))
        prompts.append(
            _choice_prompt(
                f"hard_second_largest_{idx:04d}",
                "hard_ordering",
                f"Which option is the second-largest number in this list: {values}?",
                oa,
                ob,
                expected,
            )
        )
        idx += 1

    by_family: dict[str, list[binary_eval.BinaryPrompt]] = {}
    family_order: list[str] = []
    for prompt in prompts:
        if prompt.family not in by_family:
            by_family[prompt.family] = []
            family_order.append(prompt.family)
        by_family[prompt.family].append(prompt)

    interleaved: list[binary_eval.BinaryPrompt] = []
    offset = 0
    while len(interleaved) < len(prompts):
        added = False
        for family in family_order:
            family_prompts = by_family[family]
            if offset < len(family_prompts):
                interleaved.append(family_prompts[offset])
                added = True
        if not added:
            break
        offset += 1
    return interleaved[:target_count]


def _clear_model(model: torch.nn.Module | None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _teacher_logits_for_model(
    model_id: str,
    experiment: str,
    prompts: list[binary_eval.BinaryPrompt],
    device: str,
    batch_size: int,
) -> torch.Tensor:
    model: torch.nn.Module | None = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        if experiment == "fp8_scaled_mm_vs_int32":
            model, _ = binary_eval._load_fp8_teacher(model_id, device)
            result = binary_eval._forward_prompt_logits(
                model,
                tokenizer,
                prompts,
                device,
                batch_size=batch_size,
                count_int32=False,
                count_scaled_mm=True,
            )
        elif experiment == "fp4_decompressed_checkpoint_vs_int32":
            model, _ = binary_eval._load_decompressed_teacher(model_id, device)
            result = binary_eval._forward_prompt_logits(
                model,
                tokenizer,
                prompts,
                device,
                batch_size=batch_size,
                count_int32=False,
                count_scaled_mm=False,
            )
        else:
            raise RuntimeError(f"unknown experiment {experiment}")
        return result.logits
    finally:
        _clear_model(model)


def _expected_margin(
    tokenizer: Any,
    prompt: binary_eval.BinaryPrompt,
    logits: torch.Tensor,
) -> float:
    token_ids = binary_eval._candidate_token_ids(tokenizer, [prompt])
    pair_ids = [token_ids[prompt.answer_pair[0]], token_ids[prompt.answer_pair[1]]]
    expected_id = token_ids[prompt.expected]
    expected_idx = pair_ids.index(expected_id)
    other_idx = 1 - expected_idx
    pair_logits = logits[pair_ids]
    return float((pair_logits[expected_idx] - pair_logits[other_idx]).item())


def _filter_easy_dataset(
    candidates: list[binary_eval.BinaryPrompt],
    models: list[tuple[str, str]],
    target_count: int,
    min_margin: float,
    device: str,
    batch_size: int,
) -> tuple[list[binary_eval.BinaryPrompt], dict[str, Any]]:
    per_prompt_min_margin = [float("inf")] * len(candidates)
    per_model_stats: dict[str, Any] = {}
    for experiment, model_id in models:
        print(f"[easy-hard] filtering easy candidates with {model_id}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        logits = _teacher_logits_for_model(model_id, experiment, candidates, device, batch_size)
        margins = [
            _expected_margin(tokenizer, prompt, logits[idx])
            for idx, prompt in enumerate(candidates)
        ]
        for idx, margin in enumerate(margins):
            per_prompt_min_margin[idx] = min(per_prompt_min_margin[idx], margin)
        per_model_stats[model_id] = {
            "correct": sum(m > 0 for m in margins),
            "total": len(margins),
            "min_expected_margin": min(margins),
            "median_expected_margin": statistics.median(margins),
            "mean_expected_margin": statistics.fmean(margins),
        }

    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (per_prompt_min_margin[item[0]], item[1].prompt_id),
        reverse=True,
    )
    filtered = [
        prompt
        for idx, prompt in ranked
        if per_prompt_min_margin[idx] >= min_margin
    ]
    if len(filtered) < target_count:
        filtered = [
            prompt
            for idx, prompt in ranked
            if per_prompt_min_margin[idx] > 0
        ]
    selected = sorted(filtered[:target_count], key=lambda p: p.prompt_id)
    selected_ids = {prompt.prompt_id for prompt in selected}
    selected_margins = [
        per_prompt_min_margin[idx]
        for idx, prompt in enumerate(candidates)
        if prompt.prompt_id in selected_ids
    ]
    metadata = {
        "candidate_count": len(candidates),
        "target_count": target_count,
        "selected_count": len(selected),
        "requested_min_margin": min_margin,
        "selected_min_margin": min(selected_margins) if selected_margins else None,
        "selected_median_min_margin": statistics.median(selected_margins) if selected_margins else None,
        "selected_mean_min_margin": statistics.fmean(selected_margins) if selected_margins else None,
        "per_model_filter_stats": per_model_stats,
    }
    if len(selected) < target_count:
        raise RuntimeError(
            f"only {len(selected)} easy prompts passed all-teacher correctness; "
            f"target was {target_count}"
        )
    return selected, metadata


def _write_jsonl(path: Path, prompts: list[binary_eval.BinaryPrompt]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for prompt in prompts:
            f.write(json.dumps(asdict(prompt), ensure_ascii=True) + "\n")


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return binary_eval._summarize_rows(rows)


def _run_model_on_datasets(
    experiment: str,
    model_id: str,
    datasets: dict[str, list[binary_eval.BinaryPrompt]],
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    prompts: list[binary_eval.BinaryPrompt] = []
    dataset_for_prompt: dict[str, str] = {}
    for dataset_name, dataset_prompts in datasets.items():
        for prompt in dataset_prompts:
            prompts.append(prompt)
            dataset_for_prompt[prompt.prompt_id] = dataset_name

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    token_ids = binary_eval._candidate_token_ids(tokenizer, prompts)

    teacher: torch.nn.Module | None = None
    student: torch.nn.Module | None = None
    teacher_meta: dict[str, Any]
    student_meta: dict[str, Any]
    try:
        if experiment == "fp8_scaled_mm_vs_int32":
            teacher, teacher_meta = binary_eval._load_fp8_teacher(model_id, device)
            teacher_result = binary_eval._forward_prompt_logits(
                teacher,
                tokenizer,
                prompts,
                device,
                batch_size=batch_size,
                count_int32=False,
                count_scaled_mm=True,
            )
        else:
            teacher, teacher_meta = binary_eval._load_decompressed_teacher(model_id, device)
            teacher_result = binary_eval._forward_prompt_logits(
                teacher,
                tokenizer,
                prompts,
                device,
                batch_size=batch_size,
                count_int32=False,
                count_scaled_mm=False,
            )
    finally:
        _clear_model(teacher)

    try:
        student, student_meta = binary_eval._load_direct_int32_student(
            model_id,
            device,
            compressed_checkpoint=experiment == "fp4_decompressed_checkpoint_vs_int32",
        )
        student_result = binary_eval._forward_prompt_logits(
            student,
            tokenizer,
            prompts,
            device,
            batch_size=batch_size,
            count_int32=True,
            count_scaled_mm=False,
        )
    finally:
        _clear_model(student)

    rows = binary_eval._score_prompts(
        prompts,
        token_ids,
        teacher_result.logits,
        student_result.logits,
    )
    for row in rows:
        row["dataset"] = dataset_for_prompt[row["prompt_id"]]
    by_dataset: dict[str, list[dict[str, Any]]] = {
        name: [row for row in rows if row["dataset"] == name]
        for name in datasets
    }
    return {
        "experiment": experiment,
        "model_id": model_id,
        "teacher": teacher_meta
        | {
            "forward_seconds": teacher_result.seconds,
            "reference_scaled_mm_calls": teacher_result.reference_scaled_mm_calls,
        },
        "student": student_meta
        | {
            "forward_seconds": student_result.seconds,
            "int32_kernel_calls": student_result.int32_kernel_calls,
        },
        "summary_by_dataset": {
            name: _summarize_rows(dataset_rows)
            for name, dataset_rows in by_dataset.items()
        },
        "rows": rows,
    }


def _color(index: int) -> str:
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    return colors[index % len(colors)]


def _nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10**exponent
    for step in [1, 2, 5, 10]:
        if value <= step * base:
            return step * base
    return 10 * base


def _write_confidence_plot(result: dict[str, Any], path: Path) -> None:
    points_by_dataset: dict[str, list[dict[str, Any]]] = {"easy": [], "hard": []}
    for run_index, run in enumerate(result["runs"]):
        if "rows" not in run:
            continue
        for row in run["rows"]:
            points_by_dataset[row["dataset"]].append(
                {
                    "x": float(row["teacher_binary_margin"]),
                    "y": float(row["abs_top1_logit_error"]),
                    "run_index": run_index,
                    "model": run["model_id"],
                    "agreement": bool(row["binary_top1_agreement"]),
                }
            )

    width, height = 1280, 560
    margin_left, margin_right = 72, 28
    margin_top, margin_bottom = 42, 74
    panel_gap = 70
    panel_width = (width - margin_left - margin_right - panel_gap) / 2
    panel_height = height - margin_top - margin_bottom
    max_x = _nice_max(
        max((p["x"] for points in points_by_dataset.values() for p in points), default=1.0)
    )
    max_y = _nice_max(
        max((p["y"] for points in points_by_dataset.values() for p in points), default=1.0)
    )

    def sx(panel: int, x: float) -> float:
        x0 = margin_left + panel * (panel_width + panel_gap)
        return x0 + min(max(x / max_x, 0.0), 1.0) * panel_width

    def sy(y: float) -> float:
        return margin_top + panel_height - min(max(y / max_y, 0.0), 1.0) * panel_height

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:13px;fill:#111827}.title{font-size:18px;font-weight:700}.axis{stroke:#111827;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.legend{font-size:12px}</style>',
        f'<text class="title" x="{width / 2}" y="24" text-anchor="middle">FP Teacher Confidence vs Direct-Int32 Top Binary Logit Error</text>',
    ]

    for panel, dataset_name in enumerate(["easy", "hard"]):
        x0 = margin_left + panel * (panel_width + panel_gap)
        x1 = x0 + panel_width
        y0 = margin_top
        y1 = margin_top + panel_height
        svg.append(f'<text class="title" x="{(x0 + x1) / 2}" y="{y0 + 18}" text-anchor="middle">{dataset_name.title()} Dataset</text>')
        for tick in range(6):
            x = x0 + panel_width * tick / 5
            y = y1 - panel_height * tick / 5
            x_label = max_x * tick / 5
            y_label = max_y * tick / 5
            svg.append(f'<line class="grid" x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y1}"/>')
            svg.append(f'<line class="grid" x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}"/>')
            svg.append(f'<text x="{x:.2f}" y="{y1 + 20}" text-anchor="middle">{x_label:.1f}</text>')
            svg.append(f'<text x="{x0 - 10}" y="{y + 4:.2f}" text-anchor="end">{y_label:.1f}</text>')
        svg.append(f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>')
        svg.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}"/>')
        svg.append(f'<text x="{(x0 + x1) / 2}" y="{height - 24}" text-anchor="middle">FP binary margin</text>')
        if panel == 0:
            svg.append(
                f'<text transform="translate(18,{(y0 + y1) / 2}) rotate(-90)" text-anchor="middle">abs logit error on FP top binary token</text>'
            )

        for point in points_by_dataset[dataset_name]:
            color = _color(point["run_index"])
            radius = 3.1 if point["agreement"] else 5.0
            stroke = "#111827" if not point["agreement"] else "none"
            svg.append(
                f'<circle cx="{sx(panel, point["x"]):.2f}" cy="{sy(point["y"]):.2f}" '
                f'r="{radius}" fill="{color}" fill-opacity="0.45" stroke="{stroke}" stroke-width="1"/>'
            )

    legend_x = margin_left
    legend_y = height - 52
    for run_index, run in enumerate(result["runs"]):
        x = legend_x + (run_index % 3) * 390
        y = legend_y + (run_index // 3) * 18
        svg.append(f'<circle cx="{x}" cy="{y - 4}" r="4" fill="{_color(run_index)}" fill-opacity="0.65"/>')
        label = run["model_id"].replace("RedHatAI/", "").replace("JongYeop/", "").replace("pebeto/", "")
        svg.append(f'<text class="legend" x="{x + 10}" y="{y}">{label}</text>')

    svg.append("</svg>")
    path.write_text("\n".join(svg))


def _write_report(result: dict[str, Any], path: Path, plot_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Easy vs Hard Binary Dataset Report")
    lines.append("")
    lines.append(f"- Device: {result['device']}")
    lines.append(f"- Easy prompts: {result['datasets']['easy']['count']}")
    lines.append(f"- Hard prompts: {result['datasets']['hard']['count']}")
    lines.append("- Evaluation batch size: 1")
    lines.append("- Confidence: FP teacher binary margin between its chosen candidate and the other candidate.")
    lines.append("- Error: absolute student-vs-FP logit error on the FP teacher's top binary answer token.")
    lines.append(f"- Plot: [{plot_path.name}]({plot_path.name})")
    lines.append("")
    easy_filter = result["datasets"]["easy"]["filter_metadata"]
    lines.append("## Easy Dataset Filter")
    lines.append("")
    lines.append(
        f"Selected {easy_filter['selected_count']} easy prompts from {easy_filter['candidate_count']} candidates. "
        f"Minimum margin across all FP teachers in the selected set: {easy_filter['selected_min_margin']:.4g}; "
        f"median of per-prompt worst-model margins: {easy_filter['selected_median_min_margin']:.4g}."
    )
    lines.append("")
    lines.append("| FP teacher | candidate accuracy | mean expected margin | median expected margin | min expected margin |")
    lines.append("|---|---:|---:|---:|---:|")
    for model_id, stats in easy_filter["per_model_filter_stats"].items():
        acc = stats["correct"] / stats["total"]
        lines.append(
            f"| `{model_id}` | {acc:.3f} | {stats['mean_expected_margin']:.4g} | "
            f"{stats['median_expected_margin']:.4g} | {stats['min_expected_margin']:.4g} |"
        )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| dataset | experiment | model | teacher acc | student acc | binary agree | mean abs err | p90 abs err | max abs err | mean FP margin |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset_name in ["easy", "hard"]:
        for run in result["runs"]:
            summary = run["summary_by_dataset"][dataset_name]
            lines.append(
                f"| {dataset_name} | {run['experiment']} | `{run['model_id']}` | "
                f"{summary['teacher_expected_accuracy']:.3f} | {summary['student_expected_accuracy']:.3f} | "
                f"{summary['binary_top1_agreement']:.3f} | {summary['abs_top1_logit_error_mean']:.4g} | "
                f"{summary['abs_top1_logit_error_p90']:.4g} | {summary['abs_top1_logit_error_max']:.4g} | "
                f"{summary['teacher_binary_margin_mean']:.4g} |"
            )
    lines.append("")
    lines.append("## Takeaways")
    lines.append("")
    lines.append("- The easy set is empirically all-teacher-correct by construction, so any student mistakes there are direct approximation or batch/kernel effects rather than ambiguous labels.")
    lines.append("- The hard set is intentionally not filtered for all-teacher correctness; teacher accuracy and margins should be read as part of the result.")
    lines.append("- The SVG plot shows the expected relationship directly: low FP margin points are easiest to flip, while high-error FP4 points can remain visible even at high FP confidence.")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--easy-count", type=int, default=256)
    parser.add_argument("--hard-count", type=int, default=256)
    parser.add_argument("--easy-min-margin", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/easy_hard_binary_eval.json"))
    parser.add_argument("--report", type=Path, default=Path("results/easy_hard_binary_report.md"))
    parser.add_argument("--plot", type=Path, default=Path("results/easy_hard_confidence_vs_error.svg"))
    parser.add_argument("--easy-dataset", type=Path, default=Path("results/easy_binary_dataset.jsonl"))
    parser.add_argument("--hard-dataset", type=Path, default=Path("results/hard_binary_dataset.jsonl"))
    args = parser.parse_args()

    if any(model in " ".join(FP4_MODELS) for model in FP4_MODELS):
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

    device = binary_eval._require_device()
    torch.manual_seed(0)
    models = [
        *[("fp8_scaled_mm_vs_int32", model_id) for model_id in FP8_MODELS],
        *[("fp4_decompressed_checkpoint_vs_int32", model_id) for model_id in FP4_MODELS],
    ]

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    easy_candidates = _generate_easy_candidates()
    easy_prompts, easy_metadata = _filter_easy_dataset(
        easy_candidates,
        models,
        args.easy_count,
        args.easy_min_margin,
        device,
        args.batch_size,
    )
    hard_prompts = _generate_hard_prompts(args.hard_count)
    _write_jsonl(args.easy_dataset, easy_prompts)
    _write_jsonl(args.hard_dataset, hard_prompts)

    datasets = {"easy": easy_prompts, "hard": hard_prompts}
    result: dict[str, Any] = {
        "started_utc": started,
        "finished_utc": None,
        "device": torch.cuda.get_device_name(0),
        "datasets": {
            "easy": {
                "count": len(easy_prompts),
                "path": str(args.easy_dataset),
                "filter_metadata": easy_metadata,
            },
            "hard": {
                "count": len(hard_prompts),
                "path": str(args.hard_dataset),
            },
        },
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for experiment, model_id in models:
        print(f"[easy-hard] evaluating {model_id}", flush=True)
        run = _run_model_on_datasets(
            experiment,
            model_id,
            datasets,
            device,
            args.batch_size,
        )
        result["runs"].append(run)
        args.output.write_text(json.dumps(result, indent=2))

    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.output.write_text(json.dumps(result, indent=2))
    _write_confidence_plot(result, args.plot)
    _write_report(result, args.report, args.plot)
    print(f"[easy-hard] wrote {args.output}")
    print(f"[easy-hard] wrote {args.report}")
    print(f"[easy-hard] wrote {args.plot}")


if __name__ == "__main__":
    main()
