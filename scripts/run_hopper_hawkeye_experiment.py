"""Run the Hopper Hawkeye experiment sequence from the next-steps report."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def _run_to_file(command: list[str], cwd: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[hopper] running in {cwd}: {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output_path.write_text(completed.stdout)
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise SystemExit(
            f"command failed with exit code {completed.returncode}; "
            f"wrote {output_path}\n{tail}"
        )
    print(f"[hopper] wrote {output_path}")


def _ensure_hawkeye_repo(hawkeye_dir: Path, output_dir: Path) -> None:
    if (hawkeye_dir / ".git").exists():
        return
    hawkeye_dir.parent.mkdir(parents=True, exist_ok=True)
    _run_to_file(
        ["git", "clone", "https://github.com/badasherez/gpu-simulator", str(hawkeye_dir)],
        Path.cwd(),
        output_dir / "git_clone.log",
    )


def _verify_public_simulator(args: argparse.Namespace) -> None:
    hawkeye_dir = Path(args.hawkeye_dir)
    output_dir = Path(args.output_dir)
    _ensure_hawkeye_repo(hawkeye_dir, output_dir)
    if not args.skip_cupy_install and importlib.util.find_spec("cupy") is None:
        _run_to_file(
            [sys.executable, "-m", "pip", "install", "cupy-cuda12x>=13.0.0,<14.0.0"],
            Path.cwd(),
            output_dir / "cupy_install.log",
        )
    _run_to_file(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        hawkeye_dir,
        output_dir / "hawkeye_build.log",
    )
    _run_to_file(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        hawkeye_dir / "experiments" / "wgmma_e4m3",
        output_dir / "hawkeye_wgmma_build.log",
    )
    _run_to_file(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-k",
            args.public_pytest_k,
            "--hardware",
            args.hardware,
            "-v",
            "-s",
        ],
        hawkeye_dir,
        output_dir / "hawkeye_fp8_pytest.log",
    )


def _run_repo_probe(script: str, args: list[str], output_path: Path) -> None:
    _run_to_file([sys.executable, script, *args], Path.cwd(), output_path)


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if not args.skip_public_simulator:
        _verify_public_simulator(args)

    common_shape = [
        "--m",
        str(args.m),
        "--n",
        str(args.n),
        "--k",
        str(args.k),
        "--seed",
        str(args.seed),
    ]
    if not args.skip_l40s_probes:
        _run_repo_probe(
            "scripts/probe_hawkeye_l40s_fp8.py",
            [
                *common_shape,
                "--groups",
                args.groups,
                "--widths",
                args.widths,
            ],
            output_dir / "hopper_l40s_probe_rerun.json",
        )
        _run_repo_probe(
            "scripts/probe_hawkeye_bucket_products.py",
            [
                *common_shape,
                "--group",
                str(args.group),
                "--width",
                str(args.width),
            ],
            output_dir / "hopper_bucket_probe_rerun.json",
        )

    if not args.skip_qgmma_random:
        _run_repo_probe(
            "scripts/probe_hopper_qgmma_hawkeye.py",
            [
                *common_shape,
                "--group",
                str(args.group),
                "--width",
                str(args.width),
            ],
            output_dir / "hopper_qgmma_random.json",
        )

    if not args.skip_qgmma_real_weight:
        _run_repo_probe(
            "scripts/probe_hopper_qgmma_hawkeye.py",
            [
                "--m",
                str(args.m),
                "--seed",
                str(args.seed),
                "--group",
                str(args.group),
                "--width",
                str(args.width),
                "--real-weight",
                "--model-id",
                args.model_id,
            ],
            output_dir / "hopper_qgmma_real_weight.json",
        )

    if not args.skip_freivalds_student:
        _run_repo_probe(
            "scripts/probe_hopper_hawkeye_freivalds_student.py",
            [
                *common_shape,
                "--group",
                str(args.group),
                "--width",
                str(args.width),
                "--class-chunk",
                str(args.class_chunk),
            ],
            output_dir / "hopper_freivalds_student_random.json",
        )
        _run_repo_probe(
            "scripts/probe_hopper_hawkeye_freivalds_student.py",
            [
                "--m",
                str(args.m),
                "--seed",
                str(args.seed),
                "--group",
                str(args.group),
                "--width",
                str(args.width),
                "--class-chunk",
                str(args.class_chunk),
                "--real-weight",
                "--model-id",
                args.model_id,
            ],
            output_dir / "hopper_freivalds_student_real_weight.json",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/hopper_hawkeye")
    parser.add_argument("--hawkeye-dir", default="/tmp/gpu-simulator")
    parser.add_argument("--hardware", default="H100")
    parser.add_argument("--public-pytest-k", default="fp8")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--k", type=int, default=896)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--groups", default="32")
    parser.add_argument("--widths", default="14,15,16,17,18,24")
    parser.add_argument("--group", type=int, default=32)
    parser.add_argument("--width", type=int, default=14)
    parser.add_argument("--class-chunk", type=int, default=32)
    parser.add_argument("--model-id", default="RedHatAI/Qwen2.5-0.5B-FP8-dynamic")
    parser.add_argument("--skip-cupy-install", action="store_true")
    parser.add_argument("--skip-public-simulator", action="store_true")
    parser.add_argument("--skip-l40s-probes", action="store_true")
    parser.add_argument("--skip-qgmma-random", action="store_true")
    parser.add_argument("--skip-qgmma-real-weight", action="store_true")
    parser.add_argument("--skip-freivalds-student", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
