#!/usr/bin/env python3
"""Benchmark Qwen3-TTS: Python vs C (CPU) vs Metal (GPU).

Compares end-to-end generation latency across all three backends.
Produces a summary table and JSON report.

Usage:
    python3 scripts/benchmark_all.py --model-dir tmp/model
    python3 scripts/benchmark_all.py --model-dir tmp/model --runs 3 --skip-python
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    elapsed_ms: float
    audio_sec: float
    codec_tokens: int | None = None
    stop_reason: str | None = None
    internal_total_ms: float | None = None

    @property
    def rtf(self) -> float:
        if self.elapsed_ms <= 0:
            return 0.0
        return self.audio_sec / (self.elapsed_ms / 1000.0)

    @property
    def ms_per_token(self) -> float | None:
        if self.codec_tokens and self.codec_tokens > 0:
            return self.elapsed_ms / float(self.codec_tokens)
        return None

    @property
    def tokens_per_s(self) -> float | None:
        if self.codec_tokens and self.codec_tokens > 0 and self.elapsed_ms > 0:
            return float(self.codec_tokens) / (self.elapsed_ms / 1000.0)
        return None


def detect_python_runtime() -> dict[str, Any]:
    out: dict[str, Any] = {
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "implementation": platform.python_implementation(),
    }

    env_keys = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "PYTORCH_ENABLE_MPS_FALLBACK",
    ]
    out["env"] = {k: v for k, v in ((k, os.getenv(k)) for k in env_keys) if v is not None}

    versions: dict[str, str] = {}
    try:
        from importlib import metadata as importlib_metadata
    except Exception:
        importlib_metadata = None

    if importlib_metadata is not None:
        for pkg in ("torch", "transformers", "accelerate", "numpy", "qwen-tts"):
            try:
                versions[pkg] = importlib_metadata.version(pkg)
            except Exception:
                pass
    out["package_versions"] = versions

    return out


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return float(wf.getnframes()) / float(wf.getframerate())


def parse_c_stderr(stderr: str) -> tuple[float | None, int | None, str | None]:
    """Parse internal timing, codec tokens, and stop reason from C binary stderr."""
    internal_ms = None
    m = re.search(r"Total:\s+([0-9.]+)\s*ms", stderr)
    if m:
        internal_ms = float(m.group(1))

    codec_tokens = None
    m = re.search(r"Generated\s+(\d+)\s+codec tokens", stderr)
    if m:
        codec_tokens = int(m.group(1))

    stop = None
    if re.search(r"EOS at step", stderr):
        stop = "eos"
    elif codec_tokens is not None:
        stop = "eos"

    return internal_ms, codec_tokens, stop


_PERSISTENT_RE = re.compile(
    r"\[persistent\]\s+run\s+(\d+)/(\d+):\s+elapsed=([0-9.]+)\s+ms,\s+audio=([0-9.]+)s,\s+"
    r"talker=([0-9.]+)\s+ms,\s+codec=([0-9.]+)\s+ms,\s+total=([0-9.]+)\s+ms,\s+tokens=(\d+)"
)


def parse_persistent_runs(stderr: str) -> list[RunResult]:
    stop = None
    if "Stop: eos" in stderr or "EOS at step" in stderr:
        stop = "eos"
    elif "Stop: max_tokens" in stderr:
        stop = "max_tokens"

    out: list[RunResult] = []
    for m in _PERSISTENT_RE.finditer(stderr):
        out.append(
            RunResult(
                elapsed_ms=float(m.group(3)),
                audio_sec=float(m.group(4)),
                codec_tokens=int(m.group(8)),
                stop_reason=stop,
                internal_total_ms=float(m.group(7)),
            )
        )
    return out


def build_c_cmd(
    binary: Path,
    model_dir: str,
    token_ids: str,
    speaker: str,
    language: str,
    args: argparse.Namespace,
    out_wav: Path,
    bench_runs: int | None = None,
    bench_warmup: int | None = None,
    verbose: bool = True,
) -> list[str]:
    cmd = [
        str(binary), "-d", model_dir, "-t", token_ids,
        "-o", str(out_wav), "-s", speaker, "-l", language,
        "--temperature", str(args.temperature),
        "--top-k", str(args.top_k),
        "--top-p", str(args.top_p),
        "--repetition-penalty", str(args.repetition_penalty),
        "--subtalker-temperature", str(args.subtalker_temperature),
        "--subtalker-top-k", str(args.subtalker_top_k),
        "--subtalker-top-p", str(args.subtalker_top_p),
        "--max-tokens", str(args.max_new_tokens),
    ]
    if verbose:
        cmd.append("-v")
    if bench_runs is not None:
        cmd += ["--benchmark-runs", str(bench_runs)]
    if bench_warmup is not None:
        cmd += ["--benchmark-warmup", str(bench_warmup)]
    return cmd


def run_c_binary(
    binary: Path,
    model_dir: str,
    token_ids: str,
    speaker: str,
    language: str,
    args: argparse.Namespace,
    out_wav: Path,
    env: dict[str, str] | None = None,
) -> RunResult:
    cmd = build_c_cmd(binary, model_dir, token_ids, speaker, language, args, out_wav, verbose=True)
    start = time.perf_counter()
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    audio_sec = wav_duration(out_wav)
    internal_ms, codec_tokens, stop = parse_c_stderr(proc.stderr)
    return RunResult(
        elapsed_ms=elapsed_ms,
        audio_sec=audio_sec,
        codec_tokens=codec_tokens,
        stop_reason=stop,
        internal_total_ms=internal_ms,
    )


def bench_binary(
    label: str,
    binary: Path,
    model_dir: str,
    token_ids: str,
    speaker: str,
    language: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> list[RunResult]:
    env = os.environ.copy()
    if label == "metal":
        env.setdefault("QWEN_TTS_ENABLE_METAL", "1")
        env.setdefault("QWEN_TTS_METAL_TALKER", "1")
        env.setdefault("QWEN_TTS_METAL_SUBTALKER", "1")

    if args.persistent:
        wav_path = out_dir / f"{label}_output.wav"
        cmd = build_c_cmd(
            binary, model_dir, token_ids, speaker, language, args, wav_path,
            bench_runs=args.runs, bench_warmup=args.warmup, verbose=False
        )
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)
        results = parse_persistent_runs(proc.stderr)
        if not results:
            # Fallback to single-result parsing if binary does not emit persistent lines.
            audio_sec = wav_duration(wav_path)
            internal_ms, codec_tokens, stop = parse_c_stderr(proc.stderr)
            elapsed_ms = internal_ms if internal_ms is not None else audio_sec * 1000.0
            results = [RunResult(
                elapsed_ms=elapsed_ms,
                audio_sec=audio_sec,
                codec_tokens=codec_tokens,
                stop_reason=stop,
                internal_total_ms=internal_ms,
            )]
        for i, rr in enumerate(results):
            tok_info = f", {rr.codec_tokens} tokens" if rr.codec_tokens else ""
            ms_tok = f", {rr.ms_per_token:.1f} ms/tok" if rr.ms_per_token else ""
            print(
                f"  [{label}] run {i+1}/{len(results)}: "
                f"{rr.elapsed_ms:.0f} ms, {rr.audio_sec:.2f}s audio, "
                f"RTF={rr.rtf:.2f}x{tok_info}{ms_tok}"
            )
        return results

    results: list[RunResult] = []

    # Warmup
    for i in range(args.warmup):
        fd, tmp = tempfile.mkstemp(suffix=".wav", dir=str(out_dir))
        os.close(fd)
        tmp_path = Path(tmp)
        try:
            run_c_binary(binary, model_dir, token_ids, speaker, language, args, tmp_path, env=env)
        finally:
            tmp_path.unlink(missing_ok=True)
        print(f"  [{label}] warmup {i+1}/{args.warmup}")

    # Timed runs
    for i in range(args.runs):
        wav_path = out_dir / f"{label}_output.wav" if i == args.runs - 1 else None
        if wav_path is None:
            fd, tmp = tempfile.mkstemp(suffix=".wav", dir=str(out_dir))
            os.close(fd)
            wav_path = Path(tmp)
            cleanup = True
        else:
            cleanup = False

        rr = run_c_binary(binary, model_dir, token_ids, speaker, language, args, wav_path, env=env)
        results.append(rr)

        tok_info = f", {rr.codec_tokens} tokens" if rr.codec_tokens else ""
        ms_tok = f", {rr.ms_per_token:.1f} ms/tok" if rr.ms_per_token else ""
        print(
            f"  [{label}] run {i+1}/{args.runs}: "
            f"{rr.elapsed_ms:.0f} ms, {rr.audio_sec:.2f}s audio, "
            f"RTF={rr.rtf:.2f}x{tok_info}{ms_tok}"
        )

        if cleanup:
            wav_path.unlink(missing_ok=True)

    return results


def bench_python(
    model_dir: str,
    token_ids: str,
    speaker: str,
    language: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[list[RunResult], float]:
    import numpy as np
    import torch

    from qwen_tts import Qwen3TTSModel

    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        model_dir, device_map="cpu", dtype=torch.float32
    )
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  [python] model loaded in {load_ms:.0f} ms")

    # Use the model's own tokenization helper. Passing a plain torch tensor to
    # model.generate() can break on newer qwen_tts versions that expect
    # list-style tokenized inputs.
    input_ids = model._tokenize_texts([model._build_assistant_text(args.text)])

    do_sample = args.temperature > 0.0
    subtalker_do_sample = args.subtalker_temperature > 0.0
    gen_kwargs = model._merge_generate_kwargs(
        do_sample=do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature if args.temperature > 0.0 else 1.0,
        repetition_penalty=args.repetition_penalty,
        subtalker_dosample=subtalker_do_sample,
        subtalker_top_k=args.subtalker_top_k,
        subtalker_top_p=args.subtalker_top_p,
        subtalker_temperature=args.subtalker_temperature if args.subtalker_temperature > 0.0 else 1.0,
        max_new_tokens=args.max_new_tokens,
    )

    def run_once() -> RunResult:
        start = time.perf_counter()
        talker_codes_list, _ = model.model.generate(
            input_ids=input_ids,
            instruct_ids=[None],
            languages=[language],
            speakers=[speaker],
            non_streaming_mode=True,
            **gen_kwargs,
        )
        wavs, sr = model.model.speech_tokenizer.decode(
            [{"audio_codes": c} for c in talker_codes_list]
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        audio = np.asarray(wavs[0])
        audio_sec = float(len(audio)) / float(sr)
        codes = talker_codes_list[0]
        codec_tokens = int(codes.shape[0])
        stop = "max_tokens" if codec_tokens >= args.max_new_tokens else "eos"
        return RunResult(
            elapsed_ms=elapsed_ms,
            audio_sec=audio_sec,
            codec_tokens=codec_tokens,
            stop_reason=stop,
        )

    results: list[RunResult] = []

    for i in range(args.warmup):
        run_once()
        print(f"  [python] warmup {i+1}/{args.warmup}")

    for i in range(args.runs):
        rr = run_once()
        results.append(rr)
        tok_info = f", {rr.codec_tokens} tokens" if rr.codec_tokens else ""
        ms_tok = f", {rr.ms_per_token:.1f} ms/tok" if rr.ms_per_token else ""
        print(
            f"  [python] run {i+1}/{args.runs}: "
            f"{rr.elapsed_ms:.0f} ms, {rr.audio_sec:.2f}s audio, "
            f"RTF={rr.rtf:.2f}x{tok_info}{ms_tok}"
        )

    return results, load_ms


def summarize(results: list[RunResult]) -> dict[str, Any]:
    elapsed = [r.elapsed_ms for r in results]
    audio = [r.audio_sec for r in results]
    rtf = [r.rtf for r in results]
    ms_tok = [r.ms_per_token for r in results if r.ms_per_token is not None]
    tok_s = [r.tokens_per_s for r in results if r.tokens_per_s is not None]

    s: dict[str, Any] = {
        "runs": len(results),
        "elapsed_ms": {"mean": statistics.fmean(elapsed), "median": statistics.median(elapsed),
                       "min": min(elapsed), "max": max(elapsed)},
        "audio_sec": {"mean": statistics.fmean(audio), "median": statistics.median(audio)},
        "rtf": {"mean": statistics.fmean(rtf), "median": statistics.median(rtf)},
    }
    if ms_tok:
        s["ms_per_token"] = {"mean": statistics.fmean(ms_tok), "median": statistics.median(ms_tok),
                              "min": min(ms_tok), "max": max(ms_tok)}
    if tok_s:
        s["tokens_per_s"] = {"mean": statistics.fmean(tok_s), "median": statistics.median(tok_s)}
    return s


def print_table(summaries: dict[str, dict[str, Any]]) -> None:
    """Print a comparison table."""
    labels = list(summaries.keys())
    print("\n" + "=" * 78)
    print(f"{'Metric':<28}", end="")
    for l in labels:
        print(f"{l:>16}", end="")
    print()
    print("-" * 78)

    def row(name: str, vals: list[str]) -> None:
        print(f"{name:<28}", end="")
        for v in vals:
            print(f"{v:>16}", end="")
        print()

    row("Median elapsed (ms)", [
        f"{s['elapsed_ms']['median']:.0f}" for s in summaries.values()])
    row("Mean elapsed (ms)", [
        f"{s['elapsed_ms']['mean']:.0f}" for s in summaries.values()])
    row("Median audio (sec)", [
        f"{s['audio_sec']['median']:.2f}" for s in summaries.values()])
    row("Median RTF (x realtime)", [
        f"{s['rtf']['median']:.2f}x" for s in summaries.values()])

    if all("ms_per_token" in s for s in summaries.values()):
        row("Median ms/token", [
            f"{s['ms_per_token']['median']:.1f}" for s in summaries.values()])
        row("Mean ms/token", [
            f"{s['ms_per_token']['mean']:.1f}" for s in summaries.values()])

    if all("tokens_per_s" in s for s in summaries.values()):
        row("Median tokens/sec", [
            f"{s['tokens_per_s']['median']:.1f}" for s in summaries.values()])

    # Speedups relative to first entry
    base_label = labels[0]
    base_median = summaries[base_label]["elapsed_ms"]["median"]
    print("-" * 78)
    row(f"Speedup vs {base_label}", [
        f"{base_median / s['elapsed_ms']['median']:.2f}x"
        for s in summaries.values()])

    if all("ms_per_token" in s for s in summaries.values()):
        base_ms_tok = summaries[base_label]["ms_per_token"]["median"]
        row(f"ms/tok ratio vs {base_label}", [
            f"{s['ms_per_token']['median'] / base_ms_tok:.2f}x"
            for s in summaries.values()])

    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Python vs C (CPU) vs Metal (GPU)")
    parser.add_argument("--model-dir", required=True, help="Model directory")
    parser.add_argument("--text", default="Hello from Qwen3-TTS benchmark. porting done by Muonium AI Studios")
    parser.add_argument("--language", default="English")
    parser.add_argument("--speaker", default="aiden")

    parser.add_argument("--c-bin", default="./qwen-tts", help="C CPU binary")
    parser.add_argument("--metal-bin", default="./metal/qwen-tts-metal", help="Metal GPU binary")

    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)

    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--subtalker-temperature", type=float, default=0.9)
    parser.add_argument("--subtalker-top-k", type=int, default=50)
    parser.add_argument("--subtalker-top-p", type=float, default=1.0)

    parser.add_argument("--skip-python", action="store_true", help="Skip Python benchmark")
    parser.add_argument("--skip-c", action="store_true", help="Skip C CPU benchmark")
    parser.add_argument("--skip-metal", action="store_true", help="Skip Metal GPU benchmark")
    parser.add_argument("--output-dir", default="benchmark_output")
    parser.add_argument("--persistent", action="store_true",
                        help="Use process-persistent C/Metal benchmark mode")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve paths relative to repo root (parent of scripts/)
    repo_root = Path(__file__).resolve().parent.parent
    model_dir = str((repo_root / args.model_dir).resolve()) if not Path(args.model_dir).is_absolute() else args.model_dir
    args.c_bin = str((repo_root / args.c_bin).resolve()) if not Path(args.c_bin).is_absolute() else args.c_bin
    args.metal_bin = str((repo_root / args.metal_bin).resolve()) if not Path(args.metal_bin).is_absolute() else args.metal_bin

    # Get token IDs using transformers tokenizer
    print("[setup] Tokenizing input text...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
        chat_text = f"<|im_start|>assistant\n{args.text}<|im_end|>\n<|im_start|>assistant\n"
        token_ids_list = tokenizer(chat_text, return_tensors="pt").input_ids[0].tolist()
        token_ids_str = ",".join(str(x) for x in token_ids_list)
        print(f"[setup] {len(token_ids_list)} prompt tokens")
    except ImportError:
        # Fallback: use pre-tokenized test file
        token_file = Path("test/tokens_great_power.txt")
        if token_file.exists():
            token_ids_str = token_file.read_text().strip()
            print(f"[setup] Using pre-tokenized file: {token_file}")
        else:
            print("Error: transformers not installed and no fallback token file found")
            return 1

    print(f"[setup] text: {args.text!r}")
    print(f"[setup] speaker: {args.speaker}, language: {args.language}")
    print(f"[setup] runs: {args.runs}, warmup: {args.warmup}")
    print(f"[setup] max_new_tokens: {args.max_new_tokens}")
    print()

    summaries: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        "meta": {
            "hostname": os.uname().nodename,
            "platform": sys.platform,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "python_runtime": detect_python_runtime(),
        },
        "config": {
            "text": args.text,
            "language": args.language,
            "speaker": args.speaker,
            "runs": args.runs,
            "warmup": args.warmup,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "persistent": args.persistent,
        },
    }

    # ---- Python benchmark ----
    if not args.skip_python:
        try:
            print("=== Python (PyTorch CPU) ===")
            py_results, py_load_ms = bench_python(
                model_dir, token_ids_str, args.speaker, args.language, args, out_dir
            )
            py_summary = summarize(py_results)
            py_summary["load_ms"] = py_load_ms
            summaries["Python"] = py_summary
            report["python"] = py_summary
        except Exception as e:
            print(f"  [python] FAILED: {e}")
    else:
        print("=== Python: SKIPPED ===")

    # ---- C CPU benchmark ----
    if not args.skip_c:
        c_bin = Path(args.c_bin)
        if c_bin.exists():
            print(f"\n=== C (CPU) - {c_bin} ===")
            c_results = bench_binary(
                "c_cpu", c_bin, model_dir, token_ids_str,
                args.speaker, args.language, args, out_dir
            )
            c_summary = summarize(c_results)
            summaries["C (CPU)"] = c_summary
            report["c_cpu"] = c_summary
        else:
            print(f"\n=== C (CPU): binary not found: {c_bin} ===")
    else:
        print("\n=== C (CPU): SKIPPED ===")

    # ---- Metal GPU benchmark ----
    if not args.skip_metal:
        metal_bin = Path(args.metal_bin)
        if metal_bin.exists():
            print(f"\n=== Metal (GPU) - {metal_bin} ===")
            metal_results = bench_binary(
                "metal", metal_bin, model_dir, token_ids_str,
                args.speaker, args.language, args, out_dir
            )
            metal_summary = summarize(metal_results)
            summaries["Metal (GPU)"] = metal_summary
            report["metal"] = metal_summary
        else:
            print(f"\n=== Metal (GPU): binary not found: {metal_bin} ===")
    else:
        print("\n=== Metal (GPU): SKIPPED ===")

    if len(summaries) < 2:
        print("\nNeed at least 2 backends to compare. Exiting.")
        return 1

    # ---- Comparison table ----
    print_table(summaries)

    # ---- Save JSON report ----
    report["summaries"] = summaries
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report["meta"]["timestamp"] = timestamp
    json_text = json.dumps(report, indent=2, default=str)
    report_latest = out_dir / "benchmark_all_results.json"
    report_timestamped = out_dir / f"benchmark_all_results_{timestamp}.json"
    report_latest.write_text(json_text, encoding="utf-8")
    report_timestamped.write_text(json_text, encoding="utf-8")
    print(f"\nJSON report (latest): {report_latest}")
    print(f"JSON report (timestamped): {report_timestamped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
