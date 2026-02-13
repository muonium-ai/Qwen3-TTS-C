#!/usr/bin/env python3
"""Benchmark Qwen3-TTS Python vs C implementation.

This script compares end-to-end generation latency (single sample) between:
- Python API (qwen_tts.Qwen3TTSModel)
- C binary (qwen-tts)

It writes a JSON report for CI consumption.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

torch = None
AutoTokenizer = None
Qwen3TTSModel = None


@dataclass
class RunResult:
    elapsed_ms: float
    audio_sec: float
    rtf: float
    internal_total_ms: float | None = None
    codec_tokens: int | None = None
    stop_reason: str | None = None
    stop_step: int | None = None
    ms_per_token: float | None = None
    tokens_per_s: float | None = None
    ms_per_audio_sec: float | None = None


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def _finalize_run_metrics(rr: RunResult) -> RunResult:
    if rr.codec_tokens is not None and rr.codec_tokens > 0:
        rr.ms_per_token = rr.elapsed_ms / float(rr.codec_tokens)
        rr.tokens_per_s = float(rr.codec_tokens) / (rr.elapsed_ms / 1000.0) if rr.elapsed_ms > 0 else None

    if rr.audio_sec > 0:
        rr.ms_per_audio_sec = rr.elapsed_ms / rr.audio_sec

    return rr


def _write_wav_mono_16bit(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())


def _wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        sample_rate = wf.getframerate()
    return float(frames) / float(sample_rate)


def _sync_if_cuda(device_map: str) -> None:
    if device_map.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def _chat_template(text: str) -> str:
    return f"<|im_start|>assistant\\n{text}<|im_end|>\\n<|im_start|>assistant\\n"


def _parse_c_internal_total_ms(stderr: str) -> float | None:
    match = re.search(r"^\s*Total:\s+([0-9]+(?:\.[0-9]+)?)\s*ms\s*$", stderr, re.MULTILINE)
    if match:
        return float(match.group(1))

    match = re.search(r"Total:\s+([0-9]+(?:\.[0-9]+)?)\s*ms\s*\(", stderr)
    if match:
        return float(match.group(1))

    return None


def _parse_c_generated_tokens(stderr: str) -> int | None:
    match = re.search(r"Generated\s+(\d+)\s+codec tokens", stderr)
    if match:
        return int(match.group(1))
    return None


def _parse_c_stop(stderr: str, max_new_tokens: int, generated_tokens: int | None) -> tuple[str | None, int | None]:
    match = re.search(r"Stop:\s+([a-z_]+)\s+at step\s+(\d+)", stderr)
    if match:
        return match.group(1), int(match.group(2))

    eos_match = re.search(r"EOS at step\s+(\d+)", stderr)
    if eos_match:
        step = int(eos_match.group(1))
        return "eos", step

    if generated_tokens is not None:
        if generated_tokens >= max_new_tokens:
            return "max_tokens", max_new_tokens
        return "eos", generated_tokens

    return None, None


def _summary_ms(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _summary_scalar(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _collect_entity_summary(results: list[RunResult], *, load_ms: float | None = None, speaker: str | None = None) -> dict[str, Any]:
    elapsed_values = [x.elapsed_ms for x in results]
    rtf_values = [x.rtf for x in results]
    audio_values = [x.audio_sec for x in results]

    codec_tokens_values = [float(x.codec_tokens) for x in results if x.codec_tokens is not None]
    stop_steps = [float(x.stop_step) for x in results if x.stop_step is not None]
    ms_per_token_values = [x.ms_per_token for x in results if x.ms_per_token is not None]
    tokens_per_s_values = [x.tokens_per_s for x in results if x.tokens_per_s is not None]
    ms_per_audio_sec_values = [x.ms_per_audio_sec for x in results if x.ms_per_audio_sec is not None]

    stop_reason_counts = Counter(x.stop_reason for x in results if x.stop_reason)

    summary: dict[str, Any] = {
        "runs": [asdict(x) for x in results],
        "timing": _summary_ms(elapsed_values),
        "rtf": {
            "mean": statistics.fmean(rtf_values),
            "median": statistics.median(rtf_values),
        },
        "audio_sec": {
            "mean": statistics.fmean(audio_values),
            "median": statistics.median(audio_values),
        },
    }

    if load_ms is not None:
        summary["load_ms"] = load_ms
    if speaker is not None:
        summary["speaker"] = speaker

    if codec_tokens_values:
        summary["codec_tokens"] = _summary_scalar(codec_tokens_values)
    if stop_steps:
        summary["stop_step"] = _summary_scalar(stop_steps)
    if stop_reason_counts:
        summary["stop_reason_counts"] = dict(stop_reason_counts)

    normalized: dict[str, Any] = {}
    if ms_per_token_values:
        normalized["ms_per_token"] = _summary_scalar(ms_per_token_values)
    if tokens_per_s_values:
        normalized["tokens_per_s"] = _summary_scalar(tokens_per_s_values)
    if ms_per_audio_sec_values:
        normalized["ms_per_audio_sec"] = _summary_scalar(ms_per_audio_sec_values)
    if normalized:
        summary["normalized"] = normalized

    return summary


def _bench_python(
    args: argparse.Namespace,
    out_dir: Path,
    max_new_tokens_effective: int,
) -> tuple[dict[str, Any], str, float]:
    model_kwargs: dict[str, Any] = {
        "device_map": args.python_device,
        "dtype": _resolve_dtype(args.python_dtype),
    }
    if args.attn_implementation and not args.python_device.startswith("cpu"):
        model_kwargs["attn_implementation"] = args.attn_implementation

    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(args.python_model, **model_kwargs)
    load_ms = (time.perf_counter() - t0) * 1000.0

    if getattr(model.model, "tts_model_type", "") != "custom_voice":
        raise RuntimeError(
            "Benchmark script currently supports CustomVoice models for Python benchmark. "
            f"Got tts_model_type={getattr(model.model, 'tts_model_type', None)}"
        )

    speaker = args.speaker
    if not speaker:
        supported = model.get_supported_speakers() or []
        if not supported:
            raise RuntimeError("No supported speakers returned by Python model; pass --speaker explicitly.")
        speaker = supported[0]

    input_ids = model._tokenize_texts([model._build_assistant_text(args.text)])
    gen_kwargs = model._merge_generate_kwargs(
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        subtalker_dosample=True,
        subtalker_top_k=args.subtalker_top_k,
        subtalker_top_p=args.subtalker_top_p,
        subtalker_temperature=args.subtalker_temperature,
        max_new_tokens=max_new_tokens_effective,
    )

    def run_once(save_path: Path | None) -> RunResult:
        _sync_if_cuda(args.python_device)
        start = time.perf_counter()

        talker_codes_list, _ = model.model.generate(
            input_ids=input_ids,
            instruct_ids=[None],
            languages=[args.language],
            speakers=[speaker],
            non_streaming_mode=True,
            **gen_kwargs,
        )
        wavs, sr = model.model.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])

        _sync_if_cuda(args.python_device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        audio = np.asarray(wavs[0])
        audio_sec = float(len(audio)) / float(sr)
        rtf = audio_sec / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0

        codes = talker_codes_list[0]
        codec_tokens = int(codes.shape[0])
        stop_reason = "max_tokens" if codec_tokens >= max_new_tokens_effective else "eos"
        stop_step = codec_tokens

        if save_path is not None:
            _write_wav_mono_16bit(save_path, audio, int(sr))

        return _finalize_run_metrics(
            RunResult(
                elapsed_ms=elapsed_ms,
                audio_sec=audio_sec,
                rtf=rtf,
                codec_tokens=codec_tokens,
                stop_reason=stop_reason,
                stop_step=stop_step,
            )
        )

    for i in range(args.warmup_runs):
        run_once(None)
        print(f"[python] warmup {i + 1}/{args.warmup_runs} done", flush=True)

    results: list[RunResult] = []
    for i in range(args.runs):
        save_path = out_dir / "python_output.wav" if i == args.runs - 1 else None
        rr = run_once(save_path)
        results.append(rr)
        tok_str = f", tokens={rr.codec_tokens}" if rr.codec_tokens is not None else ""
        stop_str = f", stop={rr.stop_reason}" if rr.stop_reason else ""
        print(
            f"[python] run {i + 1}/{args.runs}: {rr.elapsed_ms:.2f} ms, "
            f"audio={rr.audio_sec:.3f}s, rtf={rr.rtf:.3f}x{tok_str}{stop_str}",
            flush=True,
        )

    summary = _collect_entity_summary(results, load_ms=load_ms, speaker=speaker)
    return summary, speaker, statistics.median([x.elapsed_ms for x in results])


def _bench_c(
    args: argparse.Namespace,
    out_dir: Path,
    token_ids: str,
    speaker: str,
    max_new_tokens_effective: int,
) -> tuple[dict[str, Any], float]:
    c_bin = Path(args.c_bin).resolve()
    if not c_bin.exists():
        raise FileNotFoundError(f"C binary not found: {c_bin}")

    def run_once(save_path: Path | None) -> RunResult:
        tmp_wav_path: Path | None = None
        if save_path is not None:
            output_wav = save_path
        else:
            fd, tmp_name = tempfile.mkstemp(prefix="bench_c_", suffix=".wav", dir=str(out_dir))
            os.close(fd)
            tmp_wav_path = Path(tmp_name)
            output_wav = tmp_wav_path

        cmd = [
            str(c_bin),
            "-d",
            args.c_model_dir,
            "-t",
            token_ids,
            "-o",
            str(output_wav),
            "-v",
            "-s",
            speaker,
            "-l",
            args.language,
            "--temperature",
            str(args.temperature),
            "--top-k",
            str(args.top_k),
            "--top-p",
            str(args.top_p),
            "--repetition-penalty",
            str(args.repetition_penalty),
            "--subtalker-temperature",
            str(args.subtalker_temperature),
            "--subtalker-top-k",
            str(args.subtalker_top_k),
            "--subtalker-top-p",
            str(args.subtalker_top_p),
            "--max-tokens",
            str(max_new_tokens_effective),
        ]

        start = time.perf_counter()
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        audio_sec = _wav_duration_sec(output_wav)
        rtf = audio_sec / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0

        internal_ms = _parse_c_internal_total_ms(proc.stderr)
        codec_tokens = _parse_c_generated_tokens(proc.stderr)
        stop_reason, stop_step = _parse_c_stop(proc.stderr, max_new_tokens_effective, codec_tokens)

        if tmp_wav_path is not None:
            try:
                tmp_wav_path.unlink()
            except FileNotFoundError:
                pass

        return _finalize_run_metrics(
            RunResult(
                elapsed_ms=elapsed_ms,
                audio_sec=audio_sec,
                rtf=rtf,
                internal_total_ms=internal_ms,
                codec_tokens=codec_tokens,
                stop_reason=stop_reason,
                stop_step=stop_step,
            )
        )

    for i in range(args.warmup_runs):
        run_once(None)
        print(f"[c] warmup {i + 1}/{args.warmup_runs} done", flush=True)

    results: list[RunResult] = []
    for i in range(args.runs):
        save_path = out_dir / "c_output.wav" if i == args.runs - 1 else None
        rr = run_once(save_path)
        results.append(rr)
        internal_str = f", internal={rr.internal_total_ms:.2f} ms" if rr.internal_total_ms is not None else ""
        tok_str = f", tokens={rr.codec_tokens}" if rr.codec_tokens is not None else ""
        stop_str = f", stop={rr.stop_reason}" if rr.stop_reason else ""
        print(
            f"[c] run {i + 1}/{args.runs}: {rr.elapsed_ms:.2f} ms{internal_str}, "
            f"audio={rr.audio_sec:.3f}s, rtf={rr.rtf:.3f}x{tok_str}{stop_str}",
            flush=True,
        )

    summary = _collect_entity_summary(results)

    internal_values = [x.internal_total_ms for x in results if x.internal_total_ms is not None]
    if internal_values:
        summary["internal_timing"] = _summary_ms(internal_values)
        summary["internal_timing"]["count"] = len(internal_values)

    return summary, statistics.median([x.elapsed_ms for x in results])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-TTS Python vs C")

    parser.add_argument("--python-model", required=True, help="Model path/repo for Python benchmark")
    parser.add_argument("--c-model-dir", required=True, help="Local model directory for C binary")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path/repo id (default: --python-model)")
    parser.add_argument("--c-bin", default="./qwen-tts", help="Path to C binary")

    parser.add_argument("--text", default="Hello from Qwen3-TTS benchmark.")
    parser.add_argument("--language", default="English")
    parser.add_argument("--speaker", default="", help="Speaker name; defaults to first Python supported speaker")

    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)

    parser.add_argument("--python-device", default="cpu", help="device_map for Python model (e.g. cpu, cuda:0)")
    parser.add_argument("--python-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--attn-implementation", default="", help="Optional HF attn_implementation for GPU runs")

    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--subtalker-temperature", type=float, default=0.9)
    parser.add_argument("--subtalker-top-k", type=int, default=50)
    parser.add_argument("--subtalker-top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)

    parser.add_argument(
        "--equal-token-budget",
        type=int,
        default=0,
        help=(
            "If > 0, overrides --max-new-tokens for both Python and C with a fixed token budget "
            "to make comparisons more apples-to-apples."
        ),
    )

    parser.add_argument(
        "--gate-max-c-over-python-ms-per-token",
        type=float,
        default=0.0,
        help="Fail (exit 1) if c/python median ms_per_token ratio exceeds this value (0 disables gate).",
    )
    parser.add_argument(
        "--gate-max-c-over-python-ms-per-audio-sec",
        type=float,
        default=0.0,
        help="Fail (exit 1) if c/python median ms_per_audio_sec ratio exceeds this value (0 disables gate).",
    )

    parser.add_argument("--output-dir", default="benchmark_output")
    return parser.parse_args()


def _extract_median(summary: dict[str, Any], path: tuple[str, ...]) -> float | None:
    cur: Any = summary
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, (float, int)):
        return float(cur)
    return None


def main() -> int:
    args = parse_args()

    global torch, AutoTokenizer, Qwen3TTSModel
    try:
        import torch as _torch
        from transformers import AutoTokenizer as _AutoTokenizer
        from qwen_tts import Qwen3TTSModel as _Qwen3TTSModel
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing Python dependency: {exc.name}. "
            "Install project dependencies before running benchmark."
        ) from exc

    torch = _torch
    AutoTokenizer = _AutoTokenizer
    Qwen3TTSModel = _Qwen3TTSModel

    if args.warmup_runs < 0 or args.runs <= 0:
        raise ValueError("warmup-runs must be >= 0 and runs must be > 0")
    if args.equal_token_budget < 0:
        raise ValueError("equal-token-budget must be >= 0")

    max_new_tokens_effective = args.equal_token_budget if args.equal_token_budget > 0 else args.max_new_tokens

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_id = args.tokenizer or args.python_model
    print(f"[setup] tokenizer={tokenizer_id}")
    print(f"[setup] python_model={args.python_model}")
    print(f"[setup] c_model_dir={args.c_model_dir}")
    print(f"[setup] runs={args.runs}, warmup_runs={args.warmup_runs}")
    print(f"[setup] max_new_tokens={args.max_new_tokens}, effective_max_new_tokens={max_new_tokens_effective}")
    if args.equal_token_budget > 0:
        print(f"[setup] equal_token_budget enabled: {args.equal_token_budget}")
    if shutil.which("sox") is None:
        print("[setup] warning: sox binary not found; install via `brew install sox` to remove SoX warnings.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, fix_mistral_regex=True)
    token_ids = tokenizer(_chat_template(args.text), return_tensors="pt").input_ids[0].tolist()
    token_id_str = ",".join(str(x) for x in token_ids)
    print(f"[setup] prompt tokens={len(token_ids)}")

    py_summary, speaker, py_median_ms = _bench_python(args, out_dir, max_new_tokens_effective)
    c_summary, c_median_ms = _bench_c(args, out_dir, token_id_str, speaker, max_new_tokens_effective)

    speedup = py_median_ms / c_median_ms if c_median_ms > 0 else None

    py_median_ms_per_token = _extract_median(py_summary, ("normalized", "ms_per_token", "median"))
    c_median_ms_per_token = _extract_median(c_summary, ("normalized", "ms_per_token", "median"))
    py_median_ms_per_audio_sec = _extract_median(py_summary, ("normalized", "ms_per_audio_sec", "median"))
    c_median_ms_per_audio_sec = _extract_median(c_summary, ("normalized", "ms_per_audio_sec", "median"))

    c_over_py_ms_per_token = (
        _safe_div(c_median_ms_per_token, py_median_ms_per_token)
        if c_median_ms_per_token is not None and py_median_ms_per_token is not None
        else None
    )
    c_over_py_ms_per_audio_sec = (
        _safe_div(c_median_ms_per_audio_sec, py_median_ms_per_audio_sec)
        if c_median_ms_per_audio_sec is not None and py_median_ms_per_audio_sec is not None
        else None
    )

    report = {
        "meta": {
            "hostname": os.uname().nodename,
            "platform": sys.platform,
            "python_version": sys.version,
            "torch_version": torch.__version__,
        },
        "config": {
            "text": args.text,
            "language": args.language,
            "speaker": speaker,
            "warmup_runs": args.warmup_runs,
            "runs": args.runs,
            "python_model": args.python_model,
            "c_model_dir": args.c_model_dir,
            "c_bin": args.c_bin,
            "python_device": args.python_device,
            "python_dtype": args.python_dtype,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "subtalker_temperature": args.subtalker_temperature,
            "subtalker_top_k": args.subtalker_top_k,
            "subtalker_top_p": args.subtalker_top_p,
            "max_new_tokens": args.max_new_tokens,
            "equal_token_budget": args.equal_token_budget,
            "effective_max_new_tokens": max_new_tokens_effective,
            "gate_max_c_over_python_ms_per_token": args.gate_max_c_over_python_ms_per_token,
            "gate_max_c_over_python_ms_per_audio_sec": args.gate_max_c_over_python_ms_per_audio_sec,
        },
        "python": py_summary,
        "c": c_summary,
        "comparison": {
            "median_speedup_c_over_python": speedup,
            "python_median_ms": py_median_ms,
            "c_median_ms": c_median_ms,
            "normalized": {
                "python_median_ms_per_token": py_median_ms_per_token,
                "c_median_ms_per_token": c_median_ms_per_token,
                "c_over_python_median_ms_per_token": c_over_py_ms_per_token,
                "python_median_ms_per_audio_sec": py_median_ms_per_audio_sec,
                "c_median_ms_per_audio_sec": c_median_ms_per_audio_sec,
                "c_over_python_median_ms_per_audio_sec": c_over_py_ms_per_audio_sec,
            },
        },
    }

    report_path = out_dir / "benchmark_results.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Benchmark Summary ===")
    print(f"Python median: {py_median_ms:.2f} ms")
    print(f"C median:      {c_median_ms:.2f} ms")
    if speedup is not None:
        print(f"C speedup:     {speedup:.2f}x (median)")
    if c_over_py_ms_per_token is not None:
        print(f"C/Py ms/token: {c_over_py_ms_per_token:.2f}x")
    if c_over_py_ms_per_audio_sec is not None:
        print(f"C/Py ms/audio-sec: {c_over_py_ms_per_audio_sec:.2f}x")
    print(f"Report:        {report_path}")

    gate_failures: list[str] = []
    if args.gate_max_c_over_python_ms_per_token > 0:
        if c_over_py_ms_per_token is None:
            gate_failures.append("ms_per_token ratio unavailable")
        elif c_over_py_ms_per_token > args.gate_max_c_over_python_ms_per_token:
            gate_failures.append(
                "c_over_python_median_ms_per_token="
                f"{c_over_py_ms_per_token:.3f} > {args.gate_max_c_over_python_ms_per_token:.3f}"
            )

    if args.gate_max_c_over_python_ms_per_audio_sec > 0:
        if c_over_py_ms_per_audio_sec is None:
            gate_failures.append("ms_per_audio_sec ratio unavailable")
        elif c_over_py_ms_per_audio_sec > args.gate_max_c_over_python_ms_per_audio_sec:
            gate_failures.append(
                "c_over_python_median_ms_per_audio_sec="
                f"{c_over_py_ms_per_audio_sec:.3f} > {args.gate_max_c_over_python_ms_per_audio_sec:.3f}"
            )

    if gate_failures:
        print("\n[gate] FAILED")
        for msg in gate_failures:
            print(f"[gate] {msg}")
        return 1

    if args.gate_max_c_over_python_ms_per_token > 0 or args.gate_max_c_over_python_ms_per_audio_sec > 0:
        print("[gate] PASSED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
