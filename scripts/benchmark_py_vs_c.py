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
    # Prefer the summary line printed by main.c with verbose mode:
    #   Total:   1234.5 ms
    match = re.search(r"^\s*Total:\s+([0-9]+(?:\.[0-9]+)?)\s*ms\s*$", stderr, re.MULTILINE)
    if match:
        return float(match.group(1))

    # Fallback to the generation line in qwen_tts.c:
    #   Total: 1234.5 ms (X.XX s audio, Y.YYx realtime)
    match = re.search(r"Total:\s+([0-9]+(?:\.[0-9]+)?)\s*ms\s*\(", stderr)
    if match:
        return float(match.group(1))

    return None


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _bench_python(args: argparse.Namespace, out_dir: Path) -> tuple[dict[str, Any], str, float]:
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

    def run_once(save_path: Path | None) -> RunResult:
        _sync_if_cuda(args.python_device)
        start = time.perf_counter()
        wavs, sr = model.generate_custom_voice(
            text=args.text,
            language=args.language,
            speaker=speaker,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            subtalker_dosample=True,
            subtalker_top_k=args.subtalker_top_k,
            subtalker_top_p=args.subtalker_top_p,
            subtalker_temperature=args.subtalker_temperature,
            max_new_tokens=args.max_new_tokens,
        )
        _sync_if_cuda(args.python_device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        audio_sec = float(len(wavs[0])) / float(sr)
        rtf = audio_sec / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0

        if save_path is not None:
            _write_wav_mono_16bit(save_path, np.asarray(wavs[0]), int(sr))

        return RunResult(elapsed_ms=elapsed_ms, audio_sec=audio_sec, rtf=rtf)

    for i in range(args.warmup_runs):
        run_once(None)
        print(f"[python] warmup {i + 1}/{args.warmup_runs} done", flush=True)

    results: list[RunResult] = []
    for i in range(args.runs):
        save_path = out_dir / "python_output.wav" if i == args.runs - 1 else None
        rr = run_once(save_path)
        results.append(rr)
        print(
            f"[python] run {i + 1}/{args.runs}: {rr.elapsed_ms:.2f} ms, "
            f"audio={rr.audio_sec:.3f}s, rtf={rr.rtf:.3f}x",
            flush=True,
        )

    elapsed_values = [x.elapsed_ms for x in results]
    rtf_values = [x.rtf for x in results]
    audio_values = [x.audio_sec for x in results]

    summary: dict[str, Any] = {
        "load_ms": load_ms,
        "speaker": speaker,
        "runs": [asdict(x) for x in results],
        "timing": _summary(elapsed_values),
        "rtf": {
            "mean": statistics.fmean(rtf_values),
            "median": statistics.median(rtf_values),
        },
        "audio_sec": {
            "mean": statistics.fmean(audio_values),
            "median": statistics.median(audio_values),
        },
    }
    return summary, speaker, statistics.median(elapsed_values)


def _bench_c(
    args: argparse.Namespace,
    out_dir: Path,
    token_ids: str,
    speaker: str,
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
            "--max-tokens",
            str(args.max_new_tokens),
        ]

        start = time.perf_counter()
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        audio_sec = _wav_duration_sec(output_wav)
        rtf = audio_sec / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0

        internal_ms = _parse_c_internal_total_ms(proc.stderr)

        if tmp_wav_path is not None:
            try:
                tmp_wav_path.unlink()
            except FileNotFoundError:
                pass

        return RunResult(
            elapsed_ms=elapsed_ms,
            audio_sec=audio_sec,
            rtf=rtf,
            internal_total_ms=internal_ms,
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
        print(
            f"[c] run {i + 1}/{args.runs}: {rr.elapsed_ms:.2f} ms{internal_str}, "
            f"audio={rr.audio_sec:.3f}s, rtf={rr.rtf:.3f}x",
            flush=True,
        )

    elapsed_values = [x.elapsed_ms for x in results]
    rtf_values = [x.rtf for x in results]
    audio_values = [x.audio_sec for x in results]
    internal_values = [x.internal_total_ms for x in results if x.internal_total_ms is not None]

    summary: dict[str, Any] = {
        "runs": [asdict(x) for x in results],
        "timing": _summary(elapsed_values),
        "rtf": {
            "mean": statistics.fmean(rtf_values),
            "median": statistics.median(rtf_values),
        },
        "audio_sec": {
            "mean": statistics.fmean(audio_values),
            "median": statistics.median(audio_values),
        },
    }

    if internal_values:
        summary["internal_timing"] = _summary(internal_values)
        summary["internal_timing"]["count"] = len(internal_values)

    return summary, statistics.median(elapsed_values)


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

    parser.add_argument("--output-dir", default="benchmark_output")
    return parser.parse_args()


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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_id = args.tokenizer or args.python_model
    print(f"[setup] tokenizer={tokenizer_id}")
    print(f"[setup] python_model={args.python_model}")
    print(f"[setup] c_model_dir={args.c_model_dir}")
    print(f"[setup] runs={args.runs}, warmup_runs={args.warmup_runs}")
    if shutil.which("sox") is None:
        print("[setup] warning: sox binary not found; install via `brew install sox` to remove SoX warnings.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, fix_mistral_regex=True)
    token_ids = tokenizer(_chat_template(args.text), return_tensors="pt").input_ids[0].tolist()
    token_id_str = ",".join(str(x) for x in token_ids)
    print(f"[setup] prompt tokens={len(token_ids)}")

    py_summary, speaker, py_median_ms = _bench_python(args, out_dir)
    c_summary, c_median_ms = _bench_c(args, out_dir, token_id_str, speaker)

    speedup = py_median_ms / c_median_ms if c_median_ms > 0 else None

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
        },
        "python": py_summary,
        "c": c_summary,
        "comparison": {
            "median_speedup_c_over_python": speedup,
            "python_median_ms": py_median_ms,
            "c_median_ms": c_median_ms,
        },
    }

    report_path = out_dir / "benchmark_results.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Benchmark Summary ===")
    print(f"Python median: {py_median_ms:.2f} ms")
    print(f"C median:      {c_median_ms:.2f} ms")
    if speedup is not None:
        print(f"C speedup:     {speedup:.2f}x (median)")
    print(f"Report:        {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
