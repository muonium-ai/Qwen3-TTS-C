#!/usr/bin/env python3
"""Validate EOS behavior parity between Python and C generation paths.

This runs one deterministic decode (top-k=1) on both implementations,
compares stop reason/step, and optionally compares token trace.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _chat_template(text: str) -> str:
    return f"<|im_start|>assistant\\n{text}<|im_end|>\\n<|im_start|>assistant\\n"


def _parse_c_output(stderr: str) -> tuple[str, int, list[int], int]:
    stop_match = re.search(r"Stop:\s+([a-z_]+)\s+at step\s+(\d+)", stderr)
    if not stop_match:
        raise RuntimeError("Could not parse C stop reason/step from stderr.")

    gen_match = re.search(r"Generated\s+(\d+)\s+codec tokens", stderr)
    if not gen_match:
        raise RuntimeError("Could not parse C generated token count from stderr.")

    trace_match = re.search(r"Token trace:\s*([0-9,\-\s]*)", stderr)
    tokens: list[int] = []
    if trace_match:
        raw = trace_match.group(1).strip()
        if raw:
            tokens = [int(x.strip()) for x in raw.split(",") if x.strip()]

    stop_reason = stop_match.group(1)
    stop_step = int(stop_match.group(2))
    generated = int(gen_match.group(1))
    return stop_reason, stop_step, tokens, generated


def _resolve_dtype(dtype_name: str, torch_mod: Any) -> Any:
    mapping = {
        "float32": torch_mod.float32,
        "float16": torch_mod.float16,
        "bfloat16": torch_mod.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate EOS parity for Python vs C Qwen3-TTS")
    ap.add_argument("--python-model", required=True)
    ap.add_argument("--c-model-dir", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--c-bin", default="./qwen-tts")
    ap.add_argument("--text", default="Hello from Qwen3-TTS EOS parity validation.")
    ap.add_argument("--language", default="English")
    ap.add_argument("--speaker", default="")
    ap.add_argument("--python-device", default="cpu")
    ap.add_argument("--python-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--subtalker-top-k", type=int, default=1)
    ap.add_argument("--subtalker-top-p", type=float, default=1.0)
    ap.add_argument("--subtalker-temperature", type=float, default=1.0)
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer
        from qwen_tts import Qwen3TTSModel
    except Exception as exc:  # pragma: no cover
        print(f"Missing dependency: {exc}", file=sys.stderr)
        return 2

    tok_id = args.tokenizer or args.python_model
    tokenizer = AutoTokenizer.from_pretrained(tok_id, fix_mistral_regex=True)
    token_ids = tokenizer(_chat_template(args.text), return_tensors="pt").input_ids[0].tolist()
    token_str = ",".join(str(i) for i in token_ids)

    model = Qwen3TTSModel.from_pretrained(
        args.python_model,
        device_map=args.python_device,
        dtype=_resolve_dtype(args.python_dtype, torch),
    )

    speaker = args.speaker
    if not speaker:
        supported = model.get_supported_speakers() or []
        if not supported:
            raise RuntimeError("No supported speakers returned by Python model; pass --speaker")
        speaker = supported[0]

    input_ids = model._tokenize_texts([model._build_assistant_text(args.text)])

    py_codes_list, _ = model.model.generate(
        input_ids=input_ids,
        languages=[args.language],
        speakers=[speaker],
        non_streaming_mode=True,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        subtalker_dosample=True,
        subtalker_top_k=args.subtalker_top_k,
        subtalker_top_p=args.subtalker_top_p,
        subtalker_temperature=args.subtalker_temperature,
    )

    py_codes = py_codes_list[0]
    py_tokens = py_codes[:, 0].tolist() if py_codes.numel() > 0 else []
    py_stop_step = len(py_tokens)
    py_stop_reason = "max_tokens" if py_stop_step >= args.max_new_tokens else "eos"

    c_bin = str(Path(args.c_bin).resolve())
    if not Path(c_bin).exists():
        raise FileNotFoundError(f"C binary not found: {c_bin}")

    with tempfile.TemporaryDirectory(prefix="qwen_eos_parity_") as td:
        wav_out = os.path.join(td, "c_parity.wav")
        cmd = [
            c_bin,
            "-d", args.c_model_dir,
            "-t", token_str,
            "-o", wav_out,
            "-v", "-v",
            "-s", speaker,
            "-l", args.language,
            "--temperature", str(args.temperature),
            "--top-k", str(args.top_k),
            "--top-p", str(args.top_p),
            "--repetition-penalty", str(args.repetition_penalty),
            "--subtalker-temperature", str(args.subtalker_temperature),
            "--subtalker-top-k", str(args.subtalker_top_k),
            "--subtalker-top-p", str(args.subtalker_top_p),
            "--max-tokens", str(args.max_new_tokens),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"C command failed with exit code {proc.returncode}")

        c_stop_reason, c_stop_step, c_tokens, c_generated = _parse_c_output(proc.stderr)

    ok = True
    if c_stop_reason != py_stop_reason:
        ok = False
        print(f"Mismatch stop_reason: python={py_stop_reason} c={c_stop_reason}")

    if c_stop_step != py_stop_step:
        ok = False
        print(f"Mismatch stop_step: python={py_stop_step} c={c_stop_step}")

    if c_tokens and c_tokens != py_tokens:
        first_diff = next((i for i, (a, b) in enumerate(zip(py_tokens, c_tokens)) if a != b), None)
        if first_diff is None and len(py_tokens) != len(c_tokens):
            first_diff = min(len(py_tokens), len(c_tokens))
        print(
            "Token trace differs "
            f"(python_len={len(py_tokens)} c_len={len(c_tokens)} first_diff={first_diff})"
        )
    elif not c_tokens:
        print("Warning: C token trace not found in stderr; rebuild C binary with latest changes.")

    print("EOS parity summary:")
    print(f"  speaker: {speaker}")
    print(f"  python: stop_reason={py_stop_reason}, stop_step={py_stop_step}, tokens={len(py_tokens)}")
    print(f"  c:      stop_reason={c_stop_reason}, stop_step={c_stop_step}, tokens={c_generated}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
