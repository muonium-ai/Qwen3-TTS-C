#!/usr/bin/env python3
"""Regression check: C generation should not always run to max_tokens.

Runs one deterministic decode and asserts that generation stops with EOS
before the max token limit on a standard prompt.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def _parse_stop(stderr: str) -> tuple[str, int, int]:
    stop_match = re.search(r"Stop:\s+([a-z_]+)\s+at step\s+(\d+)", stderr)
    gen_match = re.search(r"Generated\s+(\d+)\s+codec tokens", stderr)
    if not stop_match:
        raise RuntimeError("Could not parse stop reason from C stderr")
    if not gen_match:
        raise RuntimeError("Could not parse generated token count from C stderr")
    return stop_match.group(1), int(stop_match.group(2)), int(gen_match.group(1))


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="EOS regression test for qwen-tts C binary")
    ap.add_argument("--c-bin", default=str((here / ".." / "qwen-tts").resolve()))
    ap.add_argument("--model-dir", default="tmp/model")
    ap.add_argument("--tokens-file", default=str((here / "tokens_great_power.txt").resolve()))
    ap.add_argument("--speaker", default="aiden")
    ap.add_argument("--language", default="English")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    c_bin = Path(args.c_bin).resolve()
    if not c_bin.exists():
        print(f"Missing C binary: {c_bin}", file=sys.stderr)
        return 2
    if not Path(args.model_dir).is_dir():
        print(f"Missing model dir: {args.model_dir}", file=sys.stderr)
        return 2
    if not Path(args.tokens_file).is_file():
        print(f"Missing tokens file: {args.tokens_file}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="qwen_eos_reg_") as td:
        wav_out = os.path.join(td, "out.wav")
        cmd = [
            str(c_bin),
            "-d", args.model_dir,
            "-f", args.tokens_file,
            "-s", args.speaker,
            "-l", args.language,
            "-o", wav_out,
            "-v", "-v",
            "--max-tokens", str(args.max_tokens),
            "--temperature", "1.0",
            "--top-k", "1",
            "--top-p", "1.0",
            "--repetition-penalty", "1.0",
            "--subtalker-temperature", "1.0",
            "--subtalker-top-k", "1",
            "--subtalker-top-p", "1.0",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        print(f"C command failed with exit code {proc.returncode}", file=sys.stderr)
        return 1

    stop_reason, stop_step, generated = _parse_stop(proc.stderr)

    print(f"EOS regression summary: stop_reason={stop_reason}, stop_step={stop_step}, generated={generated}")

    if stop_reason != "eos":
        print("Regression: generation did not stop on EOS", file=sys.stderr)
        return 1
    if generated >= args.max_tokens or stop_step >= args.max_tokens:
        print("Regression: generation reached max_tokens", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
