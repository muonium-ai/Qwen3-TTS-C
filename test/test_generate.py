#!/usr/bin/env python3
"""
test_generate.py - Generate test audio: "With great power comes great responsibility."

This script:
  1. Downloads the Qwen3-TTS model (requires HuggingFace login for gated model)
  2. Generates audio using the ORIGINAL Python model (reference)
  3. Generates audio using the C engine
  4. Saves both for comparison

Prerequisites:
  pip install transformers torch soundfile accelerate
  huggingface-cli login

Usage:
  python test/test_generate.py --model-dir ./model
  python test/test_generate.py --download   # Download model first
"""

import argparse
import os
import subprocess
import sys

QUOTE = "With great power comes great responsibility."
TOKEN_FILE = os.path.join(os.path.dirname(__file__), "tokens_great_power.txt")
C_BINARY = os.path.join(os.path.dirname(__file__), "..", "qwen-tts")


def download_model(model_dir):
    """Download Qwen3-TTS model from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading Qwen3-TTS to {model_dir}...")
    snapshot_download(
        repo_id="Qwen/Qwen3-TTS",
        local_dir=model_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Model downloaded to {model_dir}")


def generate_with_python(model_dir, output_path):
    """Generate audio using the original Python Qwen3-TTS model."""
    try:
        import torch
        import soundfile as sf
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    except ImportError as e:
        print(f"Cannot run Python reference: {e}")
        print("Install: pip install torch soundfile transformers accelerate")
        return False

    print(f"Loading Python model from {model_dir}...")
    model = Qwen3TTSModel.from_pretrained(model_dir)
    model.eval()

    print(f"Generating: \"{QUOTE}\"")
    with torch.no_grad():
        audio = model.generate(
            text=QUOTE,
            speaker="Chelsie",
            language="english",
        )

    if audio is not None:
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        sf.write(output_path, audio, 24000)
        duration = len(audio) / 24000
        print(f"Python reference saved: {output_path} ({duration:.2f}s)")
        return True
    else:
        print("Python model produced no audio")
        return False


def generate_with_c(model_dir, output_path):
    """Generate audio using the C engine."""
    if not os.path.isfile(C_BINARY):
        print(f"C binary not found at {C_BINARY}")
        print("Build first: make")
        return False

    cmd = [
        C_BINARY,
        "-d", model_dir,
        "-f", TOKEN_FILE,
        "-s", "Chelsie",
        "-l", "english",
        "-o", output_path,
        "-v",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stderr)
    if result.returncode != 0:
        print(f"C engine failed with exit code {result.returncode}")
        return False
    print(f"C engine output saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-TTS audio generation")
    parser.add_argument("--model-dir", default="./model",
                        help="Path to Qwen3-TTS model directory")
    parser.add_argument("--download", action="store_true",
                        help="Download the model first")
    parser.add_argument("--python-only", action="store_true",
                        help="Only run Python reference")
    parser.add_argument("--c-only", action="store_true",
                        help="Only run C engine")
    parser.add_argument("--output-dir", default="./test",
                        help="Output directory for WAV files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.download:
        download_model(args.model_dir)

    if not os.path.isdir(args.model_dir):
        print(f"Model directory not found: {args.model_dir}")
        print("Download the model first:")
        print(f"  python {sys.argv[0]} --download --model-dir {args.model_dir}")
        print("  OR")
        print("  huggingface-cli login")
        print("  huggingface-cli download Qwen/Qwen3-TTS --local-dir ./model")
        sys.exit(1)

    print(f'Quote: "{QUOTE}"')
    print(f"Token file: {TOKEN_FILE}")
    print()

    if not args.c_only:
        py_out = os.path.join(args.output_dir, "great_power_python.wav")
        generate_with_python(args.model_dir, py_out)
        print()

    if not args.python_only:
        c_out = os.path.join(args.output_dir, "great_power_c.wav")
        generate_with_c(args.model_dir, c_out)


if __name__ == "__main__":
    main()
