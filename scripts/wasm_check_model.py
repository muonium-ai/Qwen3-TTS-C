#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys


def discover_files(model_dir: Path) -> list[str]:
    files = [
        "config.json",
        "speech_tokenizer/config.json",
        "speech_tokenizer/model.safetensors",
    ]
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        files.append("model.safetensors.index.json")
        with index_path.open("r", encoding="utf-8") as f:
            index_json = json.load(f)
        weight_map = index_json.get("weight_map") or {}
        files.extend(sorted(set(weight_map.values())))
    else:
        files.append("model.safetensors")
    return files


def main() -> int:
    p = argparse.ArgumentParser(description="Check browser WASM model payload size.")
    p.add_argument("--model-dir", required=True, help="Model directory path")
    p.add_argument("--max-mib", type=float, default=1400.0, help="Max payload size in MiB")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"Error: model directory not found: {model_dir}")
        return 1

    files = discover_files(model_dir)
    missing = [rel for rel in files if not (model_dir / rel).exists()]
    if missing:
        print("Error: missing model files for wasm check:")
        for rel in missing:
            print(f"  - {rel}")
        return 1

    total_bytes = sum((model_dir / rel).stat().st_size for rel in files)
    total_mib = total_bytes / (1024 * 1024)
    print(f"WASM model payload: {total_mib:.1f} MiB")
    print(f"WASM model budget: {args.max_mib:.0f} MiB")

    limit_bytes = int(args.max_mib * 1024 * 1024)
    if total_bytes > limit_bytes:
        print(f"Error: payload exceeds browser WASM budget by {total_mib - args.max_mib:.1f} MiB")
        return 1

    print("Model payload is within browser WASM budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
