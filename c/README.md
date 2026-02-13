# Qwen3-TTS-C

Pure C inference engine for [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS), Alibaba's open-source text-to-speech model.

Generates natural 24kHz speech from text using a two-stage architecture (Talker LM + Neural Codec Decoder). Zero external dependencies beyond a C compiler and BLAS.

## Features

- **Pure C** — single-binary, no Python/PyTorch runtime needed
- **Fast** — mmap'd SafeTensors weights, BF16 storage with F32 compute, BLAS-accelerated matmul
- **macOS + Linux** — Accelerate framework (macOS) or OpenBLAS (Linux)
- **CustomVoice** — speaker and language selection via config
- **Streaming-ready** — autoregressive codec generation with progress callbacks

## Architecture

```
Text → [BPE Tokenizer] → Token IDs
                              ↓
                    ┌─────────────────┐
                    │   Talker LM     │  config-driven transformer (M-RoPE, GQA, QK-Norm)
                    │   + Sub-Talker  │  5-layer code predictor (31 codebook groups)
                    └────────┬────────┘
                             ↓
                     32 × Codec Tokens per step
                             ↓
                    ┌─────────────────┐
                    │  Codec Decoder  │  SplitRVQ → Transformer → ConvNeXt → BigVGAN
                    └────────┬────────┘
                             ↓
                    24kHz PCM Audio → WAV
```

## Building

### macOS

```bash
make
```

Uses Apple's Accelerate framework automatically.

### Linux

Install OpenBLAS first:

```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# Fedora/RHEL
sudo dnf install openblas-devel
```

Then build:

```bash
make
```

### Debug build

```bash
make debug
```

## Model Download

Download the Qwen3-TTS model from HuggingFace:

```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-TTS --local-dir ./model

# Or using git-lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-TTS ./model
```

The model directory should contain:
```
model/
  config.json
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
  model.safetensors.index.json
  speech_tokenizer/
    config.json
    model.safetensors
```

## Usage

### Tokenize text (Python helper)

Since this engine accepts pre-tokenized BPE IDs, use the Qwen2 tokenizer in Python to convert text:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS")

text = "Hello, my name is Qwen. I am an open-source text to speech model."
chat = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
ids = tok(chat, return_tensors="pt").input_ids[0].tolist()

# Write as comma-separated IDs
with open("tokens.txt", "w") as f:
    f.write(",".join(str(i) for i in ids))

print(f"Tokenized {len(ids)} tokens -> tokens.txt")
```

### Generate speech

```bash
# From token IDs file
./qwen_tts -d ./model -f tokens.txt -s "Chelsie" -l "english" -o speech.wav -v

# From inline token IDs
./qwen_tts -d ./model -t "151644,77091,198,9707,11,..." -s "Chelsie" -l "english" -o speech.wav

# List available speakers/languages (printed with -v -v)
./qwen_tts -d ./model -t "151644,77091,198,9707,151645,198,151644,77091,198" -v -v
```

### Command-line options

| Option | Description |
|--------|-------------|
| `-d <path>` | Model directory (required) |
| `-t <ids>` | Comma-separated token IDs |
| `-f <file>` | Read token IDs from file |
| `-s <speaker>` | Speaker name from config |
| `-l <language>` | Language: auto, chinese, english, etc. |
| `-o <path>` | Output WAV path (default: output.wav) |
| `-v` | Verbose output (repeat for more detail) |
| `--temperature <f>` | Sampling temperature (default: 0.9) |
| `--top-k <n>` | Top-K sampling (default: 50) |
| `--top-p <f>` | Top-P (nucleus) sampling (default: 1.0) |
| `--repetition-penalty <f>` | Repetition penalty (default: 1.05) |
| `--max-tokens <n>` | Max codec tokens (default: 4096) |

## File Structure

```
c/
  qwen_tts.h              Main header: structs, constants, API declarations
  qwen_tts.c              Config parsing, weight loading, generate pipeline
  qwen_tts_kernels.h      Math kernel declarations
  qwen_tts_kernels.c      RMSNorm, matmul, RoPE, sampling, convolutions
  qwen_tts_safetensors.h  SafeTensors reader API
  qwen_tts_safetensors.c  Mmap-based SafeTensors loader
  qwen_tts_talker.c       Talker transformer (prefill + single-token + sub-talker)
  qwen_tts_codec.c        Codec decoder (RVQ, transformer, ConvNeXt, BigVGAN vocoder)
  qwen_tts_audio.c        WAV file writer
  main.c                  CLI entry point
Makefile                  Build system
```

## Technical Details

- **Talker**: Config-driven GQA transformer, M-RoPE, QK-Norm, SwiGLU MLP
- **Qwen3-TTS 0.6B CustomVoice config**: 28 layers, 16 Q-heads, 8 KV-heads, head_dim=128
- **Sub-Talker**: 5-layer code predictor generating 31 additional codebook groups per step
- **Codec Decoder**: SplitRVQ (16 codebooks × 2048 entries) → 8-layer sliding-window transformer (window=72) → ConvNeXt upsampler → BigVGAN vocoder
- **Weights**: BF16 mmap for talker/sub-talker (zero-copy), F32 copies for codec decoder
- **Sample rate**: 24kHz mono, 12.5Hz codec frame rate (1920 samples/frame)

## Performance

On Apple M-series Macs with Accelerate, expect roughly real-time generation. The bottleneck is the autoregressive talker loop; codec decoding is fast.

## Benchmark Notes

- Use equal-work comparisons. Compare `ms/token` or runs that produce similar audio duration, not just wall-clock.
- If C does not sample EOS early, it can run until `--max-tokens`, producing much longer audio and making speed comparisons misleading.
- `make benchmark` currently uses `BENCH_MAX_TOKENS=512`, so C may generate up to `40.96s` audio (`512 * 0.08s`) while Python may stop earlier.
- For quick parity checks, start with:

```bash
make benchmark BENCH_RUNS=1 BENCH_WARMUP=0 BENCH_MAX_TOKENS=128
make validate-eos BENCH_SPEAKER=aiden
make test-eos-regression BENCH_SPEAKER=aiden
```

## Known Issues

- Talker attention shape handling is under active work.
- The model config defines explicit `head_dim=128`.
- Some C paths still derive talker head dim as `hidden/heads`, which can cause incorrect attention math, poor EOS behavior, and inflated generation length.
- Until this is fixed, prioritize normalized metrics (`ms/token`, `ms/audio_sec`) over raw total runtime.

## Credits

- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) by Alibaba Cloud
- C port architecture inspired by [antirez/qwen-asr](https://github.com/antirez/qwen-asr)

## License

Same license as the original Qwen3-TTS model. See [LICENSE](LICENSE) for details.
