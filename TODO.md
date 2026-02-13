# TODO

## P0 - Correctness (must fix before trusting perf numbers)

- [x] Fix talker `head_dim` handling to use config value (`talker_config.head_dim`) end-to-end.
- [x] Audit talker attention tensor shapes (`q_proj`, `k_proj`, `v_proj`, `o_proj`) against model safetensor metadata and fail fast on mismatch.
- [x] Validate EOS behavior parity with Python for the same prompt/sampling settings (token trace + stop step).
- [x] Add regression test that checks C does not always run to `max_tokens` for a standard prompt.

## P1 - Benchmark quality

- [x] Extend `scripts/benchmark_py_vs_c.py` report with generated token count and stop reason (EOS vs max token limit).
- [x] Add normalized comparison metrics: `ms/token`, `tokens/s`, and `ms/audio_sec`.
- [x] Add a benchmark mode that enforces equal decode length (fixed generated token budget for both paths).
- [x] Add CI benchmark gate using normalized metrics instead of only total elapsed time.

## P2 - Performance follow-ups

- [x] Re-profile after head-dim fix to identify true hotspots.
- [x] Vectorize remaining talker attention inner loops (dot + weighted sum) where BLAS is not used.
- [x] Improve subtalker throughput with better batching/reuse for repeated small matvec calls.
- [x] Reduce per-step allocator/memory traffic in generation loops.
- [x] Investigate codec decoder kernel fusion opportunities after talker parity is stable.
  First pass done: in-place SnakeBeta fusion in vocoder blocks/resunits/final stage to remove extra activation buffers.

Latest profile snapshot (2026-02-13, `./qwen-tts -v -v`, deterministic decode):
- Talker decode: ~11.2s for 74 tokens (~151 ms/token).
- Codec total: ~4.16s, dominated by vocoder (~3.90s).
