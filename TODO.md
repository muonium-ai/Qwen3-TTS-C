# TODO

## P0 - Correctness (must fix before trusting perf numbers)

- [ ] Fix talker `head_dim` handling to use config value (`talker_config.head_dim`) end-to-end.
- [ ] Audit talker attention tensor shapes (`q_proj`, `k_proj`, `v_proj`, `o_proj`) against model safetensor metadata and fail fast on mismatch.
- [ ] Validate EOS behavior parity with Python for the same prompt/sampling settings (token trace + stop step).
- [ ] Add regression test that checks C does not always run to `max_tokens` for a standard prompt.

## P1 - Benchmark quality

- [ ] Extend `scripts/benchmark_py_vs_c.py` report with generated token count and stop reason (EOS vs max token limit).
- [ ] Add normalized comparison metrics: `ms/token`, `tokens/s`, and `ms/audio_sec`.
- [ ] Add a benchmark mode that enforces equal decode length (fixed generated token budget for both paths).
- [ ] Add CI benchmark gate using normalized metrics instead of only total elapsed time.

## P2 - Performance follow-ups

- [ ] Re-profile after head-dim fix to identify true hotspots.
- [ ] Vectorize remaining talker attention inner loops (dot + weighted sum) where BLAS is not used.
- [ ] Improve subtalker throughput with better batching/reuse for repeated small matvec calls.
- [ ] Reduce per-step allocator/memory traffic in generation loops.
- [ ] Investigate codec decoder kernel fusion opportunities after talker parity is stable.
