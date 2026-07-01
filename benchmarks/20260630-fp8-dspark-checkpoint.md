# fp8 DSpark on GB10 — measurement checkpoint (2026-06-30)

A dated snapshot of the verified fp8-KV DSpark numbers on this build. All figures are
apples-to-apples: **`completion_tokens / wall_time`**, non-streaming, same prompt class,
same engine.

## Setup

- **Hardware:** 2× DGX Spark (GB10, `sm_121`), TP=2 over RoCE.
- **Engine:** vLLM, DSpark speculative method, fp8 KV cache.
- **Method:** throughput = `completion_tokens / wall_time` (do **not** count SSE events —
  that undercounts spec-decode by ~2.5×).

## Numbers

| Measurement | Result |
|---|---:|
| No-spec autoregressive (spec off) | **26.7 t/s** |
| **DSpark single-stream** | **32 → 32.4 t/s** |
| **DSpark concurrency, 8 streams** | **99 t/s** aggregate |
| **DSpark concurrency, 12 streams** | **141–146 t/s** aggregate |
| Mean draft-acceptance length | **~2.0** |

## fp8 vs NVFP4 KV

The **NVFP4-KV** variant (community-derived — tonyd2wild's recipe) is the faster **single-stream**
path. See our measurements in [`20260630-nvfp4-1m-context-curve-checkpoint.md`](20260630-nvfp4-1m-context-curve-checkpoint.md)
and [`20260630-nvfp4-c16-reproduction-checkpoint.md`](20260630-nvfp4-c16-reproduction-checkpoint.md)
in this repo, and the upstream recipe at
<https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark>.

_Recorded 2026-06-30._
