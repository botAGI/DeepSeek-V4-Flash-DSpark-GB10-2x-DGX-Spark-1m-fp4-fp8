# Benchmark checkpoint — NVFP4-KV C16 concurrency reproduction (2026-06-30)

> Measurements of tonyd2wild's NVFP4-KV build (https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark, commit 89bb82b) on our 2x DGX Spark. Reproduced and measured by us; recipe authored by tonyd2wild -- see CREDITS.md.

Independent reproduction of tonyd2wild's published concurrency numbers for the NVFP4-KV DSpark
build (commit `89bb82b`). **Method:** his Keys-concurrency harness
([`harness/bench_concurrent.py`](harness/bench_concurrent.py),
[`harness/staggered_bench.py`](harness/staggered_bench.py)) unmodified, on **2× DGX Spark GB10,
TP=2 over RoCE**, byte-identical upstream Stage-C NVFP4 image.

We did not change the build or the harness. See [`../CREDITS.md`](../CREDITS.md).

---

## Results vs published reference

| metric | tonyd2wild (published) | our reproduction |
| --- | ---: | ---: |
| **C16-static aggregate** (200K, seqs=16) | 315.1 | **324** (range 282–330) |
| **C16-static, control** (`WO_PROJECTION` lever OFF) | — | **319** |
| **C16-staggered** (200K, seqs=16) | 205.0 | 191–196 |
| **single-stream, code** | 67 | 55–63 |

**The control is the proof:** with the `WO_PROJECTION` lever **OFF**, C16-static lands on
**319**, re-deriving tonyd2wild's published **315.1** almost exactly. That validates his
benchmark methodology — and shows our **324** (lever ON) is well inside his own run-to-run
variance, not a real win.

---

## Verdict

**Clean independent reproduction; we do not claim to have beaten it.** Our aggregate C16 edges
his headline (324 vs 315.1) but inside his variance; the lever-OFF control re-derives 315.1;
staggered we slightly trail (−~4%); single-stream we tie or slightly trail. This is *his*
recipe on *our* hardware producing *his* numbers — exactly what an honest reproduction should
look like.

> The seqs=32 concurrency extension (a config tonyd2wild never benched) is documented
> separately; it scales aggregate throughput only at a sub-1M per-request context ceiling and
> is **not** a claim of 32 concurrent full-1M contexts.
