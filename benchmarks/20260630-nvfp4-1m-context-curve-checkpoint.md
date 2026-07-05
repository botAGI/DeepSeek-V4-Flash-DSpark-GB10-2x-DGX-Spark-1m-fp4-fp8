# Benchmark checkpoint — NVFP4-KV 1M-context single-stream curve (2026-06-30)

> Measurements of tonyd2wild's NVFP4-KV build (https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark, commit 89bb82b) on our 2x DGX Spark. Reproduced and measured by us; recipe authored by tonyd2wild -- see CREDITS.md.

NVFP4-KV DSpark build (`DeepSeek-V4-Flash-DSpark`) on 2× DGX Spark (GB10), TP=2 over QSFP 200G
RoCEv2, byte-identical upstream Stage-C image (`kv_cache_dtype=nvfp4_ds_mla`, 584-byte padded
sparse-MLA envelope), MTP5 DSpark, `max_model_len=1048576`, `gpu_memory_utilization=0.80`.

Single-stream throughput is **`completion_tokens / wall_time`, non-streaming** (SSE-event
counting undercounts spec-decode by ~2.5× — never used). Acceptance read from vLLM's own
`SpecDecoding metrics` log line over the timed window.

This is a **reproduction-and-measurement** result on tonyd2wild's recipe (commit `89bb82b`).
We did not change the build. See [`../CREDITS.md`](../CREDITS.md).

---

## Single-stream (short context, warm)

| metric | NVFP4 (this repo) |
| --- | ---: |
| single-stream, code prompt | **58–63** tok/s |
| single-stream, general prompt | **44.4** tok/s |
| **no-spec autoregressive** | **26.6** tok/s |

**No-spec control is the headline:** NVFP4 no-spec = **26.6** tok/s vs our fp8 build's
**26.7** tok/s — **identical**. The smaller NVFP4 KV cache buys **nothing** on the raw forward
pass. This proves the single-stream win is **acceptance, not bandwidth** (the entire advantage
over fp8 comes from longer accepted DSpark draft runs, not faster KV reads).

---

## 1M-context single-stream depth curve (warm, non-repetitive continuation prompt)

The number nobody published. The community headline (~67 tok/s) is measured near the top of the
window (~300 tokens); spec-decode throughput is `mean_accept / step_time`, and acceptance falls
as context deepens, so the real curve decays hard.

| context depth | single-stream tok/s | mean draft-accept | per-position acceptance (pos 0..4) |
| ---: | ---: | ---: | --- |
| 16K  | **42.7** | 3.30 | 0.80 / 0.56 / 0.46 / 0.30 / 0.18 |
| 64K  | **28.9** | 2.23 | 0.62 / 0.34 / 0.17 / 0.07 / 0.03 |
| 256K | **25.4** | 2.42 | 0.70 / 0.44 / 0.16 / 0.07 / 0.05 |
| 512K | **17.6** | 1.98 | 0.54 / 0.27 / 0.10 / 0.04 / 0.02 |
| **1.03M** | **17.2** | ~2.1 | (short window; steady-interval pos0 ≈ 0.68–0.72) |

Throughput more than halves from 16K → 512K, tracking the collapse in draft acceptance: pos-0
acceptance falls from **0.80 → 0.54** over 16K → 512K, and the deeper draft positions go to
near-zero. The short-context number does **not** describe deep-context serving. This curve is
true of the *unmodified upstream build*.

**Real-1M point (added 2026-07-04).** Prompt calibrated to **1,032,953 tokens** (of the
1,048,576 window), greedy, 200 generated tokens per run, two timed runs: **17.21 / 17.19 tok/s**.
The curve **flattens** past 512K (17.6 → 17.2, −2%): by this depth speculation adds little
(mean accept ~2.1) and the step is dominated by the same verify-forward as no-spec. Both runs
were **coherent** (repetition ratio 0.037 / 0.0 — no loops at full depth; see the
coherence-collapse note below: the loop is a lottery, not a depth wall). Operational note:
cold prefill of 1.03M tokens took **642 s** (~10.7 min, ~1600 tok/s prefill); a warm re-query on
the same context via prefix cache started in **4 s**. Method caveat: the 1M point uses a
different fragment of the same prose corpus than the shallower points, single config (greedy).

---

## Negative results

**(1) Draft-RoPE re-anchor (local window) — NET LOSS.** The DSpark draft sees only a 128-token
sliding window of main-KV but applies full-1M YaRN RoPE; we re-based the draft's RoPE to a
local window to fix the window-vs-YaRN mismatch. The apparent acceptance lift (256K mean-accept
2.42 → 4.00) was a **mirage** — the draft was trivially predicting its own repetition loop. The
`torch.where` re-anchor path also slowed decode catastrophically (256K: 25.4 → 1.73 tok/s, ~15×)
and produced incoherent output at every depth. Abandoned.

**(2) Dormant confidence-scheduler ON — REGRESSES.** Enabling
`VLLM_DSPARK_CONFIDENCE_SCHEDULER` (threshold 0.6) dropped C16-static from **319** (OFF, shipped
default) to **290** (ON). tonyd2wild shipping it OFF is correct; we confirmed it.

---

## Coherence-collapse note

At **≥256K with deterministic decode**, the build intermittently degenerates into **repetition
loops**, and it is **flappy** — a lever-OFF control also looped at the same depth, so the loop
is **intrinsic to the NVFP4 build + deterministic-continuation decode path**, not patch-caused.
The acceptance metric is a trap here: it *rises* during a loop because the draft trivially
predicts the repetition. Full characterization and transcripts:
[`../docs/LONG-CONTEXT-FINDINGS.md`](../docs/LONG-CONTEXT-FINDINGS.md).
