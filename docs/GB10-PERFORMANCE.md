# DSpark on DGX Spark (GB10) — Performance & The "+85%" Gap, Honestly Explained

> The first public fp8 DSpark port on GB10 we are aware of — DeepSeek's **DSpark**
> speculative-decoding module wired into vLLM on the **GB10 / sm_121** (DGX Spark, consumer
> Grace-Blackwell). This document explains the
> real-world performance we measured, and — more importantly — *why* the headline
> "+85%" number from DeepSeek's datacenter benchmarks does **not** translate 1:1 to a
> 2-node GB10 cluster. If you run LLMs on DGX Spark, the takeaways here matter.

Status: **DSpark runs in production** on 2× DGX Spark (TP=2 over QSFP 200G RoCEv2),
DeepSeek-V4-Flash-DSpark, fp8 weights + FP4 (MXFP4) experts, block_size=5, 1M context.

---

## TL;DR

- **Measured single-stream speedup on GB10: `+21–24%` over no-spec autoregressive**
  (26.7 tok/s no-spec → 32.4 tok/s DSpark, apples-to-apples). This is **modest, and honest** —
  it does **not** reproduce DeepSeek's "+85%". DeepSeek measures vs **MTP-1** (a weaker
  baseline than no-spec) on **H100/H800-class** hardware; on GB10 two things compress the
  win: (a) the decode step is **compute/bandwidth-bound**, and (b) our 3-stage fp8/fp4
  draft is heavy (~15% of the step) vs their hand-tuned tilelang.
- **Absolute single-stream tok/s is gated by raw forward compute**, not interconnect. GB10's
  **273 GB/s** unified LPDDR5X is **~12× less** than H100's 3.35 TB/s HBM3. A full 43-layer
  fp8-MLA + 256-expert-MoE verify forward dominates the step **regardless of draft block
  size** — measured: `block_size 5 → 4` changed throughput by ~0 (32.4 → 31.4 tok/s), proving
  the step is the *fixed* forward, not the token count.
- **DSpark's real, defensible win on GB10 is concurrency:** **+55%** aggregate over our
  previous MTP-2 production at 8 concurrent users (real-world ~99 vs ~64 tok/s), measured at
  the tuned prod config `--max-num-seqs 12` (sustains **141–146 tok/s** at 12 concurrent).
- **Single-stream is gated by the fixed forward, not the draft.** `block_size 5 → 4`
  changed throughput by ~0 (32.4 → 31.4 tok/s), so there is no block-size lever to push past
  it. 40+ tok/s single-stream is **not reachable** on this hardware without NVLink-class step
  times or a sub-4-bit checkpoint (neither exists for this model). For one interactive
  session, a lighter MTP draft is faster.

---

## Why DeepSeek's +85% ≠ our number — the four contributors

| # | Cause | Est. share of the gap | Evidence |
|---|---|---:|---|
| **A** | **Baseline mismatch.** DeepSeek reports "+60–85% per-user speed at matched throughput **over MTP-1**". We were comparing DSpark to our **MTP-2** production (mean accept ~2.5), a *much* stronger reference than MTP-1. Most of the headline % evaporates by construction. | **~40%** | DeepSeek/MarkTechPost/alphaXiv: "over MTP-1, not plain autoregressive" |
| **B** | **Hardware = compute/bandwidth, not interconnect.** GB10 = 273 GB/s unified LPDDR5X (~12× less BW than H100 HBM3). The verify+draft forward is **~65 ms/step, compute-bound**. The 2-node RoCE TP all-reduce is a **red herring** (~4 MiB/step, <3.5% of the step time — measured, not assumed). DeepSeek benched on H100/H800 → far cheaper step → higher absolute tok/s *and* a bigger % over their weak baseline. | **~30%** | live profile: all-reduce <2.3 ms of a 65 ms step; GB10 273 GB/s vs H100 3.35 TB/s |
| **C** | **Acceptance deficit.** Our per-position acceptance (pos0 ~0.62–0.74, mean ~2.2–2.5) trails DeepSeek's implied ~2.8–3.0. Likely fp8-weights + FP4-experts on sm_121 (NVFP4 is partially broken on sm_121) vs their bf16/tilelang reference; plus a YaRN-on-draft divergence on long context. | **~20%** | live pos0 0.62–0.74 vs reference ~0.76 |
| **D** | **Draft efficiency.** Our vLLM fp8-MLA + 256-expert-MoE draft vs DeepSeek's hand-tuned tilelang kernels. But the draft is only **~15% of the step** — even a *free* draft caps single-stream at ~37 tok/s. | **~10%** | profile: draft ~10–12 ms of a ~65 ms step |

**One line for the article:** *On DGX Spark the single-stream bottleneck is the raw
compute of the forward pass (273 GB/s unified memory, full 43-layer fp8-MLA + MoE-256
per ~65 ms verify step) — not the algorithm and not the 200G RoCE interconnect. So
speculative-decoding's percentage win is structurally smaller than on an H100/NVLink
node, even with an identical, correct DSpark implementation.*

---

## The spec-decode throughput model (so you can reason about your own setup)

```
tps = mean_acceptance_length / step_time
step_time = target_verify_forward + (non-overlapped) draft + overhead
```

Live GB10 numbers (this build, idle engine):
- step_time ≈ **65 ms**  (target verify ~85% · draft ~15%)
- mean_acceptance ≈ **2.2–2.5**  → **32.4 tok/s** single-stream measured
  (`completion_tokens / wall_time`, non-streaming)
- the all-reduce series (43×2 collectives/token over RoCE) ≈ **1.5 ms = ~2.3%** — *not*
  the cap. The cap is GB10 memory bandwidth doing the forward.

To reach **40 tok/s** you would need mean_acceptance ≈ **2.6 at the current step** *or* a
step −17%. Acceptance ≥ reference (pos0 ≥0.76 sustained) is only seen on easy/code content,
and the step is a *fixed* forward (block size is not a lever — see below). **40+ tok/s
single-stream is not reachable on this hardware**; a lighter MTP draft is faster for a
single interactive session.

---

## What actually wins on GB10: concurrency

DSpark's 6-wide verify (block5+1) amortizes across users. The tuned production config runs
`--max-num-seqs 12` (raised from 4 — see the gotcha below for why the old "don't go above 4"
warning was a red herring). With more in-flight slots the verify batches and the higher
acceptance pays off. Aggregate throughput (`completion_tokens / wall_time`, non-streaming):

| load | DSpark seqs=4 (old) | **DSpark seqs=12 (prod)** |
|---|---|---|
| concurrency c8  | 69 tok/s | **99 tok/s** (**+44%**) |
| concurrency c12 | — | **141–146 tok/s** (sustained 3×) |
| concurrency c16 | 75 tok/s | **117 tok/s** (**+56%**) |

vs the previous **MTP-2** production (real ~64 tok/s at 8 streams), DSpark seqs=12 at c8
≈ **99 tok/s = +55%** (it was +27% at seqs=4). Health under load: acceptance **2.3–2.56**,
**0 crashes**, no OOM, `RestartCount 0`.

| | DSpark (seqs=12) | prev prod (MTP-2) |
|---|---|---|
| concurrency c8 (real, measured) | **~99 tok/s** | ~64 tok/s (**+55%**) |
| single-stream | 32.4 tok/s | faster (lighter draft) |
| acceptance pos0 | 0.62–0.74 (≈ reference) | — |
| context | 1M (YaRN) | 1M |

> The ~64 tok/s MTP-2 figure is the **real** 8-stream throughput a user measured in
> production, not a synthetic single-config bench. DSpark's ~99 tok/s is the same kind of
> measurement — apples-to-apples, this is the win that holds up.

**Recommendation:** on GB10, run DSpark when you serve **multiple concurrent users or
long-context RAG**, with `--max-num-seqs 12` as the tuned prod default; a lighter MTP draft
is faster for a single interactive session.

---

## GB10 / sm_121 gotchas we hit (so you don't have to)

- **NVFP4 MMA only on sm_121a**, FlashInfer FP8 attention broken on sm_121 → use
  `VLLM_ATTENTION_BACKEND=TRITON_ATTN`; MoE via `VLLM_USE_B12X_MOE=1`.
- **`decode_dsv4` sparse-MLA wants page_block_size=64** → keep the hybrid KV cache on
  (`--block-size 256` with hybrid balancing keeps SWA native 64); do **not**
  `--disable-hybrid-kv-cache`.
- **Spec-decode + 256-expert sparse MLA across 2 nodes**: the C128A and SWA metadata
  builders must agree on the decode/prefill split threshold (`1 + num_speculative_tokens`),
  otherwise a chunked-prefill chunk of 7–11 tokens hits
  `extra_kv_cache requires extra_indices` under concurrency. (parallel_drafting doubles
  one builder's threshold — align it.)
- **Any model-code change triggers an inductor recompile** that exceeds the NCCL
  heartbeat (~8 min) during load → re-init loop. Set
  `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800` for the first (recompiling) load.
- **RoCE rendezvous recovery:** after a churn of restarts the RDMA state goes stale even
  though the GID index is present. If rank-1 logs `Broken pipe` on the TCPStore / the
  head loops at NCCL init, **force-reset** `nmcli con down qsfp-link && nmcli con up
  qsfp-link` on **both** nodes, then restart **master-first** (rank 0 hosts the
  rendezvous on the RDV port; the worker must start only *after* the port is LISTENing).
- **`--max-num-seqs 12` is the tuned prod config — *not* 4.** An earlier warning said "do
  not raise `--max-num-seqs` above 4, `seqs=8` OOMs / hangs." That was **stale and wrong
  about the cause.** The hang at higher seqs was the **RoCE rendezvous bug above** (stale
  RDMA / drifted GID) surfacing under more in-flight slots — *not* a memory limit. With the
  RoCE-GID force-reset + master-first restart fix in place, `seqs=8` and `seqs=12` load like
  any normal restart (config-only, **no recompile**): acceptance 2.3–2.56, 0 crashes, no
  OOM, `RestartCount 0`. seqs=12 sustains 141–146 tok/s at 12 concurrent streams. (Don't push
  arbitrarily higher without re-measuring — the model is 156 GB and the draft context-KV +
  SWA cache do grow with concurrency; 12 is the verified sweet spot, not a hard ceiling.)

---

## Final measured numbers

All single-stream figures are **apples-to-apples**: `completion_tokens / wall_time`,
non-streaming, same prompt class, same engine, same 2-node TP=2 deploy. (SSE-event
counting undercounts spec-decode by ~2.5× — do not use it.)

| Measurement | Result | Notes |
|---|---|---:|
| **No-spec autoregressive** (V4-Flash, spec off) | **26.7 tok/s** | the honest single-stream baseline |
| **DSpark** (block_size=5, prod config) | **32.4 tok/s** | **+21–24% over no-spec** |
| **DSpark** block_size 5 → 4 | **31.4 tok/s** | **no gain** — the step is the fixed forward |
| **DSpark concurrency** (seqs=12), 8 streams | **~99 tok/s** | vs ~64 tok/s prior MTP-2 prod = **+55%** |
| **DSpark concurrency** (seqs=12), 12 streams | **141–146 tok/s** | sustained 3× |
| **DSpark concurrency** (seqs=12), 16 streams | **117 tok/s** | vs 75 tok/s at seqs=4 = **+56%** |
| Acceptance, pos0 | 0.62–0.74 | ≈ reference ~0.76 |
| Acceptance, mean length | ~2.2–2.56 | tokens accepted per step (under c8–c16 load) |

**The block_size 5→4 result is the load-bearing one.** Dropping the draft from 5 to 4
tokens did *not* speed up single-stream decode (32.4 → 31.4 tok/s, i.e. flat / slightly
worse). If the decode step were dominated by draft work, fewer draft tokens would have
helped. It didn't — which proves the step time is the **fixed 43-layer fp8-MLA + MoE-256
verify forward**, gated by GB10's 273 GB/s unified memory, *not* the speculative token
count. There is no "increase block size to go faster" lever here.

**Honest bottom line:** single-stream is a **modest +21–24% over no-spec** (not DeepSeek's
"+85%" — see the gap analysis above; their baseline is the weaker MTP-1, on much
higher-bandwidth hardware). DSpark's defensible win on GB10 is **concurrency (+55% over
MTP-2 at 8 streams, seqs=12)** and **long-context (1M)**. For a single interactive chat
session, a lighter MTP draft decodes faster.

---

## fp8 vs NVFP4 KV on GB10 — the honest comparison (NVFP4 is faster)

This repo is our **fp8-KV** DSpark build. There is a separate, **community-derived NVFP4-KV**
build that is faster single-stream, and we measured both apples-to-apples on the same 2-node
hardware. The honest result: **NVFP4 KV is the faster path.**

| measurement | fp8 (this repo) | NVFP4 (community-derived) |
|---|---:|---:|
| **no-spec autoregressive** | 26.7 tok/s | 26.6 tok/s |
| **single-stream, DSpark, short ctx** | 32.4 tok/s | 55–63 tok/s |
| **mean draft-accept length** | ~2.1 | 3.0–4.2 |
| **C16-static concurrency (200K)** | — (not built for this profile) | ~315–324 tok/s |

**The mechanism is the interesting part, and it is *not* bandwidth.** No-spec decode is
**identical** across the two (26.7 vs 26.6 tok/s) — the smaller NVFP4 KV cache buys nothing on the
raw forward at single-stream. NVFP4's entire single-stream advantage is **higher DSpark draft
acceptance** (~3–4 vs our fp8 ~2), which directly raises `tps = mean_accept / step_time`. This is
the same `step_time`-is-the-fixed-forward picture this whole document describes — NVFP4 just gets
a longer accepted run per step.

**Why keep the fp8 repo honest and separate:** the fp8 build was the **first public DSpark
port to GB10 / sm_121 we are aware of** and carries the full bug-fix journey and the "+85% gap" analysis above,
which stand on their own. But if you are choosing what to *deploy* on 2× DGX Spark today and want
maximum single-stream speed, **use the NVFP4-KV build.** It is not our recipe — it is the
community's (tonyd2wild / drowzeys / bjk110 / rafaelcaricio / fraserprice / MiaAI-Lab), which we
independently reproduced and measured.

→ Our full NVFP4 reproduction — the **1M-context single-stream curve**, the acceptance-not-bandwidth
mechanism, the long-context coherence findings, and the negative results — lives in this repo under
[`benchmarks/`](../benchmarks/) and [`docs/LONG-CONTEXT-FINDINGS.md`](LONG-CONTEXT-FINDINGS.md). The
NVFP4-KV **recipe** itself is tonyd2wild's:
[DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
(reproduced and measured here, not authored by us — see [CREDITS.md](../CREDITS.md)).

_Last updated: 2026-06-30._
