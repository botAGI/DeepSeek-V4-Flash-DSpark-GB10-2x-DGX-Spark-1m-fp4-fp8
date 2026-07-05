# Long-Context Findings — NVFP4-KV DSpark on 2× DGX Spark (GB10)

This is the detailed writeup behind the [README](../README.md). It documents what we measured
that the upstream community ([attribution](../README.md#attribution--this-stands-entirely-on-community-work))
did not publish: the single-stream throughput curve out toward 1M context, the per-position
draft-acceptance decay that drives it, the mechanism proving the NVFP4 single-stream win is
acceptance and not bandwidth, the flappy long-context coherence bug, and two negative results.

Everything here is a reproduction-and-measurement contribution. The build, the recipe, the
patches, and the `nvfp4_ds_mla` plumbing are upstream work (tonyd2wild / bjk110 / drowzeys /
rafaelcaricio / fraserprice / MiaAI-Lab) — see the [README attribution](../README.md#attribution--this-stands-entirely-on-community-work).

---

## Test setup

- **Hardware:** 2× DGX Spark (GB10, one GPU per node), TP=2 over QSFP 200G RoCEv2, driver 580.x.
- **Model / build:** `DeepSeek-V4-Flash-DSpark`, byte-identical upstream Stage-C NVFP4 image
  (`nvfp4_ds_mla` KV, 584-byte padded sparse-MLA envelope), MTP5 DSpark speculative decoding,
  `max_model_len=1048576`, `gpu_memory_utilization=0.80`.
- **Measurement convention:** single-stream throughput is **`completion_tokens / wall_time`,
  non-streaming**. SSE-event counting undercounts spec-decode by ~2.5× and is never used.
  Acceptance figures are read from vLLM's own `SpecDecoding metrics` log line
  (mean acceptance length + per-position acceptance rate) over the timed generation window.
- **Prompt class:** a non-repetitive technical-continuation prompt padded to each target depth,
  then `g=200` (or to `stop`) generated tokens, warm engine (≥12-request warmup including deep
  contexts before timing).

---

## 1. The single-stream throughput-vs-depth curve

Throughput for a spec-decode engine is:

```
tps = mean_acceptance_length / step_time
```

`step_time` is dominated by the fixed target-verify forward (43-layer sparse-MLA + 256-expert
MoE on GB10's 273 GB/s unified memory) plus the draft. As context deepens, `mean_acceptance_length`
falls because the draft has progressively less signal — so `tps` decays. Measured:

| depth | prompt_tok (verified) | comp_tok | wall_s | **tok/s** | mean-accept | avg draft-accept rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16K  | 16,006  | 200 | 4.69 | **42.66** | 3.30 | 45.9% |
| 64K  | 64,008  | 200 | 6.92 | **28.90** | 2.23 | 24.7% |
| 256K | 255,991 | 104 | 4.10 | **25.35** | 2.42 | 28.4% |
| 512K | 511,966 | 95  | 5.40 | **17.61** | 1.98 | 19.6% |
| 1.03M | 1,032,953 | 200 | 11.62 | **17.21** | ~2.1 | ~22% |

Throughput drops from **42.7 tok/s at 16K to 17.6 at 512K and 17.2 at the full window (1.03M)**
— more than halved by 512K, then **flattening** (−2% from 512K to 1.03M: by that depth speculation
adds little and the step is dominated by the same verify-forward as no-spec). The
non-monotonicity between 64K (28.9) and 256K (25.4) is small and within sample noise; the trend
is unambiguous: **deep context is slow, and the short-context number does not describe it.**

> **Update 2026-07-04 — full-window point added.** Prompt calibrated to **1,032,953 tokens** of
> the 1,048,576 window; two timed greedy runs: **17.21 / 17.19 tok/s** (spread 0.02), both
> **coherent** (repetition ratio 0.037 / 0.0 — no loops at full depth). Cold prefill of the 1.03M
> prompt took **642 s** (~1600 tok/s prefill); a warm re-query on the same context via prefix
> cache started in **4 s**. Method caveat: different fragment of the same prose corpus than the
> shallower points; acceptance is a full-window aggregate over a short generation window. Raw
> logs: [`../benchmarks/raw/`](../benchmarks/raw/).

> **Why this matters for the community:** the published single-stream headline for this build
> (~67 tok/s) is measured near the top of the window (~300 tokens). Forum user *renek* noted
> qualitatively that "at 1M nobody is fast, <10 tok/s," but no one published the curve. This table
> fills that gap. It is true of the *unmodified upstream build* — it is not a property of any
> change we made.

### Per-position acceptance decay (the driver)

DSpark MTP5 drafts 5 tokens per step; each position is independently accepted or rejected. The
per-position acceptance rates (pos 0 → pos 4) show *where* the decay happens:

| depth | pos 0 | pos 1 | pos 2 | pos 3 | pos 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16K  | 0.803 | 0.557 | 0.459 | 0.295 | 0.180 |
| 64K  | 0.622 | 0.344 | 0.167 | 0.067 | 0.033 |
| 256K | 0.698 | 0.442 | 0.163 | 0.070 | 0.047 |
| 512K | 0.542 | 0.271 | 0.104 | 0.042 | 0.021 |

Two effects compound at depth: **pos-0 acceptance falls** (0.80 → 0.54), *and* the **deeper draft
positions collapse toward zero** faster. By 512K the draft contributes barely more than one bonus
token on average (mean-accept 1.98). The speculative speedup is structurally eroding as context
grows — this is the root cause of the throughput curve in §1.

---

## 2. Mechanism — acceptance, not bandwidth (the no-spec control)

NVFP4 KV is smaller than fp8 KV, so an obvious hypothesis is "NVFP4 is faster because it reads
less KV memory per step." We **falsified** this with a no-spec control: run the same engine with
DSpark speculative decoding **disabled**, so throughput reflects only the raw autoregressive
forward, with no draft acceptance in play.

| configuration | no-spec single-stream tok/s |
| --- | ---: |
| NVFP4-KV build (this repo) | **26.6** |
| fp8-KV build (our other repo) | **26.7** |

**Identical.** The raw forward pass decodes at the same speed under NVFP4 and fp8 KV — there is
**no bandwidth win** from the smaller KV cache at single-stream. (At GB10's 273 GB/s the
single-stream forward is compute/bandwidth-bound on the *weights and activations*, and the KV read
is not the binding term at these depths.)

Since `tps = mean_accept / step_time` and `step_time` is the same (no-spec proves it), the entire
NVFP4 single-stream advantage with spec-decode ON is the **higher mean acceptance length**:

| build | mean-accept (spec ON) | single-stream tok/s (spec ON, short ctx) |
| --- | ---: | ---: |
| NVFP4 (this repo) | **3.0–4.2** | 55–63 |
| fp8 (our other repo) | ~2.1 | 32–42 |

**Conclusion:** NVFP4 KV's single-stream win comes from the optimized base image + DSpark opts +
NVFP4 KV producing **longer accepted draft runs**, not from faster memory access. The community
presented NVFP4 as "faster" without isolating this; the no-spec control is the missing experiment.

---

## 3. Flappy long-context coherence (intrinsic to the build, characterized)

The upstream README documents "gibberish, loops, Chinese drift, prompt/XML leakage" at long
context and frames it as an orchestration / sampling problem (bounded bootstrap, repetition
penalty, deterministic decode). We found that at depth it is **deeper than orchestration** — the
*model itself*, under greedy/deterministic continuation, intermittently degenerates into
**repetition loops**, and the upstream sampling guidance only partly masks it.

### What we observed

At **≥256K context** with deterministic continuation, generations sometimes collapse into a
repeating phrase. A representative 256K control sample (real transcript):

```
'achieves 0.94 acceptance rate at 4.2x speedup over a single A100-80GB baseline, with 0.94
 acceptance rate at 4.2x speedup over a single A100-80GB baseline. The system achieves 0.94
 acceptance rate at 4.2x speedup over a single A100-80GB baseline, with 0.94 acceptance rate
 at 4.2x ...'
```

The "high acceptance" reported during such a loop is **artifactual**: the draft trivially predicts
the next token of its own repetition, so mean-accept *rises* (e.g. 3.64 at 256K) while the output
is garbage. **High acceptance during a loop is a symptom, not a win** — a trap for anyone tuning to
the acceptance metric alone.

### It is flappy (sample-dependent), not deterministic

The *same prompt at the same depth* is sometimes coherent and sometimes loops. Our baseline 256K
run produced coherent output (25.4 tok/s, mean-accept 2.42); a later control run at 256K, same image,
produced the loop above (mean-accept 3.64). It is a **lottery over the decode path**, gated by depth
(≥256K) and decode determinism.

### It is intrinsic to the build — not caused by any single patch

The decisive control: we took an experimental draft-RoPE modification (see negative result (a)
below), **turned it fully off** via its env gate, and ran the same image. **It looped at 256K
anyway.** Since the modification was disabled, the loop cannot be attributed to it — it is a
property of the **unmodified NVFP4 build + deterministic-continuation decode path**, present
regardless of our experiments. The baseline's *coherent* 256K sample was simply a luckier draw of
the same lottery.

### Why this is a contribution

This is the same "gibberish at 1M" the forum complains about, but until now it was anecdote. We
scope it: **depth-gated (≥256K), flappy (sample-dependent), decode-path-sensitive, and present in
the unmodified build**, with the acceptance metric actively misleading during a loop. That is a
reproducible, well-defined bug other contributors can collaborate on — rather than "sometimes it
breaks."

---

## 4. Negative results

Both of these attacked a real, plausible blind spot. Both lost. Documented in enough detail that
others don't burn the same cycles.

### (a) Draft YaRN re-anchor — hypothesis plausible, implementation a net loss

**Hypothesis.** The DSpark draft reads only the last **128 tokens** of main-KV (a sliding ring,
`pos % 128`) but applies the **full-1M YaRN RoPE** scaling. A draft that only sees a 128-token
window but rotates positions as if they were spread across 1M is RoPE-extrapolating — which should
degrade its predictions and collapse acceptance at depth (the same intuition as OWL,
arXiv 2510.07535). Fix: re-base the draft's RoPE to a **local 0..132 window** so the draft sees
locally-consistent rotations, while the target keeps true long-context positions.

**What happened.**

- The re-anchor *did* raise reported draft acceptance at depth (256K mean-accept 2.42 → **4.00**;
  64K 2.23 → 3.70). This looked like the lever working.
- **It was a mirage.** The "lift" came from the draft predicting a **repetition loop** — the
  acceptance metric rises during degeneration (see §3). Output was incoherent at every depth
  (`"...running on a single node with 0 GPUs."`, phrase loops at 64K/256K/512K).
- The `torch.where` re-anchor path also **slowed decode catastrophically**: 256K dropped from
  25.4 tok/s (baseline) to **1.73 tok/s** (~15× slower); 16K regressed 42.7 → 18.6 tok/s even with the
  re-anchor depth-gated off. The extra per-step tensor op is not free on GB10.

We applied the obvious fixes (identity below the window threshold, depth-gate the re-anchor to
`pos > 65536`, longer warmup including 256K/512K) and validated the modular arithmetic was correct
(no small-position fold; deep positions stay congruent mod 128 with adjacent diff = 1). The fixed
logic was still a **net loss at every depth** — slower *and* incoherent. **Lever abandoned.**

**Takeaway for others:** the window/RoPE-extrapolation hypothesis may still be directionally real,
but (1) you cannot validate it by acceptance alone — acceptance is corrupted by the loop, and
(2) any per-step `torch.where`/re-rotation on GB10 has to pay for itself against a forward step
that is already the bottleneck; it didn't.

### (b) Confidence scheduler — regresses aggregate when enabled

The build ships a dormant `VLLM_DSPARK_CONFIDENCE_SCHEDULER` (default `VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0`,
i.e. off). Enabling it to gate draft acceptance on a confidence threshold was an obvious
"free env-only win."

| confidence scheduler | C16-static aggregate |
| --- | ---: |
| **OFF** (shipped default) | **319** |
| ON (threshold 0.6) | 290 |

Enabling it **lost ~9%** aggregate. The OFF=319 result also reproduces tonyd2wild's published
315.1 — confirming both that **his benchmark is sound** and that **his decision to ship the
scheduler OFF is correct.** We verified it so you don't have to.

---

## Reproducing these measurements

- **Build:** the upstream recipe — [tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
  (commit `89bb82b`): Stage A/B/C Dockerfiles, two-node worker-first launch, model-cache prep.
  Reproduced unchanged from upstream.
- **Depth curve:** pad a non-repetitive prompt to each target depth (16K/64K/256K/512K/1.03M), generate
  `g=200` non-streaming, record `completion_tokens / wall_time`, and capture vLLM's
  `SpecDecoding metrics` log line over the timed window. Warm the engine with ≥12 requests
  including deep contexts before timing (cold-cache inflates the first deep run).
- **No-spec control:** relaunch with DSpark speculative decoding disabled; measure single-stream
  autoregressive tok/s. Compare NVFP4 vs fp8 KV (should be ≈ equal — that is the point).
- **Coherence check:** at ≥256K, deterministic decode, inspect generated text for phrase
  repetition; cross-check that high mean-accept coincides with loops (the metric trap). Repeat the
  same prompt several times — coherence is flappy, so a single coherent sample does not clear the
  build.

_Last updated: 2026-06-30._
</content>
