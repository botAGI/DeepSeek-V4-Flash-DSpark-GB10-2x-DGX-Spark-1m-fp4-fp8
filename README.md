# DSpark on DGX Spark (GB10 / sm_121)

Running DeepSeek's **DSpark** speculative decoding for `DeepSeek-V4-Flash-DSpark` on a 2× DGX
Spark cluster (GB10, consumer Grace-Blackwell, `sm_121`), TP=2 over 200G RoCEv2, at 1M-token
context — plus an **honest, cross-build benchmark suite**. This repo carries our own fp8 DSpark
GB10 bring-up (patches + launcher), and it also documents our independent **reproduction and
measurement** of the community NVFP4-KV build. We author the fp8 recipe; we do **not** author the
NVFP4 recipe — that one is tonyd2wild's and is referenced by link, not vendored here. Our unique
contribution is the **extended benchmark data**: the 1M-context depth curve, the concurrency
scaling, and a long-context coherence / sampling study.

No marketing. All throughput is apples-to-apples (`completion_tokens / wall_time`, non-streaming),
and every gap to datacenter numbers is explained rather than hidden.

---

## Two recipes on GB10

### 1. fp8 DSpark — **ours**

Our GB10 / `sm_121` bring-up of an fp8 DSpark KV path (`fp8_ds_mla`): the DSpark draft head and
proposer wired into vLLM's spec-decode loop, plus the numerical bug-fixes that took the draft from
"predicts noise" to "matches the reference." Implemented as vLLM **patches** (a new `dspark`
speculative method) — it does **not** fork the base model, and the running prod image is never
touched.

- Apply the patches: [`apply-patches.sh`](apply-patches.sh) → everything under
  [`patches/`](patches/) (the `DSparkV4MTP` draft head, the `DSparkProposer`, the
  spec-config wiring, the EAGLE3 aux plumbing).
- Launch (TP=2 over RoCE, 1M context): [`run-dspark-node.sh`](run-dspark-node.sh)
  (systemd-managed, reboot-proof, RoCE-GID re-alignment) and
  [`launch-dspark-node.sh`](launch-dspark-node.sh).
- Full recipe, anchors, and the three bug-fixes: [`docs/recipe.md`](docs/recipe.md),
  [`patches/PATCHES.md`](patches/PATCHES.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### 2. NVFP4-KV DSpark — **tonyd2wild's** (we reproduced + measured, did not author)

The faster single-stream / higher-acceptance line uses NVFP4 KV. **That recipe is not ours.** It
is tonyd2wild's:
<https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark> (commit
`89bb82b`). We cloned it, reproduced it byte-identically on our own two-node cluster, and measured
what nobody had published. To **build** it, go to the upstream repo. In this repo the NVFP4 line is
**referenced by link, not vendored**; what we ship is our launcher for the reproduction run
([`run-nvfp4-node.sh`](run-nvfp4-node.sh)) and our measurements (below). Credit chain in
[`CREDITS.md`](CREDITS.md).

---

## Benchmarks (our unique contribution)

The measurement suite is the point of this repo. Everything here is our data on 2× DGX Spark GB10,
TP=2 over RoCE, apples-to-apples.

- **[`benchmarks/20260630-fp8-dspark-checkpoint.md`](benchmarks/20260630-fp8-dspark-checkpoint.md)**
  — our fp8 DSpark build: single-stream +21–24% over no-spec, the `block_size 5→4` control, and
  the concurrency numbers (seqs=12 prod: ~99 t/s at 8 streams, 141–146 t/s at 12).
- **[`benchmarks/20260630-nvfp4-1m-context-curve-checkpoint.md`](benchmarks/20260630-nvfp4-1m-context-curve-checkpoint.md)**
  — the **1M-context single-stream depth curve** of tonyd2wild's NVFP4 build (42.7 t/s at 16K down
  to 17.6 t/s at 512K) — the number nobody in the community had published.
- **[`benchmarks/20260630-nvfp4-c16-reproduction-checkpoint.md`](benchmarks/20260630-nvfp4-c16-reproduction-checkpoint.md)**
  — independent reproduction of tonyd2wild's C16 concurrency numbers (his 315.1 re-derives as our
  control's 319), validating his benchmark.
- **[`benchmarks/20260701-sampling-coherence-probe.md`](benchmarks/20260701-sampling-coherence-probe.md)**
  — a long-context sampling / coherence probe on the NVFP4 build: an **honest negative result** (no
  sampler config is a reliable cross-depth coherence fix), with two verifiable takeaways — the
  collapse is not a fixed depth threshold, and `min_p` is blocked under spec-decode.
- **[`docs/LONG-CONTEXT-FINDINGS.md`](docs/LONG-CONTEXT-FINDINGS.md)** — the detailed analysis
  behind the NVFP4 measurements: the per-position acceptance decay driving the depth curve, the
  acceptance-not-bandwidth mechanism, the flappy coherence-collapse characterization, and two
  documented negative results.

---

## Honest stance

- The **NVFP4 numbers are measurements of tonyd2wild's build, not our recipe.** We reproduced and
  measured it; we did not author it and claim **no speed win over it** (our C16 reproduction ties
  his within his own run-to-run variance).
- Our own recipe is the **fp8 DSpark** line (patches + launcher above).
- Our unique data is the **1M-context depth curve**, the **concurrency scaling**, and the
  **coherence / sampling study**.
- No "world-first," no "we beat," no "SOTA." Where DSpark on GB10 is a modest single-stream gain,
  we say so; where it wins (concurrency, long context), we show the numbers.

---

## Credits & license

Full attribution — our fp8 contribution, the lineage we build on, and the NVFP4 line we reproduced
but did not author — is in **[`CREDITS.md`](CREDITS.md)**. This repo is released under the **MIT
License** (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). **No model weights are redistributed by
this repository.**
