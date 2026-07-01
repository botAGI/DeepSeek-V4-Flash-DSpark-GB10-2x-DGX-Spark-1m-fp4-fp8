# Benchmark checkpoint — long-context sampling / coherence probe (2026-07-01)

> Measurements of tonyd2wild's NVFP4-KV build (https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark, commit 89bb82b) on our 2x DGX Spark. Reproduced and measured by us; recipe authored by tonyd2wild -- see CREDITS.md.

Live `nvfp4_ds_mla` build, DSpark speculative decoding, 2× DGX Spark (GB10), TP=2 over QSFP 200G
RoCEv2. Endpoint model id `dspark`, inference-only (nothing about the build was changed).

This is a **follow-up to the coherence-collapse finding** in
[`docs/LONG-CONTEXT-FINDINGS.md`](../docs/LONG-CONTEXT-FINDINGS.md): that document showed the
build **intermittently degenerates into repetition loops at deep context** under greedy /
deterministic continuation, and that the collapse is flappy (sample-dependent) rather than a
clean deterministic failure. The obvious next question a serving operator asks is: **can a
sampler config make it reliably coherent at depth?** This probe answers that — the honest answer
is **no**.

---

## Goal

Test whether any practical sampling configuration turns the long-context loop into reliably
coherent generation across depth. Specifically: does raising the depth to ≥256K deterministically
trigger the collapse (a fixed threshold), and is there a single anti-loop knob that generalizes?

## Method

- **Prompt:** a non-repetitive Project-Gutenberg continuation prompt padded to depth. The 256K run
  reached **~255,284 prompt tokens** (measured `usage.prompt_tokens`); a **512K** run
  (~511,770 prompt tokens) was also executed.
- **Generation:** `max_tokens=512` per cell, non-streaming completions endpoint.
- **Configs (5):**
  - **A greedy** — `temperature=0`.
  - **B rep_pen** — `temperature=0`, `repetition_penalty=1.15`.
  - **C freq_pres** — `temperature=0`, `frequency_penalty=0.4`, `presence_penalty=0.4`.
  - **D min_p** — intended `temperature=0.7`, `min_p=0.05`, `seed=1234`. **`min_p` was rejected
    by the server with HTTP 400** ("not yet supported with speculative decoding") — see the
    community note below. The recorded D cell therefore used the supported tail knobs that stand
    in for `min_p` (`temperature=0.7`, `top_p=0.95`, `top_k=50`, `seed=1234`).
  - **E combo** — `temperature=0.6`, `repetition_penalty=1.1`, `top_p=0.92`, `top_k=40`,
    `frequency_penalty=0.2`, `seed=1234` (again substituting supported tail knobs for `min_p`).
- **Loop detection:** distinct-3gram ratio plus a repeated ≥4-word phrase scan.
  `repetitionRatio = 1 − distinct_3grams / total_3grams`. A cell is flagged **loop** if
  `repetitionRatio > 0.5` **or** any ≥4-word phrase repeats ≥3×.

## Results (depth × config)

| depth | config | params | loop? | repetitionRatio | worst repeated phrase (count) | coherent? |
| --- | --- | --- | --- | ---: | --- | --- |
| 256K | A greedy | temp0 | **LOOP** | 0.878 | "i said i would" (28×) | no — incoherent |
| 256K | B rep_pen | temp0, rep_pen1.15 | no | 0.009 | — | yes |
| 256K | C freq_pres | temp0, freq0.4, pres0.4 | **LOOP** | 0.3555 | "he is calling after" (15×) | no |
| 256K | D (min_p→top_p/top_k) | temp0.7, top_p0.95, top_k50, seed1234 | no | 0.0108 | — | yes |
| 256K | E combo | temp0.6, rep_pen1.1, top_p0.92, top_k40, freq0.2, seed1234 | no | 0.0231 | — | yes |
| 512K | A greedy | temp0 | no | 0.0131 | — | yes |
| 512K | B rep_pen | temp0, rep_pen1.15 | **LOOP** | 0.5608 | "and then with a" / verbatim "ivory heel upon the mast-head" (×5) | no |
| 512K | C freq_pres | temp0, freq0.4, pres0.4 | **LOOP** | 0.0112 | "as if it were" (3×) | no (phrase-repeat flag) |
| 512K | D (min_p→top_p/top_k) | temp0.7, top_p0.95, top_k50, seed1234 | **LOOP** | 0.1869 | "the text of _the" (10×) | no |
| 512K | E combo | temp0.6, rep_pen1.1, top_p0.92, top_k40, freq0.2, seed1234 | no | 0.0 | — | degenerate-stop (15 tokens, "…Enjoy!") |

Notes on individual cells (all from the recorded run tables, no numbers altered):

- **A greedy @256K = LOOP** (`repetitionRatio 0.878`, "I said I would not be particular about my
  breakfast, and I would go." repeated verbatim to the token budget). Clearly incoherent.
- **A greedy @512K = NO loop** (`repetitionRatio 0.013`, coherent Melville continuation). Greedy
  stayed coherent at the *deeper* depth — so the collapse is **not** a deterministic ≥256K
  threshold; it is continuation-point / depth dependent.
- **B rep_pen @256K = coherent** (`0.009`) but **B rep_pen @512K = LOOP** (`0.5608`, the verbatim
  phrase "ivory heel upon the mast-head" repeats ×5). The single best anti-loop knob at 256K does
  **not** carry to 512K.
- **C freq_pres** looped at *both* depths (a phrase-repeat trip at 512K even though the 3gram ratio
  was low — the ≥4-word phrase scan is what catches "as if it were" ×3).
- **E combo @512K** did not loop but produced a **degenerate early stop** (15 tokens, "There's
  most of what you need in these books. Enjoy!") — not a loop, but not a useful long continuation
  either.

**No single config is coherent at BOTH 256K and 512K in the regime where greedy fails.**
`repetition_penalty ≈ 1.15` (config B) is the best single anti-loop knob **at 256K**, but it does
**not** generalize to 512K.

## Community-useful takeaways

1. **The collapse is not a fixed depth threshold.** Greedy looped at 256K but stayed coherent at
   512K on this build/prompt. The failure is tied to the specific continuation point / decode
   path, not to crossing a 256K line. This matches — and sharpens — the "flappy, sample-dependent"
   characterization in [`docs/LONG-CONTEXT-FINDINGS.md`](../docs/LONG-CONTEXT-FINDINGS.md).
2. **`min_p` is unavailable under DSpark spec-decode.** Requesting `min_p` returns **HTTP 400**
   ("not yet supported with speculative decoding") — the server rejects it while spec-decode is
   active. Any anti-degeneration recipe that leans on `min_p` (a common long-context suggestion)
   cannot be used as-is on this build; you have to fall back to `top_p` / `top_k` / penalties. This
   is a small, real, verifiable constraint worth knowing before you tune.

## Verdict — honest negative result

**No sampler configuration is a reliable cross-depth coherence fix on this build.** Every config
that held at one depth failed at the other (or degenerated to an early stop). We are **not**
publishing any sampler config as "the fix" for the long-context loop, because none of the five is.

The value here is the two verifiable constraints above: the collapse is **not** a deterministic
depth threshold, and **`min_p` is blocked under spec-decode**. Both narrow the search space for the
real fix — which, on the evidence so far, is not a sampler knob.

Cross-reference: the depth-vs-throughput curve, the acceptance-not-bandwidth mechanism, and the
original coherence-collapse characterization are in
[`docs/LONG-CONTEXT-FINDINGS.md`](../docs/LONG-CONTEXT-FINDINGS.md).
