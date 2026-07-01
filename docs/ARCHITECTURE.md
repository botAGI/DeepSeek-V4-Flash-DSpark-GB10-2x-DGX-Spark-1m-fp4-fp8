# DSpark on GB10 — Architecture

How DSpark's draft head and proposer integrate into vLLM's speculative-decoding
pipeline on DeepSeek-V4-Flash. This is the "what the port actually does" doc; for
performance and the honest gap analysis see
[GB10-PERFORMANCE.md](GB10-PERFORMANCE.md), for exact patch anchors see
[../patches/PATCHES.md](../patches/PATCHES.md).

Everything below is grounded in the vendored reference (`reference/deepseek-ref/
inference/model.py`) and was verified against a loaded model on GB10 unless noted.

---

## 1. What DSpark is (and is not)

DSpark is a **speculative-decoding module attached to a DeepSeek-V4 checkpoint** —
**not** a new base model. The heavy per-stage transformer is byte-for-byte the existing
`DeepseekV4DecoderLayer` (MLA + 256-expert MoE + per-block mHC). DSpark stacks **3 such
stages** under the `mtp.*` checkpoint namespace and adds a small amount of glue:

```
target model (43 layers)                      DSpark draft (3 stages, mtp.*)
┌──────────────────────────┐                  ┌────────────────────────────────┐
│  layers 0..42            │                  │ stage 0:  main_proj + main_norm │
│  emit hidden @ [40,41,42]│ ──aux hidden──▶  │ stage 1:  DeepseekV4DecoderLayer│
│  (EAGLE3 interface)      │                  │ stage 2:  + norm + hc_head       │
└──────────────────────────┘                  │           + markov_head          │
        ▲          │ verify 6 tokens          │           + confidence_head      │
        │          ▼                          └────────────────────────────────┘
        └────── accept/reject ◀── block of block_size(5)+bonus(1) draft tokens ──┘
```

- **stage 0** consumes the concatenation of the target's hidden states from
  `dspark_target_layer_ids = [40,41,42]` (3 × `hidden` → `hidden`) through `main_proj`,
  then `main_norm`. This is the context summary the draft conditions on.
- **stages 1–2** are full DeepSeek-V4 decoder stages (MLA + MoE-256).
- **stage 2** additionally carries the final mHC vocab projection (`norm` + `hc_head_*`),
  the **Markov head** (low-rank rank-256 per-token bias), and a confidence / adaptive-depth
  head.

Checkpoint facts (`config.json`, verified): `dspark_block_size=5`,
`dspark_noise_token_id=128799`, `dspark_target_layer_ids=[40,41,42]`,
`dspark_markov_rank=256`, 3 MTP stages, fp8 block-quant weights, FP4 experts,
vocab 129280.

---

## 2. The draft step — block-parallel backbone + Markov refinement

Per decode step the draft produces `block_size` (5) candidate tokens, which the target
then verifies in a single 6-wide forward (5 draft + 1 bonus). DSpark generates the block
in two parts:

### 2a. Backbone (parallel block)

The 3 draft stages run a **single parallel forward** over the block. The block query is
`[bonus@L, noise@L+1 .. noise@L+5]`: slot 0 is the real bonus token at the natural
next-token position `L`, slots 1..5 are the noise token (`128799`) at future positions
`L+1..L+5`. The block attends **non-causally within itself** — every block token sees
`[context window] + [all block tokens]` (reference `get_dspark_topk_idxs`) — over the
context-KV written from `main_x` into a sliding-window ring cache.

This backbone is heavy: 3 × (MLA + 256-expert MoE) for 6 tokens. On GB10 it's ~15% of the
step, and because the verify forward is the fixed ~65 ms cost, this is what makes a lighter
MTP draft preferable for a single interactive stream (see GB10-PERFORMANCE.md).

### 2b. Markov autoregressive refinement

The backbone gives base logits per block position. The **Markov head** then refines them
autoregressively across the block (reference `forward_head`):

```
out[0] = sample(base_logits[0])
for i in 1..block_size-1:
    base_logits[i] += markov_step(out[i-1])     # low-rank rank-256 token→token bias
    out[i]          = sample(base_logits[i])
```

`markov_step` is `bias = w2(w1(token))` — a cheap embedding + linear. Empirically (A/B on
GB10) the Markov refinement is **essential**: it lifts position-0 acceptance from ~5%
(backbone alone) to ~41% pre-fix, and the full chain to mean ~2.5 post-fix. The backbone
provides the context; Markov provides the sharp token-to-token structure.

---

## 3. vLLM integration

DSpark is a new speculative method (`method="dspark"`) wired into vLLM through patches —
no base-model fork. The pieces:

| Layer | Patch | What it does |
|---|---|---|
| `config/speculative.py` | enum + predicates | register `dspark`; `use_dspark()`, `parallel_drafting=True`; `hf_config_override` maps the DSpark checkpoint to `DSparkV4MTP` arch |
| `gpu_model_runner.py` | dispatch | instantiate `DSparkProposer` (above the `use_dflash()` branch); `use_aux_hidden_state_outputs=True` |
| `registry.py` | model map | `DSparkV4MTP` → `vllm.models.deepseek_v4.nvidia.dspark` |
| `models/deepseek_v4/.../model.py` | EAGLE3 interface | `SupportsEagle3` on the base model so it emits hidden states from `[40,41,42]` |
| `nvidia/dspark.py` | new module | the `DSparkV4MTP` draft head (§1–2) |
| `v1/spec_decode/dspark.py` | new module | `DSparkProposer` |

### `DSparkProposer` (the proposer)

`DSparkProposer` subclasses `DFlashProposer`, reusing its non-causal cross-attention
context-KV plumbing, the Triton input kernel, and dummy-run / cudagraph machinery.
`propose()` mirrors the base proposer up to `sample_hidden_states` (combine → set inputs →
build attn metadata → forward), then replaces the parallel sampler with the **Markov AR
loop** (§2b). Because DSpark is `parallel_drafting=True`, only the single-forward branch
applies (no EAGLE-style sequential loop).

The DFlash cross-attention scheme turns out to *be* the DSpark scheme: query = bonus
(real next-token) + mask tokens (`128799` = DSpark's noise token), with separate
context/query buffers and the block at future positions `[L .. L+block]`. The
`apply-patches.sh` config sets `dflash_config={mask_token_id: 128799, ...}` so the inherited
machinery produces exactly the reference input layout.

---

## 4. The bugs that mattered

The hard part of this port was **numerical correctness of the draft**, not getting it to
load. The draft initially produced near-noise (pos0 ≈ 5%). A multi-agent adversarial audit
and a long grounded bug-hunt produced three fixes (all default-on, in `patches/`):

1. **Bonus-slot sampling** (`dspark.py` proposer, `token_indices_to_sample -= 1`).
   The inherited kernel sampled block slots **1..5**, so draft #1 came from a *noise* query
   2 RoPE-steps ahead → pos0 ≈ 5%. The reference samples **all** slots including slot 0 =
   the real bonus query at the natural next-token position. Shift by −1 so output `k` reads
   slot `k`. **pos0 0.41 → 0.74 (≈ reference 0.76), mean 1.75 → 2.5, 26 → 36 t/s.** This was
   the root cause — and it disproved the earlier (wrong) hypotheses that the gap was an
   fp8/fp4 precision wall or an EAGLE-vs-DSpark position-regime mismatch.

2. **Aux hc-fold** (`model.py` base patch). The EAGLE3 aux hidden states must be the
   **per-layer** `hc_post(...).mean(dim=1)` of the hc-folded hidden (reference `model.py:921`),
   not the raw single tensor. The base model calls `hc_post` once after the loop; the draft
   needs it per aux layer, mean-folded over the `hc_mult` axis, so the runner concatenates a
   correct `[T, 3*dim]` for `main_proj`.

3. **LM-head norm + non-causal block** (`dspark.py` head + proposer). Apply `last.norm`
   before the head (the checkpoint ships `mtp.N.norm.weight`, not `shared_head.norm`, which
   stays at init), and broadcast the SWA window so all block tokens see `[context] + [all
   block]` — DSpark is non-causal *within* the block, the inherited builder made it causal.

Separately, a **concurrency stability fix** in `flashmla_sparse` aligns the C128A and SWA
metadata builders' decode/prefill split threshold to `1 + num_spec`. Without it, a 7–11-token
chunked-prefill extend co-occurring with verifies under concurrency hits
`extra_kv_cache requires extra_indices`. Prod MTP-2 was immune (`parallel_drafting=False`
made both thresholds equal); DSpark's `parallel_drafting=True` exposed the mismatch. Fixed →
8/8 concurrent streams stable.

---

## 5. What's verified vs what isn't

**Verified (live, against a loaded model on GB10):** the full pipeline matches the
reference structurally and numerically where checkable — positions (future block
`[L..L+block]`), input scheme (bonus + mask=`128799`), `markov_step`, the mHC head, the
attention math (a plain-torch dense A/B reproduced `decode_dsv4`), expert loading
(`seen == placed`), `q_head_norm`, and the output RoPE. Acceptance pos0 0.62–0.74 ≈
reference ~0.76.

**Not verifiable here:** a full layer-by-layer numerical diff against DeepSeek's reference
inference. The reference uses `tilelang` kernels that don't build on `sm_121`, so it can't
be run on GB10 (or on CPU — it's GPU-only). That oracle is unavailable. After the bonus-slot
fix, acceptance reaches the reference ballpark, so a remaining oracle-only discrepancy — if
any — is small. We flag it rather than claim parity we can't measure.
