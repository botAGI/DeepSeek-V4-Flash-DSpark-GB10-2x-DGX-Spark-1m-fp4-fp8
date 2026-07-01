# Credits & Lineage

This repo carries two DSpark recipes on GB10 (DGX Spark, `sm_121`): an **fp8** line that is ours,
and an **NVFP4-KV** line that is **not** ours (tonyd2wild's, reproduced and measured here). None of
the core DSpark algorithm or the DeepSeek-V4-Flash model is ours — our contribution is the GB10
fp8 bring-up and the extended cross-build benchmarks. Full credit, with gratitude.

## Our contribution

- **The fp8 GB10 / `sm_121` DSpark bring-up** (`fp8_ds_mla` KV path): wiring the DSpark draft head
  and proposer into vLLM's spec-decode loop on consumer Grace-Blackwell. Our patch-set is
  `apply-patches.sh` plus everything under `patches/` (the `DSparkV4MTP` draft head, the
  `DSparkProposer`, the speculative-config enum/predicate wiring, and the base-model EAGLE3 aux
  plumbing).
- **Three GB10-specific bug-fixes** that took the draft from "predicts noise" to "matches the
  reference": the bonus-slot sampling shift, the aux hc-fold, and the correct LM-head norm +
  non-causal block window. Plus a **concurrency-stability fix** (`flashmla_sparse` C128A / SWA
  threshold alignment) and the **seqs-scaling** work behind the seqs=12 prod config.
- **The extended, cross-build benchmark suite** — the value-add of this repo, all measured on our
  2× DGX Spark (TP=2 over RoCE):
  - the **1M-context single-stream depth curve** (the number nobody in the community published);
  - the **concurrency scaling** (fp8 seqs=12; NVFP4 C16 reproduction);
  - the **long-context coherence / sampling study** — including the honest negative result that no
    sampler config is a reliable cross-depth fix, and the `min_p`-blocked-under-spec-decode finding.

## Lineage we build on

- **DeepSeek-AI** — the **DeepSeek-V4-Flash** model and the **DSpark** speculative-decoding
  algorithm (block-parallel draft + Markov autoregressive refinement), released 2026-06-27 under the
  **MIT License**. The draft-head architecture (`DSparkBlock` / `markov_head` / `confidence_head`),
  the `mtp.*` weight layout, and the accept/verify algorithm derive from DeepSeek's published
  reference (`inference/model.py`, `config.json`) and from
  [DeepSpec](https://github.com/deepseek-ai/DeepSpec) (MIT). → <https://github.com/deepseek-ai>
- **vLLM** ([vllm-project](https://github.com/vllm-project/vllm), **Apache-2.0**) — the serving
  engine. Our port is implemented as vLLM patches (a new `dspark` speculative method, a
  DeepSeek-V4 draft-head model class, a DFlash-derived proposer).
- **jasl vLLM fork / NVIDIA DGX-Spark playbook lineage** — the `sm_120`/`sm_121` enablement that
  made DeepSeek-V4 run on GB10 at all (the B12X MoE path, Triton sparse-MLA attention, the `12.1a`
  arch build). Without this lineage there is no GB10 base image to patch.

## The NVFP4-KV line (not ours)

The faster NVFP4-KV recipe measured in this repo's benchmarks is **not ours**. Full credit to the
community chain that produced it:

| Contributor | What we depend on |
| --- | --- |
| **tonyd2wild** | The [NVFP4-1M-2×Spark recipe](https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark) (commit `89bb82b`) — Stage A/B/C runtime packaging (`nvfp4_ds_mla` → 584-byte padded envelope), the two-node launch flow, and the validated benchmark artifacts we compared against. |
| **bjk110** | The `vllm-spark` **unholy-fusion** base image (`ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready`) the NVFP4 stack builds on. |
| **drowzeys / "Keys"** | The `nvfp4_ds_mla` KV-cache plumbing **and** the in-server DSpark concurrency patch (request-stable main-KV slots + ragged `query_start_loc` handling) that makes `max_num_seqs > 1` correct. Every NVFP4 concurrency number depends on this patch. |
| **rafaelcaricio** | The DSpark ↔ vLLM integration. |
| **fraserprice** | The DeepSeek-V4-Flash-DSpark model/runtime (`dspark-vllm`). |
| **MiaAI-Lab** | The two-node DGX Spark packaging and worker-first launch lineage. |

**Plainly:** we **reproduced and measured** the NVFP4-KV line byte-identically on our own 2× DGX
Spark cluster. We did **not** author it, we changed **nothing** about how the build is produced, and
we claim **no speed win over it** — our C16-static reproduction ties tonyd2wild's published number
within his own run-to-run variance (a `WO_PROJECTION`-lever-off control re-derives his 315.1). To
build the NVFP4 recipe, go to the [upstream repo](https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
and its own README / CREDITS / license.

## License

Our patches, documentation, and measurement scripts are released under the **MIT License** (see
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). The reproduced NVFP4 recipe, the vLLM-derived overlay,
and Keys' concurrency patch retain their upstream **MIT / Apache-2.0** licenses and authorship.
**No model weights are redistributed by this repository.** Model weights, base images, CUDA/NCCL,
FlashInfer, TileLang, and Triton are separate upstream artifacts under their own terms.
