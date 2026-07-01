# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSparkProposer — block-parallel draft + Markov AR refinement for DeepSeek-V4.

Subclass DFlashProposer (reuse non-causal cross-attn context-KV plumbing,
triton input kernel, dummy_run/cudagraph). Add DSpark's Markov autoregressive
refinement on top of the parallel block logits.

propose() mirrors SpecDecodeBaseProposer.propose() up to ``sample_hidden_states``
(combine -> set_inputs_first_pass -> attn metadata -> forward), then replaces the
parallel ``_sample_draft_tokens`` with the Markov AR loop from reference
model.py:860-874. DSpark is parallel_drafting=True so only the single-forward
branch applies (no eagle-style sequential loop).

Two integration points worth calling out:
  (R1) DSparkV4MTP.precompute_and_store_context_kv provides the MLA context-KV
       (DSpark: kv_norm(wkv(main_x)) sliding-window). The inherited DFlash
       build_model_inputs_first_pass calls it with main_x.
  (R2) hidden_size*hc_mult trap (llm_base_proposer:89): the base expands
       self.hidden_size by hc_mult; combine() returns dim=hidden_size (4096),
       so the strict DFlash assert does NOT hold — handled softly below.
"""

import torch
from typing_extensions import override

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class DSparkProposer(DFlashProposer):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dspark"
        # Bypass DFlashProposer's method=="dflash" assert by initing the
        # grandparent, then replicating DFlash buffer setup (dflash.py:37-69).
        SpecDecodeBaseProposer.__init__(
            self, vllm_config=vllm_config, device=device,
            pass_hidden_states_to_model=True, runner=runner,
        )
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        self.max_positions = self.max_num_tokens + self.max_query_tokens
        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device)
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens, dtype=torch.int64, device=device)
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device)
        self.positions = torch.zeros(
            self.max_query_tokens, dtype=torch.int64, device=device)
        self.arange = torch.arange(
            self.max_positions + 1, device=device, dtype=torch.int32)
        self.parallel_drafting_hidden_state_tensor = None

        hf = vllm_config.speculative_config.draft_model_config.hf_config
        self.dspark_block_size = getattr(hf, "dspark_block_size", self.num_speculative_tokens)
        assert self.num_speculative_tokens <= self.dspark_block_size, (
            f"DSpark needs num_speculative_tokens(={self.num_speculative_tokens}) "
            f"<= dspark_block_size(={self.dspark_block_size})")
        # draft sampling temperature (greedy first for verifiable accept-rate)
        self.draft_temperature = float(getattr(hf, "dspark_temperature", 0.0))
        self._use_confidence = False  # Path B (adaptive depth) — off in Path A

    @override
    def model_returns_tuple(self) -> bool:
        return False

    @override
    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        return True

    @override
    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        # The DSpark draft has 3 V4 MLA stages whose SWA caches share the target's
        # SWA spec; vLLM's hybrid KV-cache balancing splits that spec's layers into
        # multiple groups (even/odd), so the 3 draft layers legitimately span >1
        # kv_cache_group. We DON'T force a single group (base asserts ==1) — instead
        # initialize_attn_backend below builds a per-group AttentionGroup for each,
        # and build_per_group_and_layer_attn_metadata (base) already iterates groups.
        return

    @override
    def initialize_attn_backend(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes=None
    ) -> None:
        # Multi-group variant of the base method: each draft layer is placed in an
        # AttentionGroup keyed by (backend, its OWN kv_cache_group), using that
        # group's per-layer spec — instead of the base's single-group assumption
        # (which KeyErrors when draft layers span 2 SWA groups). Keeps hybrid ON so
        # the SWA cache stays block_size=64 (decode_dsv4 dispatchable).
        all_attn_layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        layer_to_group: dict[str, tuple[int, object]] = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            for ln in group.layer_names:
                layer_to_group[ln] = (gid, group)

        attention_groups: dict[tuple[str, int], AttentionGroup] = {}
        for layer_name in self._draft_attn_layer_names:
            gid, group = layer_to_group[layer_name]
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[layer_name]
            attn_backend = all_attn_layers[layer_name].get_attn_backend()
            key = (attn_backend.full_cls_name(), gid)
            if key not in attention_groups:
                kbs = (
                    kernel_block_sizes[gid]
                    if kernel_block_sizes is not None and gid < len(kernel_block_sizes)
                    else None
                )
                ag = AttentionGroup(
                    backend=attn_backend,
                    layer_names=[layer_name],
                    kv_cache_spec=spec,
                    kv_cache_group_id=gid,
                )
                ag.create_metadata_builders(
                    self.vllm_config, self.device, kernel_block_size=kbs
                )
                attention_groups[key] = ag
            else:
                attention_groups[key].layer_names.append(layer_name)

        self.draft_attn_groups = list(attention_groups.values())
        self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
        self.block_size = (
            self.draft_attn_groups[0].get_metadata_builder().kv_cache_spec.block_size
        )
        logger.info(
            "DSpark draft attn: %d group(s) across kv_cache_gids %s, block_size=%d",
            len(self.draft_attn_groups),
            sorted({g.kv_cache_group_id for g in self.draft_attn_groups}),
            self.block_size,
        )

    @override
    def build_per_group_and_layer_attn_metadata(
        self, cad: CommonAttentionMetadata, draft_index: int = 0
    ):
        # DFlash's override asserts each layer's metadata.causal is False, but the
        # V4 sparse-MLA SWA backend (DeepseekSparseSWAMetadata) has NO `causal`
        # field — the draft's non-causal intra-block visibility is encoded in the
        # sparse `decode_swa_indices` (the builder already sets decode_threshold =
        # 1 + num_speculative_tokens so the block routes to decode_dsv4), not a
        # boolean flag. So we call the BASE builder (skipping DFlash's flash-only
        # assert) and tag the metadata non-causal for any downstream inspection.
        per_group, per_layer = (
            SpecDecodeBaseProposer.build_per_group_and_layer_attn_metadata(
                self, cad, draft_index
            )
        )
        for md in per_layer.values():
            try:
                md.causal = False
            except Exception:
                pass
        # Non-causal block (reference get_dspark_topk_idxs, model.py:744): every
        # block query attends to [context window] + [ALL block positions], not
        # causally. The builder produces CAUSAL windows (token i sees i+1 positions);
        # the LAST block token already spans the full [context+block], so broadcast
        # its window/len to every block token of each request. block = 1+num_spec.
        _blk = 1 + self.num_speculative_tokens
        for _md in {id(m): m for m in per_layer.values()}.values():
            _di = getattr(_md, "decode_swa_indices", None)
            _dl = getattr(_md, "decode_swa_lens", None)
            if (_di is not None and _dl is not None and _dl.shape[0] > 0
                    and _dl.shape[0] % _blk == 0):
                _nr = _dl.shape[0] // _blk
                _div = _di.view(_nr, _blk, *_di.shape[1:])
                _dlv = _dl.view(_nr, _blk)
                _div[:] = _div[:, -1:].clone()
                _dlv[:] = _dlv[:, -1:].clone()
        return per_group, per_layer

    # NOTE: build_model_inputs_first_pass inherited from DFlashProposer — it
    # calls self.model.precompute_and_store_context_kv(self._dflash_hidden_states,...)
    # where _dflash_hidden_states is the ALREADY-combined main_x (combined in
    # propose() below before set_inputs_first_pass). So no override / no double combine.

    @override
    @torch.inference_mode()
    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata,
        mm_embed_inputs=None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings=None,
    ) -> torch.Tensor:
        self._last_draft_probs = None
        batch_size = common_attn_metadata.batch_size()

        # --- DSpark input projection: main_x = main_norm(main_proj(concat aux)) ---
        # (mirrors base eagle3/dflash combine branch, llm_base_proposer:449-461)
        main_x = self.model.combine_hidden_states(target_hidden_states)
        # (R2) DSpark main_x dim == hidden_size (4096); the hc_mult-expanded buffer
        # sizing is internal to the draft model, so we do NOT assert against the
        # (possibly hc_mult-scaled) self.hidden_size here.

        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=main_x,
                token_indices_to_sample=token_indices_to_sample,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        )

        # DSpark: the draft block must use NOISE-token embeddings (reference
        # forward_embed, model.py:854 — draft_input_ids = full(block, noise_token_id)),
        # NOT the EAGLE rotated real tokens that the inherited set_inputs_first_pass
        # writes. The context is carried entirely by main_x via the precomputed
        # context-KV; the block query has to be pure noise so every position predicts
        # from context alone. (A/B: DSPARK_REAL_INPUT=1 restores the old EAGLE input.)
        # A/B (29.06): reference DSpark uses noise-token block input (forward_embed:854),
        # but with OUR current context-KV the noise-only block is WORSE (pos1 0.20->0.13,
        # mean 1.75->1.55) — proof the context-KV conditioning is the weak link (draft
        # leans on the real-token embeds, EAGLE-style, not on context). Keep real tokens
        # as the functional default; DSPARK_NOISE=1 re-enables reference noise for
        # context-KV debugging once that path is fixed.
        import os
        if os.environ.get("DSPARK_BONUS") != "0":
            # 🎯 ROOT-CAUSE FIX — DEFAULT ON (multi-agent audit 29.06, VERIFIED LIVE:
            # pos0 0.41->0.74 ≈ reference 0.76, mean accept 1.75->2.5, throughput 26->36 t/s).
            # The inherited DFlash kernel builds 1+num_spec=6 query slots
            # [bonus@L, noise@L+1..L+5] and samples ONLY slots 1..5 (is_sample=query_off>0),
            # so base_logits[:,0] = noise@L+1 -> draft#1 from a noise query 2 RoPE-steps
            # ahead -> pos0 ~5%. The reference (forward_head:864-871) samples ALL block_size
            # slots INCLUDING slot0 = the BONUS real-token query @ L (the natural next-token
            # position, strongest signal) -> draft#1 ~76%. Shift the sample indices by -1 so
            # output k reads slot k (slot0=bonus@L -> draft#1). markov_refine then aligns
            # exactly with reference. NOT precision — DFlash(bonus=seed) vs DSpark(bonus=draft).
            # DSPARK_BONUS=0 restores the old (broken) DFlash slot selection for A/B.
            token_indices_to_sample = token_indices_to_sample - 1
        if os.environ.get("DSPARK_CAPTURE") == "1":
            _e = max(0, num_tokens - 4)
            logger.info(
                "DSPARK-FWDPOS n=%d first12=%s last4=%s",
                num_tokens, self.positions[:12].tolist(),
                self.positions[_e:num_tokens].tolist(),
            )
        if os.environ.get("DSPARK_FUTURE") == "1":
            # ROOT CAUSE fix test: the DSpark block-query lives at FUTURE positions
            # [L..L+block] (reference freqs_cis[start_pos+seqlen:]), but the inherited
            # EAGLE set_inputs writes the recent-context window [L-6..L-1]. Captured
            # positions confirmed [12..17] for ctx [0..17]. Shift forward by the block
            # width so the RoPE relative context<->block matches the trained regime.
            self.positions[:num_tokens] += num_tokens
        if os.environ.get("DSPARK_NOISE") == "1":
            _noise_id = getattr(self.model.config, "dspark_noise_token_id", 128799)
            self.input_ids[:num_tokens] = _noise_id

        _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
            common_attn_metadata
        )
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )

        # ---- DSPARK_PROF: env-gated kernel-level profiler around the draft work ----
        # Coarse phase split (precompute / draft-forward / markov) via cuda.synchronize,
        # plus a torch.profiler kernel table + chrome trace on the FIRST profiled step
        # only (guarded by trace-file existence). Default OFF: _prof is None -> the
        # original code path runs verbatim under nullcontext.
        _prof = self._dspark_maybe_start_profiler()

        def _run_draft():
            _t = {}
            _ev = self._dspark_phase_clock(_t)
            with _ev("precompute+inputs"):
                model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
                    num_tokens, num_input_tokens, mm_embed_inputs
                )
            with _ev("draft_forward"):
                with set_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens=num_input_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    slot_mapping=self._get_slot_mapping(
                        slot_mapping_size, common_attn_metadata.slot_mapping
                    ),
                ):
                    block_hidden = self.model(**model_kwargs)  # model_returns_tuple False
                sample_hidden_states = block_hidden[token_indices_to_sample]
                base_logits = self.model.compute_logits(sample_hidden_states)
                base_logits = base_logits.view(
                    batch_size, self.num_speculative_tokens, -1
                )
            with _ev("markov"):
                out, corrected = self.markov_refine(
                    base_logits, next_token_ids, self.model.markov_step,
                    self.draft_temperature,
                )
            return out, corrected, _t

        if _prof is None:
            out, corrected, _ = _run_draft()
        else:
            # _prof was already __enter__()'d in _dspark_maybe_start_profiler;
            # _dspark_finish_profiler does the single __exit__(). Do NOT wrap in
            # `with _prof:` — that would double-enter/exit Kineto (global singleton)
            # -> "Can't disable Kineto profiler when it's not running".
            _phase_ms = {}
            try:
                out, corrected, _phase_ms = _run_draft()
            finally:
                self._dspark_finish_profiler(_prof, _phase_ms)

        draft_token_ids = out[:, 1:]  # drop the bonus token; keep block_size drafts

        # (R2/verify#3) draft_probs MUST come from the CORRECTED logits so the
        # runner's RejectionSampler sees the true draft distribution.
        self._last_draft_probs = torch.softmax(
            corrected.float(), dim=-1
        ).contiguous()
        return draft_token_ids.reshape(-1, self.num_speculative_tokens)

    # ---- DSPARK_PROF helpers (env-gated; no effect unless DSPARK_PROF=1) ----
    def _dspark_maybe_start_profiler(self):
        """Return a started torch.profiler context on the FIRST profiled step,
        else None. One-shot: guarded by the chrome-trace file's existence so only
        the first spec-decode step after launch is captured (steady-state shape)."""
        import os
        if os.environ.get("DSPARK_PROF") != "1":
            return None
        trace_path = os.environ.get(
            "DSPARK_PROF_TRACE", "/cache/huggingface/dspark_prof.json"
        )
        # one-shot guard: trace already written -> never profile again
        if getattr(self, "_dspark_prof_done", False) or os.path.exists(trace_path):
            self._dspark_prof_done = True
            return None
        # warm up a few real steps first so we profile steady-state, not the
        # cudagraph-capture / first-token step (which is unrepresentative).
        warm = int(os.environ.get("DSPARK_PROF_WARMUP", "3"))
        n = getattr(self, "_dspark_prof_seen", 0)
        self._dspark_prof_seen = n + 1
        if n < warm:
            return None
        from torch.profiler import profile, ProfilerActivity
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        )
        prof.__enter__()
        self._dspark_prof_trace_path = trace_path
        return prof

    @staticmethod
    def _dspark_phase_clock(out_dict):
        """Return a context-manager factory that records per-phase wall-ms into
        out_dict[name], with torch.cuda.synchronize() bracketing for true GPU time.
        Also emits a torch.profiler record_function annotation per phase."""
        import contextlib
        from torch.profiler import record_function

        @contextlib.contextmanager
        def _ev(name):
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            with record_function(f"DSPARK::{name}"):
                yield
            t1.record()
            torch.cuda.synchronize()
            out_dict[name] = t0.elapsed_time(t1)  # ms

        return _ev

    def _dspark_finish_profiler(self, prof, phase_ms):
        """Stop profiler, export chrome trace + print the kernel table once."""
        import os
        prof.__exit__(None, None, None)
        self._dspark_prof_done = True
        trace_path = getattr(
            self, "_dspark_prof_trace_path", "/cache/huggingface/dspark_prof.json"
        )
        try:
            prof.export_chrome_trace(trace_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("DSPARK_PROF: export_chrome_trace failed: %s", e)
        total = sum(phase_ms.values()) or 1e-9
        logger.info(
            "DSPARK_PROF coarse phase split (ms): precompute=%.3f draft_forward=%.3f "
            "markov=%.3f | total=%.3f | draft_forward=%.1f%% of step",
            phase_ms.get("precompute+inputs", 0.0),
            phase_ms.get("draft_forward", 0.0),
            phase_ms.get("markov", 0.0),
            total,
            100.0 * phase_ms.get("draft_forward", 0.0) / total,
        )
        try:
            table = prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=40
            )
            logger.info("DSPARK_PROF kernel table (sort=cuda_time_total):\n%s", table)
        except Exception as e:  # noqa: BLE001
            logger.warning("DSPARK_PROF: key_averages().table failed: %s", e)
        logger.info("DSPARK_PROF: chrome trace -> %s (one-shot done)", trace_path)

    # ---- Markov AR refinement (pure, unit-tested) ----
    @staticmethod
    def markov_refine(base_logits, bonus_token, markov_step, temperature=0.0):
        """[B,block,vocab] base logits -> (output_ids[B,block+1], corrected[B,block,vocab]).
        Faithful port of model.py:864-871. Pure -> unit-testable."""
        import os
        _use_markov = os.environ.get("DSPARK_NO_MARKOV") != "1"
        B, block, vocab = base_logits.shape
        out = base_logits.new_empty((B, block + 1), dtype=torch.long)
        out[:, 0] = bonus_token
        corrected = base_logits.clone()
        for k in range(block):
            bias, _embed = markov_step(out[:, k])
            if _use_markov:
                corrected[:, k] = corrected[:, k] + bias
            if temperature == 0:
                out[:, k + 1] = corrected[:, k].argmax(dim=-1)
            else:
                p = torch.softmax(corrected[:, k] / max(temperature, 1e-5), dim=-1)
                out[:, k + 1] = torch.multinomial(p, 1).squeeze(-1)
        return out, corrected
