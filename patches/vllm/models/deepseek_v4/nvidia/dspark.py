# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft head for DeepSeek-V4 (codename DeepseekV4), GB10/sm_121 fork.

DSpark (DeepSeek + PKU, 2026-06-27, MIT) is a speculative-decoding *module*
attached to a DeepSeek-V4 checkpoint. It is NOT a new base model — the heavy
per-stage transformer is byte-for-byte our existing ``DeepseekV4DecoderLayer``
(MLA + 256-expert MoE + per-block mHC). DSpark adds, on top of 3 stacked stages
stored under the ``mtp.*`` checkpoint namespace:

  * stage 0  — ``main_proj`` (concat of target hidden states from
    ``dspark_target_layer_ids`` = [40,41,42], i.e. 3*hidden -> hidden) + ``main_norm``;
  * stage N-1 (==2) — ``norm`` + ``hc_head_{fn,base,scale}`` (final mHC vocab
    projection) + ``markov_head`` (low-rank rank=256 per-token bias) +
    ``confidence_head`` (adaptive-depth accept-rate predictor).

Reference (source of truth): DeepSeek's published DSpark inference reference
``inference/model.py`` (see CREDITS.md):
  ParallelHead 719-740, DSparkAttention 750-792, DSparkMarkovHead 795-804,
  DSparkConfidenceHead 807-815, DSparkBlock 818-874, Transformer.forward_spec 928-936.

This file is a sibling of ``nvidia/mtp.py`` (DeepSeekV4MTP). It reuses that file's
load_weights machinery (expert mapping, fused_wqa_wkv stacking, attn_sink TP-narrow,
E8M0 scale handling) and only diverges in:
  (a) per-stage module layout (main_proj/main_norm vs enorm/hnorm/e_proj/h_proj;
      markov/confidence/norm/hc_head only on the last stage);
  (b) spec_layer_weight_names;
  (c) the markov autoregressive refinement exposed for DSparkProposer.

NOTE: the *attention context-KV* path of DSpark (DSparkAttention: sliding-window
ring cache fed from ``main_x``) does not map 1:1 onto vLLM's paged MLA precompute.
The model below exposes the clean primitives (``combine_hidden_states``,
``forward``, ``compute_block_logits``, ``markov_step``, ``predict_confidence``);
wiring them to the runner's attention backend is done in
``vllm/v1/spec_decode/dspark.py``. The weight-name routing and heavy module reuse
are exercised offline by ``patches/tests/test_dspark_weight_routing.py``.
"""

import typing
from collections.abc import Callable, Iterable

import regex as re
import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mhc import HCHeadOp
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.model_executor.models.deepseek_v2 import get_spec_layer_idx_from_weight_name
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .model import (
    DeepseekV4DecoderLayer,
    make_deepseek_v4_expert_params_mapping,
)

logger = init_logger(__name__)

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


class _DSparkMarkovHead(nn.Module):
    """Nested to match checkpoint names mtp.N.markov_head.markov_w{1,2}.weight.

    TP-REPLICATED (perf). The Markov head is tiny
    (vocab*rank fp32 ≈ 130M params ≈ 530MB/table) but ran in the spec-decode hot
    loop block_size times per step. When sharded (VocabParallelEmbedding +
    ParallelLMHead) each iteration paid two fp32 cross-node collectives —
    an all-reduce (combine vocab-sharded embedding) and an all-gather (gather
    vocab-sharded logits). Profiling showed 5 serialized fp32 RoCE collectives =
    12.2ms = 56% of the whole spec-decode step. Replicating the full [vocab, rank]
    tables on every rank makes step() purely local: the embedding lookup and the
    linear yield the FULL [B, vocab] result with ZERO collectives. Numerically
    identical (each rank computes the same full result), so acceptance is unchanged.
    """

    def __init__(self, vocab_size: int, rank: int, prefix: str):
        super().__init__()
        self.vocab_size = vocab_size
        # Full-vocab REPLICATED tables (plain nn modules => .weight has no
        # vocab-sharding weight_loader; default_weight_loader loads the full
        # [vocab, rank] checkpoint tensor on every rank). fp32 (markov computes
        # in fp32; FP32_CAST_KEYS still applies on load).
        self.markov_w1 = nn.Embedding(vocab_size, rank, dtype=torch.float32)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False, dtype=torch.float32)

    def step(self, token_ids: torch.Tensor):
        embed = self.markov_w1(token_ids)                          # [B, rank] fp32 (LOCAL lookup)
        # markov_w2.weight is the FULL [vocab, rank] table on every rank, so the
        # matmul yields the FULL [B, vocab] logits bias locally — NO all-gather.
        bias = torch.nn.functional.linear(embed, self.markov_w2.weight)  # [B, vocab]
        return bias, embed


class _DSparkConfidenceHead(nn.Module):
    """Nested to match checkpoint name mtp.N.confidence_head.proj.weight (no bias)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = ReplicatedLinear(
            input_dim, 1, bias=False, return_bias=False,
            params_dtype=torch.float32, quant_config=None,
        )

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor):
        x = torch.cat([hidden, markov_embed], dim=-1).float()
        return self.proj(x).squeeze(-1)


class DSparkV4MultiTokenPredictorLayer(nn.Module):
    """One DSpark stage. Heavy ``mtp_block`` reused verbatim; per-stage extras
    (main_proj/main_norm on stage 0; norm/hc_head/markov/confidence on last
    stage) added to mirror the reference DSparkBlock (model.py:818-874)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        topk_indices_buffer: torch.Tensor,
        prefix: str,
        stage_id: int,
        num_mtp_layers: int,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ) -> None:
        super().__init__()

        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
        self.rms_norm_eps = config.rms_norm_eps
        self.stage_id = stage_id

        hidden_size = config.hidden_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult

        # --- DSpark stage-0 input projection (replaces V3-MTP enorm/hnorm/e_proj/h_proj) ---
        # main_proj takes the concatenation of target hidden states from
        # dspark_target_layer_ids (=[40,41,42]) -> hidden_size. (model.py:832-833)
        if stage_id == 0:
            target_layers = list(config.dspark_target_layer_ids)
            assert len(target_layers) > 0, "DSpark needs target layers"
            self.main_proj = ReplicatedLinear(
                hidden_size * len(target_layers),
                hidden_size,
                bias=False,
                return_bias=False,
                quant_config=quant_config,
            )
            self.main_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        # --- DSpark last-stage output head (norm + mHC head + markov + confidence) ---
        # Only the final stage carries the vocab projection machinery and the
        # markov/confidence heads (index.json: these tensors exist ONLY on mtp.2).
        if stage_id == num_mtp_layers - 1:
            self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
            self.hc_dim = self.hc_mult * hidden_size
            self.hc_head_fn = nn.Parameter(
                torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(self.hc_mult, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_scale = nn.Parameter(
                torch.empty(1, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_op = HCHeadOp()
            self.shared_head = SharedHead(
                config=config, prefix=prefix, quant_config=quant_config
            )
            # Markov head: low-rank per-token bias. markov_w1 looks up a rank-256
            # embedding of the previous draft token, markov_w2 projects it back to
            # vocab logits-space (model.py:795-804). Checkpoint stores both bf16;
            # we keep them fp32 (reference keeps fp32 "for easier logits compute").
            rank = config.dspark_markov_rank
            # Nested module names MUST match checkpoint: markov_head.markov_w{1,2},
            # confidence_head.proj (verify via index.json).
            self.markov_head = _DSparkMarkovHead(
                config.vocab_size, rank, maybe_prefix(prefix, "markov_head"))
            self.confidence_head = _DSparkConfidenceHead(hidden_size + rank)

        self.mtp_block = DeepseekV4DecoderLayer(
            vllm_config,
            prefix,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )

    # ---- DSpark primitives (mirror model.py DSparkBlock) ----

    def combine_hidden_states(self, main_hidden: torch.Tensor) -> torch.Tensor:
        """stage-0 only: main_norm(main_proj(concat target hiddens)). (model.py:853)"""
        assert self.stage_id == 0
        return self.main_norm(self.main_proj(main_hidden))

    def markov_step(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """last-stage: returns (per-token logits bias, rank-256 markov embed).
        (model.py:801-804)"""
        return self.markov_head.step(token_ids)

    def predict_confidence(
        self, hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        """last-stage: sigmoid-able accept-rate score per position. (model.py:813-815)"""
        return self.confidence_head(hidden, markov_embed)

    def apply_hc_head(self, x: torch.Tensor) -> torch.Tensor:
        """last-stage final mHC vocab projection before head (model.py:862)."""
        return self.hc_head_op(
            x,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )


class DSparkV4MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        # ⚠ R1: DSpark uses 3 stacked stages applied SEQUENTIALLY (mtp.0/1/2),
        # NOT the modulo-reused single MTP layer. Root config.json misleadingly
        # has num_nextn_predict_layers=1; the real count is n_mtp_layers=3
        # (inference/config.json). Default 3; overridable via hf_config_override.
        self.num_mtp_layers = getattr(
            config, "dspark_n_mtp_layers",
            getattr(config, "n_mtp_layers", 3),
        )
        self.device = current_platform.device_type

        topk_tokens = config.index_topk
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            topk_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        aux_stream_list = (
            None
            if current_platform.is_rocm()
            else [torch.cuda.Stream() for _ in range(3)]
        )

        self.layers = torch.nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + s): DSparkV4MultiTokenPredictorLayer(
                    vllm_config,
                    self.topk_indices_buffer,
                    f"{prefix}.layers.{self.mtp_start_layer_idx + s}",
                    stage_id=s,
                    num_mtp_layers=self.num_mtp_layers,
                    aux_stream_list=aux_stream_list,
                )
                for s in range(self.num_mtp_layers)
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    @property
    def first_stage(self) -> DSparkV4MultiTokenPredictorLayer:
        return self.layers[str(self.mtp_start_layer_idx)]

    @property
    def last_stage(self) -> DSparkV4MultiTokenPredictorLayer:
        return self.layers[str(self.mtp_start_layer_idx + self.num_mtp_layers - 1)]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, main_hidden: torch.Tensor) -> torch.Tensor:
        # contract used by DSparkProposer (mirror qwen3_dflash.combine_hidden_states)
        return self.first_stage.combine_hidden_states(main_hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # Heavy draft forward: run all stacked DSpark stages over the
        # noise-token block embedding. The block geometry / context-KV is
        # supplied by DSparkProposer via attn metadata. Returns last-stage
        # pre-hc_head hidden (flattened), like DeepSeekV4MTP.
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # ⚠ R2: DeepseekV4DecoderLayer expects x of shape [T, hc_mult, dim].
        # Expand token embeds to hc_mult copies for Hyper-Connections
        # (reference model.py:857: x.unsqueeze(2).repeat(1,1,hc_mult,1)).
        hc_mult = self.config.hc_mult
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, hc_mult, 1)
        # DSpark stages fold per-stage (hc_post after each), mirroring the reference
        # forward_spec which passes the folded 4D h between stages. (A/B 29.06:
        # carrying hc state across stages — target's 43-layer pattern — HURT
        # positions 1+ (0.20->0.11), confirming per-stage fold is correct here.)
        residual = post_mix = res_mix = None
        for s in range(self.num_mtp_layers):
            layer = self.layers[str(self.mtp_start_layer_idx + s)]
            hidden_states, residual, post_mix, res_mix = layer.mtp_block(
                positions=positions, x=hidden_states, input_ids=None
            )
            if current_platform.is_cuda():
                hidden_states = layer.mtp_block.hc_post(
                    hidden_states, residual, post_mix, res_mix
                )
        return hidden_states.flatten(1)

    def compute_block_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Base (pre-markov) block logits from last-stage hc_head+head. (model.py:862-863)"""
        last = self.last_stage
        hidden_states = hidden_states.view(-1, last.hc_mult, self.config.hidden_size)
        hidden_states = last.apply_hc_head(hidden_states)
        # Reference forward_head (model.py:863): head(self.norm(x)). The DSpark ckpt
        # stores the final norm as "mtp.N.norm.weight" (-> last.norm) and a GLOBAL
        # "head.weight"; there is NO "shared_head.norm" in the ckpt, so
        # shared_head.norm stays at init (ones) and must NOT be used. Apply the
        # loaded last.norm explicitly, then the (loaded) head.
        hidden_states = last.norm(hidden_states)
        return self.logits_processor(last.shared_head.head, hidden_states)


@support_torch_compile
class DSparkV4MTP(nn.Module):
    """vLLM model class registered as architecture ``DSparkV4MTP`` and selected
    by ``method=dspark`` (speculative.py hf_config_override)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DSparkV4MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,        # main_x = main_norm(main_proj(aux)) [num_ctx, hidden]
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """Write the DSpark context K/V (from main_x) into each draft stage's KV cache.

        Reference (model.py DSparkAttention prefill, start_pos==0, lines 758-768):
            main_kv = kv_norm(wkv(main_x)); RoPE on last rope_head_dim; act_quant
            the non-rope part; store into the (sliding-window) cache.

        Unlike DFlash (MHA: separate K/V heads -> attn.impl.do_kv_cache_update),
        DeepSeek-V4 attention is **MLA**: a single compressed latent (wkv -> 512-d:
        non-rope + rope_head_dim). The per-stage projection/norm/rope below follows
        the reference; the CACHE WRITE uses this build's MLA attention cache-update
        API (mtp_block.attn.impl.*).

        When context_slot_mapping is None (dummy_run) only compute runs, no write.

        Implementation (R4, grounded in DeepseekV4 wrapper attention.py:586-736,
        SWA-only branch): per draft stage, project main_x via the SAME fused chain
        as normal forward (fused_wqa_wkv -> split -> fused_q_kv_rmsnorm), then call
        the SAME fused insert op with an EXPLICIT context slot_mapping. The op does
        q_head_norm + RoPE + UE8M0 FP8 quant + paged cache insert internally — no
        hand-written kernel. Draft stages are SWA-only (compress_ratio<=1 ->
        compressor/indexer None), so only swa_cache_layer is written.
        """
        # dummy_run (slot_mapping None): skip entirely. DFlash dummy_run feeds an
        # hc_mult-scaled hidden buffer (R8) that doesn't match fused_wqa_wkv's
        # hidden_size input; the real propose path passes main_x [.,hidden] from
        # combine_hidden_states. Context KV is only written during real drafting.
        if context_slot_mapping is None:
            return
        import os
        if os.environ.get("DSPARK_NO_CONTEXT") == "1":
            return  # A/B: skip context-KV write -> draft sees ONLY the block
        from vllm.models.deepseek_v4.common.ops import fused_q_kv_rmsnorm

        _cap = os.environ.get("DSPARK_CAPTURE") == "1"
        _cap_path = "/cache/huggingface/dspark_cap.pt"
        _cap_first = _cap and not os.path.exists(_cap_path)
        for _si, stage in enumerate(self.model.layers.values()):
            mla = stage.mtp_block.attn.mla_attn   # DeepseekV4MLAAttention wrapper
            # project + fused RMSNorm (mirror attention.py:614-623)
            qr_kv, _ = mla.fused_wqa_wkv(context_states)
            qr, kv = qr_kv.split([mla.q_lora_rank, mla.head_dim], dim=-1)
            qr, kv = fused_q_kv_rmsnorm(
                qr, kv, mla.q_norm.weight.data, mla.kv_norm.weight.data, mla.eps,
            )
            # STEP 1 dense-bypass (env DSPARK_DENSE): stash bf16 context KV (post-norm,
            # pre-RoPE) + positions per stage so the draft forward can do a plain-torch
            # dense MLA attention over [context KV + block KV], bypassing decode_dsv4.
            # Off by default → no effect on the working fp8-cache path.
            if os.environ.get("DSPARK_DENSE") == "1":
                if not hasattr(self.model, "_dense_ctx_kv"):
                    self.model._dense_ctx_kv = {}
                self.model._dense_ctx_kv[_si] = kv.detach().to(torch.bfloat16)
                self.model._dense_ctx_pos = context_positions.detach()
            q = mla.wq_b(qr).view(-1, mla.n_local_heads, mla.head_dim)
            swa = mla.swa_cache_layer
            swa2d = swa.kv_cache.view(swa.kv_cache.shape[0], -1)
            block_size = getattr(swa, "block_size", 64)
            if _cap_first and _si == 0:
                try:
                    _attn = stage.mtp_block.attn   # parent of mla_attn; holds attn_sink param
                    _d = {"context_states": context_states.detach().cpu().float(),
                          "kv_postnorm": kv.detach().cpu().float(),
                          "context_positions": context_positions.detach().cpu(),
                          "n_local_heads": mla.n_local_heads, "padded_heads": mla.padded_heads,
                          "compress_ratio": getattr(mla, "compress_ratio", None)}
                    _as = getattr(_attn, "attn_sink", None)
                    if _as is not None:
                        _d["attn_sink"] = _as.detach().cpu().float()
                    torch.save(_d, _cap_path)
                    import sys as _cs; print(f"DSPARK-CAP saved attn_sink={'yes' if _as is not None else 'NONE'}", file=_cs.stderr, flush=True)
                except Exception as _e:
                    import sys as _cs; print(f"DSPARK-CAP failed: {_e}", file=_cs.stderr, flush=True)
            # SAME op as the live forward (attention.py:726-736), explicit ctx slot_mapping
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                q, kv, swa2d, context_slot_mapping,
                context_positions.to(torch.int64),
                mla.rotary_emb.cos_sin_cache, mla.padded_heads, mla.eps, block_size,
            )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        return self.model.compute_block_logits(hidden_states)

    # markov / confidence exposed for DSparkProposer's AR refinement loop
    def markov_step(self, token_ids: torch.Tensor):
        return self.model.last_stage.markov_step(token_ids)

    def predict_confidence(self, hidden: torch.Tensor, markov_embed: torch.Tensor):
        return self.model.last_stage.predict_confidence(hidden, markov_embed)

    # ---------------- weight loading ----------------
    # Reuses the DeepSeekV4MTP machinery (expert mapping, fused stacking,
    # attn_sink TP-narrow, E8M0 scales). Diverges only in spec_layer_weight_names
    # and the fp32 cast for markov/confidence params.

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def _find_mtp_layer_idx(name: str) -> int:
            for subname in name.split("."):
                try:
                    return int(subname)
                except ValueError:
                    continue
            return 0

        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        first_layer = next(iter(self.model.layers.values()))
        if first_layer.mtp_block.ffn.use_mega_moe:
            expert_mapping = make_deepseek_v4_expert_params_mapping(
                self.config.n_routed_experts
            )
        else:
            expert_mapping = FusedMoE.make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.n_routed_experts,
            )

        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )

        # markov_w1/markov_w2/confidence_proj are stored bf16 in the checkpoint
        # but live as fp32 params. Cast on load.
        FP32_CAST_KEYS = ("markov_head.markov_w1", "markov_head.markov_w2",
                          "confidence_head.proj")

        # DIAG: detect FusedMoE silent expert-drop (per-expert weight not matched in
        # expert_mapping -> dropped without warning). seen = expert weight tensors from
        # checkpoint; placed = successfully written into a fused param. placed<seen = bug.
        _dbg_exp_seen = 0
        _dbg_exp_ok = 0
        for name, loaded_weight in weights:
            mtp_layer_idx = _find_mtp_layer_idx(name)
            name = name.replace(
                f"mtp.{mtp_layer_idx}.",
                f"model.layers.{self.config.num_hidden_layers + mtp_layer_idx}.",
            )

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue

            name = self._rewrite_spec_layer_name(spec_layer, name)
            if spec_layer != self.model.mtp_start_layer_idx and ".layers" not in name:
                continue

            if any(k in name for k in FP32_CAST_KEYS) and loaded_weight.dtype != torch.float32:
                loaded_weight = loaded_weight.float()

            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                name = name.removesuffix(".scale") + suffix

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip experts (handled below) AND DSpark head weights — otherwise the
                # "w1"->gate_up_proj substring rule corrupts markov_w1 (markov_gate_up_proj).
                if ".experts." in name or "markov_head" in name or "confidence_head" in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    _dbg_exp_seen += 1
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        param = params_dict[name_mapped]
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param, loaded_weight, name_mapped,
                            shard_id=expert_shard_id, expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            _dbg_exp_ok += 1
                            loaded_params.add(name_mapped)
                            break
                    continue
                elif "attn_sink" in name:
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    if ".shared_experts.w2" in name:
                        name = name.replace(
                            ".shared_experts.w2", ".shared_experts.down_proj"
                        )
                    if name.endswith(".ffn.gate.bias"):
                        name = name.replace(
                            ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                        )
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        # require all 3 stages present
        loaded_layers: set[int] = set()
        for param_name in loaded_params:
            sl = get_spec_layer_idx_from_weight_name(self.config, param_name)
            if sl is not None:
                loaded_layers.add(sl)
        for layer_idx in range(
            self.model.mtp_start_layer_idx,
            self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
        ):
            if layer_idx not in loaded_layers:
                raise ValueError(
                    f"DSpark draft stage {layer_idx} weights missing from "
                    f"checkpoint (expected mtp.* in shards 46-48)."
                )
        self.finalize_mega_moe_weights()
        logger.info(
            "DSpark draft head loaded: %d params (expert weights seen=%d placed=%d%s)",
            len(loaded_params), _dbg_exp_seen, _dbg_exp_ok,
            " !! DROPPED experts -> backbone garbage" if _dbg_exp_ok < _dbg_exp_seen
            else " all-OK",
        )
        return loaded_params

    def finalize_mega_moe_weights(self) -> None:
        for layer in self.model.layers.values():
            layer.mtp_block.ffn.finalize_mega_moe_weights()

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        # DSpark top-level (not inside mtp_block) per-stage weights.
        # main_proj/main_norm -> stage 0; norm/hc_head/markov/confidence/shared_head
        # -> last stage; embed_tokens shared/top-level. Everything else -> mtp_block.
        spec_layer_weight_names = [
            "embed_tokens",
            "main_proj",
            "main_norm",
            "norm",          # last-stage final RMSNorm (NOT attn_norm/ffn_norm)
            "shared_head",
            "markov_head",
            "confidence_head",
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
        ]
        shared_weight_names = ["embed_tokens"]
        # Guard: ".norm." must match the standalone last-stage norm, not
        # attn_norm/ffn_norm/q_norm/kv_norm which belong to mtp_block.
        def _is_spec(n: str) -> str | None:
            for wn in spec_layer_weight_names:
                if wn == "norm":
                    if ".norm.weight" in n and ".attn_norm" not in n \
                            and ".ffn_norm" not in n and "_norm." not in n.replace(".norm.", ".X."):
                        return wn
                elif wn in n:
                    return wn
            return None

        matched = _is_spec(name)
        if matched is None:
            name = name.replace(
                f"model.layers.{spec_layer}.",
                f"model.layers.{spec_layer}.mtp_block.",
            )
        elif matched in shared_weight_names:
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name
