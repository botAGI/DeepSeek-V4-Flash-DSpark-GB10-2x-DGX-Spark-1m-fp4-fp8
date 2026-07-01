#!/usr/bin/env bash
# Build a DSpark-enabled vLLM image from a base SM12x DeepSeek-V4 image, WITHOUT
# touching any running container. Copies the two new modules and applies the
# surgical enum/runner/registry edits (PATCHES.md) via an idempotent python patcher.
#
#   bash apply-patches.sh <base-image> <target-image>
set -euo pipefail

BASE="${1:?usage: apply-patches.sh <base-image> <target-image>}"
TARGET="${2:?usage: apply-patches.sh <base-image> <target-image>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PKG="/opt/env/lib/python3.12/site-packages/vllm"   # vLLM package root in the image

BUILD="$(mktemp -d)"; trap 'rm -rf "$BUILD"' EXIT
cp "$HERE/patches/vllm/models/deepseek_v4/nvidia/dspark.py" "$BUILD/dspark_head.py"
cp "$HERE/patches/vllm/spec_decode/dspark.py"               "$BUILD/dspark_proposer.py"
cp "$HERE/patches/PATCHES.md"                               "$BUILD/PATCHES.md"

# --- idempotent in-image patcher (enum + predicates + registry + runner) ---
cat > "$BUILD/patch_vllm_dspark.py" <<'PY'
import re, sys
PKG = "/opt/env/lib/python3.12/site-packages/vllm"

def patch(path, edits):
    with open(path) as f: s = f.read()
    orig = s
    for old, new, once in edits:
        if new in s and once:        # already applied
            continue
        if old not in s:
            print(f"  ! anchor not found in {path}: {old[:60]!r}"); continue
        s = s.replace(old, new, 1)
    if s != orig:
        with open(path, "w") as f: f.write(s)
        print(f"  patched {path}")
    else:
        print(f"  no-op {path}")

sp = f"{PKG}/config/speculative.py"
patch(sp, [
    ('DFlashModelTypes = Literal["dflash"]',
     'DFlashModelTypes = Literal["dflash"]\nDSparkModelTypes = Literal["dspark"]', True),
    ('"eagle", "eagle3", "extract_hidden_states", MTPModelTypes, DFlashModelTypes',
     '"eagle", "eagle3", "extract_hidden_states", MTPModelTypes, DFlashModelTypes, DSparkModelTypes', True),
    ('return self.method in ("eagle", "eagle3", "mtp", "dflash")',
     'return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")', True),
    ('    def use_dflash(self) -> bool:\n        return self.method == "dflash"',
     '    def use_dflash(self) -> bool:\n        return self.method == "dflash"\n\n'
     '    def use_dspark(self) -> bool:\n        return self.method == "dspark"', True),
    # hf_config_override: DSpark checkpoint (has dspark_block_size) -> DSparkV4MTP arch
    ('        if hf_config.model_type == "deepseek_v4":\n'
     '            hf_config.model_type = "deepseek_mtp"\n'
     '            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)\n'
     '            hf_config.update(\n'
     '                {"n_predict": n_predict, "architectures": ["DeepSeekV4MTPModel"]}\n'
     '            )',
     '        if hf_config.model_type == "deepseek_v4":\n'
     '            if getattr(hf_config, "dspark_block_size", None) is not None:\n'
     '                _stages = getattr(hf_config, "n_mtp_layers", 3)\n'
     '                _blk = getattr(hf_config, "dspark_block_size", 5)\n'
     '                hf_config.update({\n'
     '                    "n_predict": _blk, "num_nextn_predict_layers": _blk,\n'
     '                    "dspark_n_mtp_layers": _stages,\n'
     '                    "architectures": ["DSparkV4MTP"],\n'
     '                    "eagle_aux_hidden_state_layer_ids": list(getattr(hf_config, "dspark_target_layer_ids", [40, 41, 42])),\n'
     '                    "dflash_config": {"mask_token_id": getattr(hf_config, "dspark_noise_token_id", 128799), "use_aux_hidden_state": True},\n'
     '                })\n'
     '            else:\n'
     '                hf_config.model_type = "deepseek_mtp"\n'
     '                n_predict = getattr(hf_config, "num_nextn_predict_layers", None)\n'
     '                hf_config.update(\n'
     '                    {"n_predict": n_predict, "architectures": ["DeepSeekV4MTPModel"]}\n'
     '                )', True),
    # __post_init__ method-dispatch: dspark joins eagle/dflash "pass" branch
    # (skip auto-detect + avoid "Unsupported speculative method" at :739).
    # Deliberately NOT added to the EAGLE-wrap branch (:743) — preserve DSparkV4MTP cfg.
    ('if self.method in ("eagle", "eagle3", "dflash"):\n                    pass',
     'if self.method in ("eagle", "eagle3", "dflash", "dspark"):\n                    pass', True),
    # parallel_drafting on, like dflash
    ('if self.method == "dflash":\n                    self.parallel_drafting = True',
     'if self.method in ("dflash", "dspark"):\n                    self.parallel_drafting = True', True),
])

# runner: import + dispatch (insert dspark branch ABOVE use_dflash branch)
gr = f"{PKG}/v1/worker/gpu_model_runner.py"
patch(gr, [
    ('from vllm.v1.spec_decode.dflash import DFlashProposer',
     'from vllm.v1.spec_decode.dflash import DFlashProposer\n'
     'from vllm.v1.spec_decode.dspark import DSparkProposer', True),
    # dispatch: DSpark branch BEFORE use_eagle() (use_eagle now matches dspark too).
    # isinstance(.., DFlashProposer) unions auto-cover DSparkProposer (subclass).
    ('            elif self.speculative_config.use_dflash():\n'
     '                self.drafter = DFlashProposer(self.vllm_config, self.device, self)\n'
     '                self.use_aux_hidden_state_outputs = True',
     '            elif self.speculative_config.use_dflash():\n'
     '                self.drafter = DFlashProposer(self.vllm_config, self.device, self)\n'
     '                self.use_aux_hidden_state_outputs = True\n'
     '            elif self.speculative_config.use_dspark():\n'
     '                self.drafter = DSparkProposer(self.vllm_config, self.device, self)\n'
     '                self.use_aux_hidden_state_outputs = True', True),
])

# registry: map DSparkV4MTP arch
reg = f"{PKG}/model_executor/models/registry.py"
patch(reg, [
    ('"DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM")',
     '"DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM"),\n'
     '    "DSparkV4MTP": ("vllm.models.deepseek_v4.nvidia.dspark", "DSparkV4MTP")', True),
])

# R3: EAGLE3 aux-hidden interface on the BASE DeepseekV4 model so the target emits
# hidden states from dspark_target_layer_ids ([40,41,42]) for the DSpark draft.
mdl = f"{PKG}/models/deepseek_v4/nvidia/model.py"
patch(mdl, [
    # import official EAGLE3 interface (proper pattern, like llama)
    ('from vllm.model_executor.models.interfaces import SupportsPP',
     'from vllm.model_executor.models.interfaces import SupportsPP, SupportsEagle3', True),
    # capture aux hidden states in the decoder loop (mean over hc_mult -> [T, dim])
    # In the BASE model hc_post is called ONCE after the loop (on the last layer),
    # NOT per-layer. The reference aux (model.py:921) is per-layer h.mean(dim=2) of
    # the hc-FOLDED hidden. So at each aux layer we call layer.hc_post(...) on a
    # throwaway (its 4 per-layer outputs) -> [T, hc_mult, dim], then mean over hc ->
    # 2D [T, dim]. The main `hidden_states` is left UNtouched (base still does its
    # single post-loop hc_post). Runner cats the 2D aux list -> [T, 3*dim] (4952).
    ('        residual, post_mix, res_mix = None, None, None\n'
     '        for layer in islice(self.layers, self.start_layer, self.end_layer):\n'
     '            hidden_states, residual, post_mix, res_mix = layer(\n'
     '                hidden_states,\n'
     '                positions,\n'
     '                input_ids,\n'
     '                post_mix,\n'
     '                res_mix,\n'
     '                residual,\n'
     '            )',
     '        residual, post_mix, res_mix = None, None, None\n'
     '        _dspark_aux = []\n'
     '        _dspark_aux_layers = getattr(self, "aux_hidden_state_layers", ())\n'
     '        for _dspark_i, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):\n'
     '            hidden_states, residual, post_mix, res_mix = layer(\n'
     '                hidden_states,\n'
     '                positions,\n'
     '                input_ids,\n'
     '                post_mix,\n'
     '                res_mix,\n'
     '                residual,\n'
     '            )\n'
     '            if _dspark_aux_layers and (self.start_layer + _dspark_i) in _dspark_aux_layers:\n'
     '                _dspark_aux.append(layer.hc_post(hidden_states, residual, post_mix, res_mix).mean(dim=1))', True),
    # return (hidden, aux) when aux layers configured
    ('        hidden_states = self.norm(hidden_states)\n'
     '        return hidden_states',
     '        hidden_states = self.norm(hidden_states)\n'
     '        if getattr(self, "aux_hidden_state_layers", ()):\n'
     '            return hidden_states, _dspark_aux\n'
     '        return hidden_states', True),
    # EAGLE3 interface on ForCausalLM
    ('class DeepseekV4ForCausalLM(nn.Module, SupportsPP):\n'
     '    model_cls = DeepseekV4Model',
     'class DeepseekV4ForCausalLM(nn.Module, SupportsPP, SupportsEagle3):\n'
     '    model_cls = DeepseekV4Model\n'
     '    supports_eagle3 = True\n'
     '    def set_aux_hidden_state_layers(self, layers):\n'
     '        self.model.aux_hidden_state_layers = tuple(layers)\n'
     '    def get_eagle3_default_aux_hidden_state_layers(self):\n'
     '        return tuple(getattr(self.config, "dspark_target_layer_ids", (40, 41, 42)))', True),
])

# DIAG (temporary): dump draft-layer -> kv_cache_group mapping + bypass assert to
# observe the real group split AND the next wall in one load.
lbp = f"{PKG}/v1/spec_decode/llm_base_proposer.py"
patch(lbp, [
    ('        assert (\n'
     '            len(\n'
     '                set(\n'
     '                    [\n'
     '                        kv_cache_groups[layer_name]\n'
     '                        for layer_name in self._draft_attn_layer_names\n'
     '                    ]\n'
     '                )\n'
     '            )\n'
     '            == 1\n'
     '        ), "All drafting layers should belong to the same kv cache group"',
     '        import sys as _sys\n'
     '        _gids = {ln: kv_cache_groups.get(ln) for ln in self._draft_attn_layer_names}\n'
     '        print("DSPARK-DIAG draft_attn_layers:", _gids, file=_sys.stderr, flush=True)\n'
     '        for _gi, _gg in enumerate(kv_cache_config.kv_cache_groups):\n'
     '            print("DSPARK-DIAG group", _gi, type(_gg.kv_cache_spec).__name__, "n_layers", len(_gg.layer_names), "sample", list(_gg.layer_names)[:4], file=_sys.stderr, flush=True)\n'
     '        if len(set(_gids.values())) != 1:\n'
     '            print("DSPARK-DIAG WARN draft spans", len(set(_gids.values())), "groups - bypassing assert for diagnosis", file=_sys.stderr, flush=True)', True),
])

# STEP 2 dense-bypass A/B (env DSPARK_DENSE=1): replace decode_dsv4 (_wrapper.run) in
# forward_mqa with a plain-torch sparse-MLA softmax. Decisive kernel-vs-fp4 test;
# target self-check first (target must stay coherent => dense math correct).
sm = f"{PKG}/v1/attention/backends/mla/sparse_mla_sm120.py"
patch(sm, [
    ('        output = q.new_empty(\n'
     '            (num_actual_toks, self.num_heads, self.kv_lora_rank),\n'
     '            dtype=q.dtype,\n'
     '        )',
     '        output = q.new_empty(\n'
     '            (num_actual_toks, self.num_heads, self.kv_lora_rank),\n'
     '            dtype=q.dtype,\n'
     '        )\n'
     '        import os as _dos\n'
     '        if _dos.environ.get("DSPARK_DENSE") == "1":\n'
     '            _c = kv_c_and_k_pe_cache.view(torch.uint8).reshape(-1, 656)\n'
     '            _T, _H, _qd = q.shape\n'
     '            _idx = topk_indices_physical\n'
     '            _tk = _idx.shape[-1]\n'
     '            _v = _idx >= 0\n'
     '            _s = _c[_idx.clamp(min=0).long().reshape(-1)].reshape(_T, _tk, 656)\n'
     '            _kn = (_s[..., 0:512].view(torch.float8_e4m3fn).float().reshape(_T, _tk, 4, 128) * _s[..., 512:528].contiguous().view(torch.float32)[..., None]).reshape(_T, _tk, 512)\n'
     '            _kr = _s[..., 528:656].contiguous().view(torch.bfloat16).float()\n'
     '            _qf = q.float()\n'
     '            _qn = _qf[..., 0:512]\n'
     '            _qr = _qf[..., 512:576]\n'
     '            _sc = (torch.einsum("thd,tkd->thk", _qn, _kn) + torch.einsum("thd,tkd->thk", _qr, _kr)) * self.scale\n'
     '            _sc = _sc.masked_fill(~_v[:, None, :], float("-inf"))\n'
     '            _sk = layer.attn_sink.float().reshape(-1)[:_H]\n'
     '            _m = torch.maximum(_sc.amax(-1, keepdim=True), _sk[None, :, None])\n'
     '            _ex = torch.exp(_sc - _m) * _v[:, None, :].float()\n'
     '            _dn = _ex.sum(-1, keepdim=True) + torch.exp(_sk[None, :, None] - _m)\n'
     '            _o = torch.einsum("thk,tkd->thd", _ex / _dn, _kn)\n'
     '            output.copy_(_o.to(output.dtype))\n'
     '            return output, None', True),
])

# 🎯 STABILITY FIX (multi-agent root-cause 29.06): concurrency crash
# "extra_kv_cache requires extra_indices". Two metadata builders serving the SAME
# target-verify forward disagree on the decode/prefill split: SWA uses 1+num_spec(=6),
# C128A uses parallel_drafting-doubled reorder_batch_threshold(=11). A 7..11-token
# chunked-prefill extend co-occurring with verifies under concurrency -> SWA enters
# _forward_prefill but C128A left c128a_prefill_topk_indices=None -> flashinfer ICHECK.
# Prod MTP-2 is immune (parallel_drafting=False -> both thresholds equal). Fix: align
# C128A split to SWA (1+num_spec); the 2x is only for the DRAFT proposer batch.
fms = f"{PKG}/v1/attention/backends/mla/flashmla_sparse.py"
patch(fms, [
    ('        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)',
     '        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)\n'
     '        # DSpark: C128A topk split must match SWA builder (1+num_spec), which\n'
     '        # decides _forward_decode vs _forward_prefill. parallel_drafting doubles\n'
     '        # reorder_batch_threshold for the DRAFT batch; the TARGET verify has only\n'
     '        # 1+num_spec query tokens/req. Misalignment -> c128a_prefill_topk_indices\n'
     '        # None on a 7..11 chunked-prefill extend -> "extra_kv_cache requires\n'
     '        # extra_indices" crash under concurrency.\n'
     '        _spec = vllm_config.speculative_config\n'
     '        if (_spec is not None and getattr(_spec, "parallel_drafting", False)\n'
     '                and _spec.num_speculative_tokens is not None):\n'
     '            self._c128a_split_threshold = 1 + _spec.num_speculative_tokens\n'
     '        else:\n'
     '            self._c128a_split_threshold = self.reorder_batch_threshold or 1', True),
    ('                cm,\n'
     '                decode_threshold=self.reorder_batch_threshold or 1,\n'
     '            )',
     '                cm,\n'
     '                decode_threshold=getattr(\n'
     '                    self, "_c128a_split_threshold",\n'
     '                    self.reorder_batch_threshold or 1),\n'
     '            )', True),
])

print("NOTE: remaining runner edits (isinstance unions @ ~2421/4421/4892/5856/6696,"
      " dispatch elif use_dspark() @ ~584, hf_config_override architectures=['DSparkV4MTP'])"
      " — see PATCHES.md; verify against the loaded model.")
PY

cat > "$BUILD/Dockerfile" <<DOCKER
FROM ${BASE}
COPY dspark_head.py     ${PKG}/models/deepseek_v4/nvidia/dspark.py
COPY dspark_proposer.py ${PKG}/v1/spec_decode/dspark.py
COPY patch_vllm_dspark.py /tmp/patch_vllm_dspark.py
COPY PATCHES.md /opt/DSPARK-PATCHES.md
RUN python /tmp/patch_vllm_dspark.py
DOCKER

echo "Building ${TARGET} from ${BASE} ..."
docker build -t "${TARGET}" "$BUILD"
echo "Done. Derived image: ${TARGET}  (prod image ${BASE} untouched)"
