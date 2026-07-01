# DSpark — surgical edits to existing vLLM-fork files

Base image `aidendle94/sparkrun-vllm-ds4-gb10`, vLLM `0.21.1rc1.dev339`. Line numbers are
relative to the live package. Apply to a **copy** of the image; never modify a running engine.

## 1. `vllm/config/speculative.py` (enum + predicates)

```python
# :55  after  DFlashModelTypes = Literal["dflash"]
DSparkModelTypes = Literal["dspark"]
# :56-58  EagleModelTypes — add DSparkModelTypes to the union
EagleModelTypes = Literal[
    "eagle", "eagle3", "extract_hidden_states", MTPModelTypes, DFlashModelTypes, DSparkModelTypes
]
# :279  uses_aux_hidden_states = self.method in ( ... , "dspark")
# :688  if self.method in ("eagle","eagle3","dflash","dspark"):
# :732  if self.method in ("eagle","eagle3","dflash","dspark"):
# :752  if self.method in ("dflash","dspark"): self.parallel_drafting = True
# :~698 __post_init__ auto-detect:
#       elif "dspark" in model.lower(): method = "dspark"
# :1059-1060
def use_eagle(self) -> bool:
    return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")
# :1062  new predicate (copy of use_dflash)
def use_dspark(self) -> bool:
    return self.method == "dspark"
# hf_config_override (300-514): when method=="dspark" and model_type=="deepseek_v4"
#   set architectures=["DSparkV4MTP"] (NOT DeepSeekV4MTP/deepseek_mtp) so the resolver
#   picks our dspark.py. + move dspark_* fields into the draft hf_config
#   + dflash_config={"mask_token_id":128799,"use_aux_hidden_state":True} so
#   _init_parallel_drafting_params (316-335) finds parallel_drafting_token_id.
```

## 2. `vllm/v1/worker/gpu_model_runner.py` (dispatcher)

```python
# :176  import
from vllm.v1.spec_decode.dspark import DSparkProposer
# :584  instantiation — ABOVE the use_dflash() branch (order matters: use_eagle()
#       is now True for dspark, otherwise it would be caught earlier)
elif self.speculative_config.use_dspark():
    self.drafter = DSparkProposer(self.vllm_config, self.device, self)
    self.use_aux_hidden_state_outputs = True
# isinstance unions — add DSparkProposer:
#   :2421 (spec_decode_common_attn_metadata), :4421 (use_gpu_toks),
#   :4892 (propose_draft_token_ids), :5856 (dummy_run), :6696 (initialize_attn_backend)
# :6744-6750 cudagraph — DO NOT TOUCH (enforce-eager; the markov AR loop
#   risks a CUDAGraph deadlock on GB10 / driver 580).
```

## 3. Model registration

```python
# vllm/models/deepseek_v4/__init__.py  (or nvidia/__init__.py) — export:
from .nvidia.dspark import DSparkV4MTP
# model registry (registry.py): "DSparkV4MTP" -> ("vllm.models.deepseek_v4", "DSparkV4MTP")
```

## 4. Draft-model config

The checkpoint `config.json` already carries `dspark_block_size=5 / dspark_noise_token_id=128799 /
dspark_target_layer_ids=[40,41,42] / dspark_markov_rank=256 / num_nextn_predict_layers=1`
(but there are really 3 mtp stages — `n_mtp_layers=3` in the inference config). Verify that
`transformers_utils/configs/deepseek_v4.py` parses `dspark_*` (otherwise add them as attributes).

## Launch (window with the previous engine stopped)

```
--speculative-config '{"method":"dspark","num_speculative_tokens":5}' --enforce-eager
```
The draft loads from the FULL checkpoint by the `mtp.*` namespace (no separate extract needed);
`model` in the spec-config = path to `DeepSeek-V4-Flash-DSpark`.

## 5. Additional notes from solo research — required

- **R1:** in the draft hf_config_override set `n_mtp_layers=3` (the root config gives
  `num_nextn_predict_layers=1`, which is WRONG — there are 3 stages). The head already
  defaults to 3, but make it explicit.
- **R3 (base patch!):** the target `DeepseekV4Model`/`DeepseekV4ForCausalLM` does NOT implement
  `set_aux_hidden_state_layers` / `get_eagle3_default_aux_hidden_state_layers`. These must be
  ADDED to the BASE model (emit hidden states from layers [40,41,42]) — otherwise the proposer
  gets no aux input. + set `eagle_aux_hidden_state_layer_ids=[40,41,42]` in the draft hf_config.
  This widens the patch beyond draft-only.
- **R4:** `precompute_and_store_context_kv` — under `DeepseekV4SparseMLAAttentionImpl` + SWA +
  `fp8_ds_mla` (attention.py:829+), not MHA. Finalized against the loaded model.
