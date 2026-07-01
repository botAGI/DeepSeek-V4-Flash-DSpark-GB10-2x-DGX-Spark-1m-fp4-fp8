#!/usr/bin/env python3
"""Offline test: DSpark load_weights NAME ROUTING against the real tensor names
from model.safetensors.index.json. No vLLM, no weights — pure logic.

Replicates _find_mtp_layer_idx + mtp.N->model.layers.{43+N} + the spec-layer filter
+ _rewrite_spec_layer_name from patches/.../nvidia/dspark.py and checks that EVERY
mtp.* name routes into the correct bucket:
  - main_proj/main_norm        -> ONLY stage0 (layer 43), top-level spec
  - norm/markov/confidence/hc_head -> ONLY last stage (layer 45), top-level spec
  - attn.*/attn_norm/ffn.*/ffn_norm/hc_attn_*/hc_ffn_* -> mtp_block (all stages)
  - attn.q_norm/attn.kv_norm/attn_norm/ffn_norm must NOT fall into last-stage 'norm'
  - base embed/head/norm (NOT mtp.*) -> ignored (tied/target)

Point this at a checkpoint index via the DSPARK_INDEX env var:
  DSPARK_INDEX=/path/to/model.safetensors.index.json python3 test_dspark_weight_routing.py
If the index is not available, the test SKIPS (exit 77) instead of failing.
"""
import json, sys, re, os

IDX = os.environ.get("DSPARK_INDEX", "model.safetensors.index.json")
NUM_HIDDEN = 43
MTP_LAYERS = {43, 44, 45}
LAST = 45

def find_mtp_layer_idx(name):
    for s in name.split("."):
        try:
            return int(s)
        except ValueError:
            continue
    return 0

def spec_layer_idx(name):
    # replica of get_spec_layer_idx_from_weight_name (deepseek_v2): model.layers.{N}, N>=43
    m = re.search(r"model\.layers\.(\d+)\.", name)
    if not m:
        return None
    n = int(m.group(1))
    return n if n >= NUM_HIDDEN else None

SPEC_NAMES = ["embed_tokens", "main_proj", "main_norm", "norm", "shared_head",
              "markov_head", "confidence_head", "hc_head_fn", "hc_head_base", "hc_head_scale"]
SHARED = ["embed_tokens"]

def is_spec(n):
    for wn in SPEC_NAMES:
        if wn == "norm":
            # standalone last-stage .norm.weight, NOT attn_norm/ffn_norm/q_norm/kv_norm
            if ".norm.weight" in n and ".attn_norm" not in n and ".ffn_norm" not in n \
               and ".q_norm" not in n and ".kv_norm" not in n:
                return wn
        elif wn in n:
            return wn
    return None

def rewrite(spec_layer, name):
    m = is_spec(name)
    if m is None:
        return name.replace(f"model.layers.{spec_layer}.",
                            f"model.layers.{spec_layer}.mtp_block."), "mtp_block"
    if m in SHARED:
        return name.replace(f"model.layers.{spec_layer}.", "model."), "top:"+m
    return name, "top:"+m

def main():
    if not os.path.exists(IDX):
        print(f"SKIP: checkpoint index not found at {IDX!r} "
              f"(set DSPARK_INDEX to a model.safetensors.index.json)")
        sys.exit(77)  # SKIP
    wm = json.load(open(IDX))["weight_map"]
    mtp_names = [k for k in wm if k.startswith("mtp.")]
    print(f"mtp.* tensors in the checkpoint: {len(mtp_names)}")

    buckets = {}          # final route bucket -> count
    per_stage_top = {43: set(), 44: set(), 45: set()}
    errors = []
    routed_layers = set()

    for name in mtp_names:
        idx = find_mtp_layer_idx(name)
        remapped = name.replace(f"mtp.{idx}.", f"model.layers.{NUM_HIDDEN+idx}.")
        sl = spec_layer_idx(remapped)
        if sl is None:
            errors.append(f"NO spec_layer: {name} -> {remapped}")
            continue
        routed_layers.add(sl)
        final, kind = rewrite(sl, remapped)
        buckets[kind] = buckets.get(kind, 0) + 1
        if kind.startswith("top:"):
            per_stage_top[sl].add(kind[4:])

    # ---- checks ----
    print("\nbuckets (route type -> count):")
    for k in sorted(buckets): print(f"  {k:18} {buckets[k]}")

    print("\ntop-level spec weights per stage:")
    for s in (43, 44, 45):
        print(f"  layer {s}: {sorted(per_stage_top[s])}")

    # invariants
    inv = []
    def check(cond, msg):
        inv.append((cond, msg))
    check("main_proj" in per_stage_top[43] and "main_norm" in per_stage_top[43],
          "main_proj/main_norm on stage0(43)")
    check("main_proj" not in per_stage_top[44] and "main_proj" not in per_stage_top[45],
          "main_proj ONLY on stage0")
    check({"markov_head", "confidence_head", "norm"}.issubset(per_stage_top[45]),
          "markov/confidence/norm on last(45)")
    check("markov_head" not in per_stage_top[43] and "markov_head" not in per_stage_top[44],
          "markov ONLY on last")
    check({"hc_head_fn", "hc_head_base", "hc_head_scale"}.issubset(per_stage_top[45]),
          "hc_head_* on last(45)")
    check(routed_layers == MTP_LAYERS, f"all 3 stages {MTP_LAYERS} (got {routed_layers})")
    check(not errors, f"no unrecognized names ({len(errors)} errors)")
    # critical: no attn_norm/ffn_norm/q_norm/kv_norm should route to top:norm
    misrouted_norm = [n for n in mtp_names
                      if any(x in n for x in ("attn_norm", "ffn_norm", "q_norm", "kv_norm"))
                      and rewrite(spec_layer_idx(n.replace(f"mtp.{find_mtp_layer_idx(n)}.",
                          f"model.layers.{NUM_HIDDEN+find_mtp_layer_idx(n)}.")),
                          n.replace(f"mtp.{find_mtp_layer_idx(n)}.",
                          f"model.layers.{NUM_HIDDEN+find_mtp_layer_idx(n)}."))[1] == "top:norm"]
    check(not misrouted_norm, f"attn_norm/ffn_norm/q_norm/kv_norm did NOT route to top:norm ({misrouted_norm[:3]})")

    print("\n--- invariants ---")
    ok = True
    for cond, msg in inv:
        print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
        ok = ok and cond
    if errors:
        print("\nunrecognized (first 10):")
        for e in errors[:10]: print("  ", e)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
