#!/usr/bin/env bash
# Fetch + verify the DeepSeek-V4-Flash-DSpark checkpoint into the local Hugging Face
# cache, so the node launchers (run-fp8-node.sh / run-nvfp4-node.sh) can find the
# weights. Run this ONCE on each node (the ~156 GiB checkpoint must be present locally
# on every node in the TP=2 group).
#
# The download + shard-completeness-verify approach is adapted from tonyd2wild's
# prepare-dspark-model-cache.sh
# (https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark) —
# see CREDITS.md. The verify logic (index weight_map vs present shards) is tested;
# the download uses huggingface_hub.snapshot_download.
#
# usage: ./prepare-model-cache.sh [REPO_ID]      (default: deepseek-ai/DeepSeek-V4-Flash-DSpark)
#        HF_HOME / HF_TOKEN are honoured. Needs: pip install huggingface_hub
set -euo pipefail

REPO_ID="${1:-deepseek-ai/DeepSeek-V4-Flash-DSpark}"

python3 - "$REPO_ID" <<'PY'
import sys, os, json
from huggingface_hub import snapshot_download

repo = sys.argv[1]
cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
print(f"-> downloading {repo} into the HF cache ({cache}) ...")
path = snapshot_download(repo_id=repo)
print(f"   cached at: {path}")

# Shard-completeness gate: every shard referenced by the index must be on disk.
idx = os.path.join(path, "model.safetensors.index.json")
if not os.path.exists(idx):
    print("   (no sharded index — single-file model, nothing to cross-check)")
else:
    shards = sorted(set(json.load(open(idx))["weight_map"].values()))
    missing = [s for s in shards if not os.path.exists(os.path.join(path, s))]
    if missing:
        sys.exit(f"   INCOMPLETE: {len(missing)}/{len(shards)} shards missing "
                 f"(e.g. {missing[:3]}). Re-run to resume the download.")
    print(f"   ok: all {len(shards)} weight shards present")

print("model cache OK")
PY
