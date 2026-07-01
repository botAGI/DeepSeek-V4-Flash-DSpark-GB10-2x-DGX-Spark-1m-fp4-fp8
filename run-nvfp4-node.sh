#!/usr/bin/env bash
# Sanitized two-node launcher for OUR reproduction of the NVFP4-KV DSpark build.
# The runtime image is tonyd2wild's Stage-C (`nvfp4_ds_mla` KV, 584-byte padded
# sparse-MLA envelope) — credit: see CREDITS.md. This launcher only wraps it with
# peer-wait + RoCE-GID re-alignment for a robust two-node bring-up; it does NOT
# change the build. Fill in <MASTER_IP>/<PEER_IP>/<RDV_PORT>/<ROCE_IFACE>/<ETH_IFACE>
# for your own cluster before use.
#
# usage: run-nvfp4-node.sh <0|1> [extra serve args...]
#   0 = head (OpenAI API on :8000), 1 = worker (headless)
# Env knobs:
#   MAX_MODEL_LEN (default 1048576), MAX_NUM_SEQS (default 16),
#   GPU_MEM_UTIL (default 0.80), MTP_NUM_TOKENS (default 5),
#   NO_SPEC=1 -> drop --speculative-config
set -uo pipefail

NODE_RANK="${1:?usage: run-nvfp4-node.sh <0|1> [extra serve args]}"
shift || true
EXTRA_ARGS=("$@")

IMAGE="vllm-dspark-runtime:dspark-nvfp4-stage-c"
CONTAINER="dspark-vllm"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
NCCL_PROTO_OPT="${NCCL_PROTO_OPT:-}"
NCCL_NCH_OPT="${NCCL_NCH_OPT:-}"
MTP_NUM_TOKENS="${MTP_NUM_TOKENS:-5}"
NO_SPEC="${NO_SPEC:-0}"

# Cluster topology — set these to your own RoCE/IB addresses.
MASTER_IP="${MASTER_IP:-<MASTER_IP>}"
RDV_PORT="${RDV_PORT:-<RDV_PORT>}"
ROCE_IFACE="${ROCE_IFACE:-<ROCE_IFACE>}"   # RoCE HCA, e.g. rocepXsYfZ
ETH_IFACE="${ETH_IFACE:-<ETH_IFACE>}"      # socket NIC, e.g. enpXsYfZnpN

HEADLESS_FLAG=""
PEER_IP="${PEER_IP:-<PEER_IP>}"
OWN_IP="$MASTER_IP"
HOST_BIND="0.0.0.0"
if [ "$NODE_RANK" = "1" ]; then
  HEADLESS_FLAG="--headless"
  PEER_IP="$MASTER_IP"
  OWN_IP="${OWN_IP_WORKER:-<PEER_IP>}"
fi

# Locate HF cache holding the DSpark weights on THIS node.
HF_CACHE=""
for c in "$HOME/.cache/huggingface" /root/.cache/huggingface; do
  if [ -d "$c/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark" ]; then HF_CACHE="$c"; break; fi
done
: "${HF_CACHE:?DeepSeek-V4-Flash-DSpark weights not found in any HF cache dir}"

echo "nvfp4: node-rank=$NODE_RANK cache=$HF_CACHE seqs=$MAX_NUM_SEQS len=$MAX_MODEL_LEN no_spec=$NO_SPEC — waiting for peer $PEER_IP ..."
until ping -c1 -W1 "$PEER_IP" >/dev/null 2>&1; do sleep 3; done
echo "nvfp4: peer $PEER_IP reachable"

# Re-align RoCE GID index 3 (RoCE v2) to our QSFP IPv4. On these clusters the GID
# index can drift after a link flap, which silently breaks the TP rendezvous.
ensure_roce_gid() {
  command -v show_gids >/dev/null 2>&1 || { echo "nvfp4: show_gids absent — skip RoCE check"; return 0; }
  for try in 1 2 3 4 5; do
    if show_gids 2>/dev/null | awk -v ip="$OWN_IP" -v hca="$ROCE_IFACE" '$1==hca&&$3=="3"&&$5==ip&&$6=="v2"{f=1} END{exit !f}'; then
      echo "nvfp4: RoCE GID idx3=${OWN_IP}/v2 — ok"; return 0
    fi
    echo "nvfp4: RoCE GID idx3 drifted (try $try/5) — reactivating link"
    nmcli con down qsfp-link >/dev/null 2>&1 || true; sleep 2
    nmcli con up   qsfp-link >/dev/null 2>&1 || true; sleep 6
  done
  echo "nvfp4: WARN RoCE GID still not aligned after 5 tries — launching anyway"
}
ensure_roce_gid

# Build extra serve args as a plain string (empty when no-spec + no extras),
# so an empty array never leaves a stray '' token that vLLM rejects.
EXTRA_SERVE=""
if [ "$NO_SPEC" != "1" ]; then
  # Single-quote the JSON inside the bash -lc string: braces+commas would
  # otherwise trigger brace-expansion and mangle the speculative config.
  EXTRA_SERVE="--speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":${MTP_NUM_TOKENS}}'"
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  EXTRA_SERVE="$EXTRA_SERVE ${EXTRA_ARGS[*]}"
fi

echo "nvfp4: launching $CONTAINER (node-rank $NODE_RANK)"
docker rm -f "$CONTAINER" 2>/dev/null || true

# Build the serve command (recipe image uses /opt/env/bin/vllm serve).
# Full PATH/CUDA env from the recipe compose: FlashInfer JIT calls nvcc which
# needs cicc from nvvm/bin on PATH (else 'sh: cicc: not found' -> ninja fail).
SERVE="export PATH=/opt/env/bin:/opt/env/nvvm/bin:/opt/env/targets/sbsa-linux/nvvm/bin:\${PATH:-}; \
export CUDA_HOME=\${CUDA_HOME:-/opt/env/targets/sbsa-linux}; \
export CUDA_PATH=\${CUDA_PATH:-\$CUDA_HOME}; \
export CUDAToolkit_ROOT=\${CUDAToolkit_ROOT:-\$CUDA_HOME}; \
export LD_LIBRARY_PATH=/opt/env/lib:/opt/env/targets/sbsa-linux/lib:\${LD_LIBRARY_PATH:-}; \
exec /opt/env/bin/vllm serve deepseek-ai/DeepSeek-V4-Flash-DSpark \
--served-model-name dspark deepseek-v4-flash-dspark --host ${HOST_BIND} --port 8000 --trust-remote-code \
--tensor-parallel-size 2 --pipeline-parallel-size 1 \
--kv-cache-dtype nvfp4_ds_mla --block-size 256 \
--max-model-len ${MAX_MODEL_LEN} --max-num-seqs ${MAX_NUM_SEQS} \
--max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} --gpu-memory-utilization ${GPU_MEM_UTIL} \
--tokenizer-mode deepseek_v4 --distributed-executor-backend mp \
--tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
${EXTRA_SERVE} \
--nnodes 2 --node-rank ${NODE_RANK} --master-addr ${MASTER_IP} --master-port ${RDV_PORT} ${HEADLESS_FLAG}"

exec docker run --gpus all --privileged --network host --ipc host --shm-size 10g \
  --ulimit memlock=-1 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HF_CACHE:/cache/huggingface" \
  --name "$CONTAINER" \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_USE_B12X_MOE=1 -e VLLM_USE_B12X_WO_PROJECTION=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 -e VLLM_TRITON_MLA_SPARSE=1 \
  -e VLLM_SKIP_INIT_MEMORY_CHECK=1 \
  -e VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0 -e VLLM_DSPARK_CONFIDENCE_SCHEDULER=off \
  -e VLLM_DSPARK_LOCAL_ARGMAX=1 -e VLLM_DSPARK_REPLICATE_MARKOV_W1=1 \
  -e VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0 -e VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1 \
  -e VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0 -e VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=1 \
  -e VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM=0 -e VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M=16 \
  -e B12X_W4A16_TC_DECODE=0 \
  -e VLLM_DSV4_B12X_COMPRESSED_MLA=0 -e VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0 \
  -e VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT=0 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e TILELANG_CLEANUP_TEMP_FILES=1 \
  -e DG_JIT_USE_NVRTC=0 -e DG_JIT_NVCC_COMPILER=/opt/env/bin/nvcc \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="${ROCE_IFACE}" -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME="${ETH_IFACE}" -e GLOO_SOCKET_IFNAME="${ETH_IFACE}" -e TP_SOCKET_IFNAME="${ETH_IFACE}" \
  -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_DEBUG=WARN -e NCCL_NVLS_ENABLE=0 -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
  ${NCCL_PROTO_OPT:+-e NCCL_PROTO=$NCCL_PROTO_OPT} ${NCCL_NCH_OPT:+-e NCCL_MIN_NCHANNELS=$NCCL_NCH_OPT} \
  --entrypoint bash \
  "$IMAGE" \
  -lc "$SERVE"
