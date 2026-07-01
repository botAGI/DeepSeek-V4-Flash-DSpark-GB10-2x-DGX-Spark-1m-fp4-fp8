#!/usr/bin/env bash
# OUR fp8-KV DSpark build — dual-Spark (GB10) load TEST runner. Clone of run-fp8-node.sh.
# usage: launch-fp8-node.sh <0|1>   (0=head/API :8000, 1=worker/headless)
# Detached (-d) so we can tail logs. Test config: small ctx, enforce-eager, no prefix-cache.
#
# Config via env (see .env.dspark.example): MASTER_ADDR, WORKER_HOST, NCCL_IB_HCA,
# NCCL_IB_GID_INDEX, RDV_PORT, HF_CACHE.
set -uo pipefail
NODE_RANK="${1:?usage: launch-fp8-node.sh <0|1>}"

MASTER_IP="${MASTER_ADDR:-<MASTER_IP>}"
PEER_NODE_IP="${WORKER_HOST:-<PEER_IP>}"
RDV_PORT="${RDV_PORT:-<RDV_PORT>}"
ROCE_HCA="${NCCL_IB_HCA:-<ROCE_IFACE>}"
ROCE_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
ETH_IFACE="${ETH_IFACE:-<ETH_IFACE>}"
RDV_LINK="${RDV_LINK:-qsfp-link}"   # nmcli connection name for the cabled RoCE link

HEADLESS_FLAG=""; PEER_IP="$PEER_NODE_IP"; OWN_IP="$MASTER_IP"
if [ "$NODE_RANK" = "1" ]; then HEADLESS_FLAG="--headless"; PEER_IP="$MASTER_IP"; OWN_IP="$PEER_NODE_IP"; fi

HF_CACHE_DIR=""
for c in "${HF_CACHE:-}" "$HOME/.cache/huggingface" "/root/.cache/huggingface"; do
  [ -n "$c" ] || continue
  if [ -d "$c/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark" ]; then HF_CACHE_DIR="$c"; break; fi
done
: "${HF_CACHE_DIR:?DSpark weights not found}"
echo "dspark: node-rank=$NODE_RANK cache=$HF_CACHE_DIR — waiting for peer $PEER_IP ..."
until ping -c1 -W1 "$PEER_IP" >/dev/null 2>&1; do sleep 3; done

ensure_roce_gid() {
  command -v show_gids >/dev/null 2>&1 || return 0
  for try in 1 2 3 4 5; do
    if show_gids 2>/dev/null | awk -v ip="$OWN_IP" -v hca="$ROCE_HCA" -v gi="$ROCE_GID_INDEX" \
        '$1==hca&&$3==gi&&$5==ip&&$6=="v2"{f=1} END{exit !f}'; then return 0; fi
    nmcli con down "$RDV_LINK" >/dev/null 2>&1 || true; sleep 2; nmcli con up "$RDV_LINK" >/dev/null 2>&1 || true; sleep 6
  done
}
ensure_roce_gid

docker rm -f dspark-vllm 2>/dev/null || true
docker run -d --gpus all --privileged --network host --ipc host --shm-size 10g \
  --ulimit memlock=-1 --device /dev/infiniband:/dev/infiniband \
  -v "$HF_CACHE_DIR:/cache/huggingface" --name dspark-vllm \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_USE_B12X_MOE=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e VLLM_TRITON_MLA_SPARSE=1 -e DSPARK_BONUS=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$ROCE_HCA" -e NCCL_IB_GID_INDEX="$ROCE_GID_INDEX" \
  -e NCCL_SOCKET_IFNAME="$ETH_IFACE" -e GLOO_SOCKET_IFNAME="$ETH_IFACE" -e TP_SOCKET_IFNAME="$ETH_IFACE" \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  --entrypoint bash aidendle94/sparkrun-vllm-ds4-gb10:dspark \
  -lc "exec /usr/local/bin/dsv4-vllm-entrypoint serve deepseek-ai/DeepSeek-V4-Flash-DSpark --served-model-name dspark --host 0.0.0.0 --port 8000 --trust-remote-code --tensor-parallel-size 2 --pipeline-parallel-size 1 --kv-cache-dtype fp8 --block-size 256 --max-model-len 1000000 --max-num-seqs 4 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.86 --speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"model\":\"deepseek-ai/DeepSeek-V4-Flash-DSpark\"}' --tokenizer-mode deepseek_v4 --distributed-executor-backend mp --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --nnodes 2 --node-rank $NODE_RANK --master-addr $MASTER_IP --master-port $RDV_PORT $HEADLESS_FLAG"
echo "dspark: dspark-vllm launched (node-rank $NODE_RANK)"
