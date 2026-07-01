#!/usr/bin/env bash
# DSpark (DeepSeek-V4-Flash-DSpark, spec-decode) dual-Spark (GB10) vLLM node runner.
# Production service runner — systemd-managed, FOREGROUND (no -d), so systemd owns the
# container lifecycle and Restart=always works.
#
# usage: run-dspark-node.sh <0|1>   (0 = head / OpenAI API :8000, 1 = worker / headless)
#
# Promoted from the verified-working manual launch (launch-dspark-node.sh): TP=2,
# method=dspark, num_spec=5, kv-cache fp8, block-size 256, max-model-len 1000000,
# max-num-seqs 12, gpu-mem-util 0.86, DSPARK_BONUS bonus-slot default-on. Served on the
# SAME host:port (<MASTER_IP>:8000) and served-model-name (dspark) as the running
# engine, so downstream clients are untouched.
#
# Robustness:
#   - HF cache auto-detected (path differs per node).
#   - Waits for the partner node's RoCE fabric IP before NCCL rendezvous, so a
#     simultaneous (incl. dirty) reboot of both Sparks converges without manual ordering.
#   - Re-aligns the RoCE GID after a link flap so NCCL TP=2 reliably forms
#     (the "one node up, the other not" symptom).
#
# max-num-seqs 12: concurrency lever (DSpark strength). Tuned 2026-06-30 — seqs=4 c8=69 / c16=75 t/s;
# seqs=12 c12=145 / c16=117 t/s (+40-90% over seqs=4); acceptance ~2.4 healthy, no OOM.
#
# Config via the cluster env (see .env.dspark.example):
#   MASTER_ADDR, WORKER_HOST, NCCL_IB_HCA, NCCL_IB_GID_INDEX, RDV_PORT, HF_CACHE.
set -uo pipefail

NODE_RANK="${1:?usage: run-dspark-node.sh <0|1>  (0=head, 1=worker)}"

# RoCE fabric IPs of the two nodes (override via env / .env).
MASTER_IP="${MASTER_ADDR:-<MASTER_IP>}"
PEER_NODE_IP="${WORKER_HOST:-<PEER_IP>}"
RDV_PORT="${RDV_PORT:-<RDV_PORT>}"

# RoCE / fabric interface names (override for your hardware).
ROCE_HCA="${NCCL_IB_HCA:-<ROCE_IFACE>}"
ROCE_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
ETH_IFACE="${ETH_IFACE:-<ETH_IFACE>}"
RDV_LINK="${RDV_LINK:-qsfp-link}"   # nmcli connection name for the cabled RoCE link

HEADLESS_FLAG=""
PEER_IP="$PEER_NODE_IP"             # rank 0 (head) waits for the worker
OWN_IP="$MASTER_IP"                 # this node's RoCE fabric IP
if [ "$NODE_RANK" = "1" ]; then
  HEADLESS_FLAG="--headless"
  PEER_IP="$MASTER_IP"             # rank 1 (worker) waits for the head
  OWN_IP="$PEER_NODE_IP"
fi

# Locate the HF cache that actually holds the DSpark weights on THIS node.
HF_CACHE_DIR=""
for c in "${HF_CACHE:-}" "$HOME/.cache/huggingface" "/root/.cache/huggingface"; do
  [ -n "$c" ] || continue
  if [ -d "$c/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark" ]; then HF_CACHE_DIR="$c"; break; fi
done
: "${HF_CACHE_DIR:?DeepSeek-V4-Flash-DSpark weights not found in any HF cache dir}"

echo "dspark: node-rank=$NODE_RANK cache=$HF_CACHE_DIR — waiting for peer $PEER_IP ..."
until ping -c1 -W1 "$PEER_IP" >/dev/null 2>&1; do sleep 3; done
echo "dspark: peer $PEER_IP reachable"

# Ensure the cabled RoCE HCA exposes our fabric IPv4 at the configured GID index (RoCE v2).
# A dirty reboot can flap the RoCE link and shift the GID; NCCL_IB_GID_INDEX then points
# at an empty slot, the NCCL QP handshake dies, and one node never joins TP=2.
# Reactivate the link until the index is our own IPv4/v2 again.
ensure_roce_gid() {
  command -v show_gids >/dev/null 2>&1 || { echo "dspark: show_gids absent — skip RoCE check"; return 0; }
  for try in 1 2 3 4 5; do
    if show_gids 2>/dev/null | awk -v ip="$OWN_IP" -v hca="$ROCE_HCA" -v gi="$ROCE_GID_INDEX" \
        '$1==hca&&$3==gi&&$5==ip&&$6=="v2"{f=1} END{exit !f}'; then
      echo "dspark: RoCE GID idx${ROCE_GID_INDEX}=${OWN_IP}/v2 — ok"; return 0
    fi
    echo "dspark: RoCE GID idx${ROCE_GID_INDEX} drifted (try $try/5) — reactivating $RDV_LINK"
    nmcli con down "$RDV_LINK" >/dev/null 2>&1 || true; sleep 2
    nmcli con up   "$RDV_LINK" >/dev/null 2>&1 || true; sleep 6
  done
  echo "dspark: WARN RoCE GID still not aligned after 5 tries — launching anyway"
}
ensure_roce_gid

echo "dspark: launching dspark-vllm (node-rank $NODE_RANK)"
docker rm -f dspark-vllm 2>/dev/null || true

# Foreground (exec, no -d): systemd Type=simple tracks the docker client; ExecStop
# (docker stop dspark-vllm) cleanly tears the container down.
exec docker run --gpus all --privileged --network host --ipc host --shm-size 10g \
  --ulimit memlock=-1 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HF_CACHE_DIR:/cache/huggingface" \
  --name dspark-vllm \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_USE_B12X_MOE=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e VLLM_TRITON_MLA_SPARSE=1 -e DSPARK_BONUS=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$ROCE_HCA" -e NCCL_IB_GID_INDEX="$ROCE_GID_INDEX" \
  -e NCCL_SOCKET_IFNAME="$ETH_IFACE" -e GLOO_SOCKET_IFNAME="$ETH_IFACE" -e TP_SOCKET_IFNAME="$ETH_IFACE" \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
  --entrypoint bash \
  aidendle94/sparkrun-vllm-ds4-gb10:dspark \
  -lc "exec /usr/local/bin/dsv4-vllm-entrypoint serve deepseek-ai/DeepSeek-V4-Flash-DSpark --served-model-name dspark --host 0.0.0.0 --port 8000 --trust-remote-code --tensor-parallel-size 2 --pipeline-parallel-size 1 --kv-cache-dtype fp8 --block-size 256 --max-model-len 1000000 --max-num-seqs 12 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.86 --speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"model\":\"deepseek-ai/DeepSeek-V4-Flash-DSpark\"}' --tokenizer-mode deepseek_v4 --distributed-executor-backend mp --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --nnodes 2 --node-rank $NODE_RANK --master-addr $MASTER_IP --master-port $RDV_PORT $HEADLESS_FLAG"
