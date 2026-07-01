# Recipe — serve DeepSeek-V4-Flash-DSpark on 2× DGX Spark (GB10, TP=2)

> This is the working launch path: build a `:dspark` image from a base SM12x
> DeepSeek-V4 image, then serve it on 2× DGX Spark with TP=2. The draft produces
> tokens (the MLA `precompute_and_store_context_kv` path is in place).

## 0. Prereqs
- 2× DGX Spark (GB10, sm_121), DGX OS 7.5 / driver 580 (do NOT bump — the qualified
  driver; newer drivers regress GB10 unified memory).
- The community SM12x vLLM fork image that already serves plain DeepSeek-V4-Flash
  (here: `aidendle94/sparkrun-vllm-ds4-gb10:production-ready`, vLLM 0.21.1rc1.dev339).
- Checkpoint `deepseek-ai/DeepSeek-V4-Flash-DSpark` (~165 GB) in the HF cache on BOTH nodes.

## 1. Apply patches to a COPY of the image
Do NOT modify a running container. Build a derived image:
```bash
bash apply-patches.sh aidendle94/sparkrun-vllm-ds4-gb10:production-ready \
                      aidendle94/sparkrun-vllm-ds4-gb10:dspark
```

## 2. Free memory (no room for a 2nd full instance)
```bash
# both nodes — 128 GB unified can't hold a running engine (~110 GB) + DSpark (~82 GB/node).
# Stop any existing vLLM service before launching DSpark.
sudo systemctl stop <existing-vllm-service>
```

## 3. Launch (deltas vs a plain serve command marked ←)
```bash
docker run --gpus all --privileged --network host --ipc host --shm-size 10g \
  --ulimit memlock=-1 --device /dev/infiniband:/dev/infiniband \
  -v "$HOME/.cache/huggingface:/cache/huggingface" --name dspark-vllm \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e VLLM_USE_B12X_MOE=1 -e VLLM_TRITON_MLA_SPARSE=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=<ROCE_IFACE> -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME=<ETH_IFACE> -e GLOO_SOCKET_IFNAME=<ETH_IFACE> -e TP_SOCKET_IFNAME=<ETH_IFACE> \
  --entrypoint bash aidendle94/sparkrun-vllm-ds4-gb10:dspark -lc \
  'exec /usr/local/bin/dsv4-vllm-entrypoint serve deepseek-ai/DeepSeek-V4-Flash-DSpark \
     --served-model-name dspark \
     --host 0.0.0.0 --port 8000 --trust-remote-code \
     --tensor-parallel-size 2 --pipeline-parallel-size 1 \
     --kv-cache-dtype fp8 --block-size 256 --max-model-len 65536 --max-num-seqs 8 \
     --gpu-memory-utilization 0.86 --enable-prefix-caching \
     --speculative-config '"'"'{"method":"dspark","num_speculative_tokens":5}'"'"' \  # ← DSpark
     --enforce-eager \                                                                  # ← no cudagraph
     --tokenizer-mode deepseek_v4 --distributed-executor-backend mp \
     --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
     --nnodes 2 --node-rank 0 --master-addr <MASTER_IP> --master-port <RDV_PORT>'
# peer node: same but --node-rank 1 + --headless
```

## 4. Verify + benchmark
```bash
curl -s http://<MASTER_IP>:8000/v1/models                  # dspark listed
curl -s .../v1/chat/completions -d '{"model":"dspark",...}'   # "Paris" smoke
python3 benchmarks/bench_vllm.py http://<MASTER_IP>:8000/v1/chat/completions dspark 256
# compare accept-rate + tok/s vs the no-spec baseline
```

## 5. Restore the previous engine
```bash
docker rm -f dspark-vllm            # both nodes
sudo systemctl start <existing-vllm-service>
```
