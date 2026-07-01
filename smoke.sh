#!/usr/bin/env bash
# Smoke test for a running DSpark vLLM endpoint (fp8 or nvfp4 — same OpenAI API).
# Checks: (1) /v1/models advertises the `dspark` model, (2) a chat completion returns
# non-empty text. Exit 0 = healthy, non-zero = problem. Needs curl + python3.
#
# usage: ./smoke.sh [BASE_URL]        (default: http://localhost:8000)
#        DSPARK_URL / DSPARK_MODEL env vars also honoured.
set -euo pipefail

BASE_URL="${1:-${DSPARK_URL:-http://localhost:8000}}"
MODEL="${DSPARK_MODEL:-dspark}"

echo "-> smoke test against ${BASE_URL} (model=${MODEL})"

# 1) model list advertises the served model
curl -sf --max-time 10 "${BASE_URL}/v1/models" | MODEL="${MODEL}" python3 -c "
import os, sys, json
ids = [m['id'] for m in json.load(sys.stdin)['data']]
assert os.environ['MODEL'] in ids, f'model {os.environ[\"MODEL\"]!r} not served; got {ids}'
print(f'  ok /v1/models -> {ids}')
"

# 2) one deterministic chat completion returns non-empty text
curl -sf --max-time 60 "${BASE_URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: OK\"}],\"max_tokens\":8,\"temperature\":0}" \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
txt = d['choices'][0]['message']['content']
assert txt.strip(), 'empty completion'
u = d.get('usage', {}) or {}
print(f'  ok /v1/chat/completions -> {txt!r}  (completion_tokens={u.get(\"completion_tokens\")})')
"

echo 'smoke OK'
