#!/usr/bin/env bash
# Minimal end-to-end reproduction of SkyRL-Agent (arXiv 2511.16108).
#
# Spins up a local vLLM OpenAI-compatible server with a small instruct model
# (same backend path as the paper's examples/run_openai, scaled down from
# Qwen3-32B), then runs the SkyRL-Agent ReAct loop over a tiny math set and
# scores the rollouts with the math verifier. Demonstrates the paper's core
# mechanism: tool-centric agent loop + async dispatcher + transition-based
# backend bridge + verifiable reward. Runs on a single GPU.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export REPRO_MODEL="${REPRO_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export REPRO_API_URL="${REPRO_API_URL:-http://127.0.0.1:8000}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local}"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p .openresearch/artifacts

echo "==================================================================="
echo " SkyRL-Agent minimal repro"
echo "   model   = $REPRO_MODEL"
echo "   api_url = $REPRO_API_URL"
echo "==================================================================="

# ---------------------------------------------------------------------------
# 1. Environment: a single venv with vLLM (serving) + the lightweight deps the
#    math/finish/openai inference path needs.
# ---------------------------------------------------------------------------
VENV=.repro_venv
uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[setup] installing vLLM + inference deps (this can take a few minutes)..."
# IMPORTANT: this repo's root pyproject.toml / uv.lock pin
# transformers>=5.6.1,<=5.8.0 for the SkyRL training stack. `uv pip install`
# run inside the repo treats that as a workspace constraint and overrides any
# transformers we ask for (forcing 5.8.0), which breaks vLLM. We install our
# isolated inference env with plain pip from a neutral directory so the repo's
# workspace pins do not leak in.
#
# vLLM 0.9.2 needs transformers>=4.51.1 but breaks on transformers 5.x (config
# registry collisions, e.g. 'aimv2'); pin an exact 0.9.2-compatible version.
( cd /tmp && python -m pip install -q --upgrade pip )
( cd /tmp && python -m pip install -q \
    "vllm==0.9.2" \
    "transformers==4.53.2" \
    "datasets" "omegaconf" "aiohttp" "loguru" \
    "pandas" "numpy" "sympy" "pylatexenc" "litellm" "math_verify" \
    "pyarrow" "json5" "retry" "codetiming" "hf_transfer" )

python - <<'PY'
import transformers, vllm
print(f"[setup] vllm={vllm.__version__} transformers={transformers.__version__}")
assert transformers.__version__.startswith("4.53"), (
    f"Expected transformers 4.53.x for vLLM 0.9.2, got {transformers.__version__}"
)
PY

# ---------------------------------------------------------------------------
# 2. Launch the vLLM OpenAI-compatible server in the background.
# ---------------------------------------------------------------------------
echo "[serve] starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
  --model "$REPRO_MODEL" \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --disable-log-requests \
  > .openresearch/artifacts/vllm_server.log 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "[cleanup] stopping vLLM server (pid $VLLM_PID)..."
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 3. Wait for the server to become healthy.
# ---------------------------------------------------------------------------
echo "[serve] waiting for vLLM to be ready..."
READY=0
for i in $(seq 1 120); do
  if curl -s -m 5 "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1; then
    READY=1
    echo "[serve] vLLM is ready after ${i}0s."
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[serve] ERROR: vLLM process died during startup. Last log lines:"
    tail -n 50 .openresearch/artifacts/vllm_server.log || true
    exit 1
  fi
  sleep 10
done
if [ "$READY" -ne 1 ]; then
  echo "[serve] ERROR: vLLM did not become ready in time. Last log lines:"
  tail -n 50 .openresearch/artifacts/vllm_server.log || true
  exit 1
fi

# ---------------------------------------------------------------------------
# 4. Run the agent rollouts + evaluation.
# ---------------------------------------------------------------------------
echo "[run] launching SkyRL-Agent ReAct rollouts..."
python run_minimal_repro.py

echo "==================================================================="
echo " Done. Summary (EVAL.md):"
echo "==================================================================="
cat EVAL.md
