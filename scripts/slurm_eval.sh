#!/bin/bash
# Run `framagen eval` on a GPU node with several single-request Ollama servers on the allocated GPU.
#
#   sbatch scripts/slurm_eval.sh --split svcomp --run-name svcomp-qwen3-8b-s42-t30 \
#       --model ollama:qwen3:8b --seed 42 --timeout 30
#
# All arguments are passed to `framagen eval`; --ollama-hosts and --jobs are
# filled in here (FRAMAGEN_SERVERS_PER_GPU servers per GPU, default 4; one
# request per server and one task per server at a time, so seeded
# sampling stays reproducible). The run resumes from results.jsonl, so
# resubmitting the same command continues after the 8 h time limit.
#
#SBATCH --job-name=framagen-eval
#SBATCH --partition=batch
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm-%j.out

set -euo pipefail

REPO="${FRAMAGEN_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
OLLAMA_BIN="${OLLAMA_BIN:-$HOME/.local/ollama/bin/ollama}"
MODEL_TAG="${FRAMAGEN_OLLAMA_MODEL:-qwen3:8b}"
NUM_CTX="${FRAMAGEN_NUM_CTX:-32768}"
# Several single-request servers per GPU: qwen3:8b needs ~10 GB at 32k context,
# so a 48 GB L40S fits 4 (the gpu-limited QOS allows 1 GPU per user).
SERVERS_PER_GPU="${FRAMAGEN_SERVERS_PER_GPU:-4}"
export FRAMAGEN_FRAMAC_BACKEND="${FRAMAGEN_FRAMAC_BACKEND:-apptainer}"
export FRAMAGEN_FRAMAC_IMAGE="${FRAMAGEN_FRAMAC_IMAGE:-$HOME/containers/frama-c_32.1}"

cd "$REPO"
mkdir -p runs

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
BASE_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 2000) * 16))
HOSTS=()
PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT

for i in $(seq 0 $(( ${#GPUS[@]} * SERVERS_PER_GPU - 1 ))); do
  gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  port=$((BASE_PORT + i))
  log="runs/ollama-${SLURM_JOB_ID:-local}-gpu${gpu}-s${i}.log"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OLLAMA_HOST="127.0.0.1:${port}" \
  OLLAMA_NUM_PARALLEL=1 \
  OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_KEEP_ALIVE=-1 \
  OLLAMA_CONTEXT_LENGTH="$NUM_CTX" \
    "$OLLAMA_BIN" serve > "$log" 2>&1 &
  PIDS+=($!)
  HOSTS+=("http://127.0.0.1:${port}")
done

for host in "${HOSTS[@]}"; do
  for _ in $(seq 120); do
    curl -sf "$host/api/version" > /dev/null && break
    sleep 1
  done
  curl -sf "$host/api/version" > /dev/null || { echo "Ollama at $host did not start"; exit 1; }
  # Load the model onto this server's GPU before timing anything.
  curl -sf "$host/api/generate" -d "{\"model\":\"$MODEL_TAG\",\"prompt\":\"ok\",\"stream\":false,\"options\":{\"num_ctx\":$NUM_CTX,\"num_predict\":1}}" > /dev/null
done

echo "node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-} hosts=${HOSTS[*]}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || true
"$OLLAMA_BIN" --version 2>/dev/null | tail -1 || true

HOST_LIST=$(IFS=','; echo "${HOSTS[*]}")
"$REPO/.venv/bin/framagen" eval --ollama-hosts "$HOST_LIST" --jobs "${#HOSTS[@]}" "$@"
