#!/bin/bash
# Локальный LLM-сервер для LLM Monitor (ARCHITECTURE.md §9) на узле Slurm с GPU.
#
#   sbatch --nodelist=<узел> scripts/cluster/ollama_serve.sbatch
#   # с рабочей машины: ssh -N -L 11434:<узел>:11434 <login-хост>
#   python scripts/run_demo_cycle.py --timestamp "2025-12-01 20:00:00" \
#       --llm-url http://localhost:11434/v1 --llm-model qwen2.5:7b
#
# Ollama ставится в $HOME/ollama без root; модели -- в $HOME/.ollama/models.
#SBATCH --job-name=neftekod_llm
#SBATCH --time=08:00:00
#SBATCH --output=ollama_%j.log
set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_MODELS=$HOME/.ollama/models
OL=$HOME/ollama/bin/ollama
if [ ! -x "$OL" ]; then
  mkdir -p "$HOME/ollama"
  # с 2025 года дистрибутив -- .tar.zst (прежний .tgz отдаёт 404)
  curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst -o "$HOME/ollama.tar.zst"
  zstd -dc "$HOME/ollama.tar.zst" | tar -C "$HOME/ollama" -xf -
  rm "$HOME/ollama.tar.zst"
  echo INSTALLED
fi
"$OL" serve &
SERVE=$!
for _ in $(seq 1 60); do "$OL" list >/dev/null 2>&1 && break; sleep 2; done
echo "SERVER_UP on $(hostname)"
"$OL" pull qwen2.5:7b && echo PULLED_7B
wait $SERVE
