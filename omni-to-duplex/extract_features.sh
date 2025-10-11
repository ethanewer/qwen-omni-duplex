nvidia-smi -L
NGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')

echo "Using ${NGPUS} GPUs"

uv run -q python -m torch.distributed.run --nproc_per_node="${NGPUS}" \
  extract_features.py