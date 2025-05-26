set -x

CHECKPOINT=${1}
DATASET=${2}
CHECKPOINT="$(pwd)/${CHECKPOINT}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
echo "CHECKPOINT: ${CHECKPOINT}"

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

# Save original arguments
ARGS=("$@")

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto)
      GPUS=1
      shift
      ;;
    *)
      shift
      ;;
  esac
done
echo "GPUS: ${GPUS}"

if [[ "${DATASET}" == *"k12"* ]]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/prm/evaluate_k12_prm.py --checkpoint ${CHECKPOINT} --datasets ${DATASET} "${ARGS[@]:2}"
fi

if [[ "${DATASET}" == *"mathvista"* ]]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/prm/evaluate_mathvista_prm.py --checkpoint ${CHECKPOINT} --datasets ${DATASET} "${ARGS[@]:2}"
fi

if [[ "${DATASET}" == *"mathverse"* ]]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/prm/evaluate_mathverse_prm.py --checkpoint ${CHECKPOINT} --datasets ${DATASET} "${ARGS[@]:2}"
fi

if [[ "${DATASET}" == *"mathvision"* ]]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/prm/evaluate_mathvision_prm.py --checkpoint ${CHECKPOINT} --datasets ${DATASET} "${ARGS[@]:2}"
fi

if [[ "${DATASET}" == *"olympiadbench"* ]]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/prm/evaluate_olympiadbench_prm.py --checkpoint ${CHECKPOINT} --datasets ${DATASET} "${ARGS[@]:2}"
fi
