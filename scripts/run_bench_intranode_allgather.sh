set -e


if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <nproc_per_node>"
    exit 1
fi

NPROC_PER_NODE=$1

# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    benchmarks/bench_allgather.py
    