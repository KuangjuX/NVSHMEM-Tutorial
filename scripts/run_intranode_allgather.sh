set -e


if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <nproc_per_node>"
    exit 1
fi

NPROC_PER_NODE=$1

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    tests/test_intranode_allgather.py
    