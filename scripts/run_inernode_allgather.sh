#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <node_rank>"
    exit 1
fi

NODE_RANK=$1

# MASTER_ADDR and MASTER_PORT are expected to be set as environment variables on the server
: "${MASTER_ADDR:?Need to set MASTER_ADDR}"
: "${MASTER_PORT:?Need to set MASTER_PORT}"

torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    tests/test_internode_allgather.py