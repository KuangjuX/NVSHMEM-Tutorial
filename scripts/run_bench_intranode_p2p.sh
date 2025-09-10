#!/bin/bash

set -e


if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <nproc_per_node>"
    exit 1
fi

NPROC_PER_NODE=$1

# MASTER_ADDR and MASTER_PORT are expected to be set as environment variables on the server
: "${MASTER_ADDR:?Need to set MASTER_ADDR}"
: "${MASTER_PORT:?Need to set MASTER_PORT}"

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    benchmarks/bench_p2p.py
  