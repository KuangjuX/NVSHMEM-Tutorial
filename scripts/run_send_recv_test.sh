#!/bin/bash

# Script to run intranode send/recv tests
# This script assumes you have at least 2 GPUs on the same node

set -e

# Default values
WORLD_SIZE=${WORLD_SIZE:-2}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29501"}

echo "Starting Intranode Send/Recv Tests"
echo "World Size: $WORLD_SIZE"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "=================================="

# Check if we have enough GPUs
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ $GPU_COUNT -lt $WORLD_SIZE ]; then
    echo "Error: Need at least $WORLD_SIZE GPUs, but only found $GPU_COUNT"
    exit 1
fi

# Export environment variables
export WORLD_SIZE=$WORLD_SIZE
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# Function to run a single process
run_process() {
    local rank=$1
    export RANK=$rank
    export LOCAL_RANK=$rank
    
    echo "Starting rank $rank..."
    python tests/test_intranode_send_recv.py
}

# Run processes
if [ $WORLD_SIZE -eq 2 ]; then
    # For 2 processes, run them sequentially for easier output reading
    run_process 0 &
    run_process 1 &
    wait
else
    # For more processes, run them in parallel
    for i in $(seq 0 $((WORLD_SIZE-1))); do
        run_process $i &
    done
    wait
fi

echo "All tests completed!"
