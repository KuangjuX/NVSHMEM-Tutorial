#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <node_rank>"
    exit 1
fi

NODE_RANK=$1

# Set unlimited memory limit
ulimit -l unlimited

export GLOO_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond1
export NVSHMEM_HCA_LIST=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1
export NCCL_IB_TC=160
export NVSHMEM_IB_TRAFFIC_CLASS=160
export NVSHMEM_DEBUG_SUBSYS=INIT


# MASTER_ADDR and MASTER_PORT are expected to be set as environment variables on the server
: "${MASTER_ADDR:?Need to set MASTER_ADDR}"
: "${MASTER_PORT:?Need to set MASTER_PORT}"

torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    benchmarks/bench_allgather.py
