#!/bin/bash

set -e


# Set unlimited memory limit
ulimit -l unlimited

export NVSHMEM_DEBUG=TRACE
# export NCCL_DEBUG=WARN

export GLOO_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond1
export NVSHMEM_HCA_LIST=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1
export NCCL_IB_TC=160
export NVSHMEM_IB_TRAFFIC_CLASS=160


# MASTER_ADDR and MASTER_PORT are expected to be set as environment variables on the server
: "${MASTER_ADDR:?Need to set MASTER_ADDR}"
: "${MASTER_PORT:?Need to set MASTER_PORT}"

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    tests/test_internode_allgather.py
