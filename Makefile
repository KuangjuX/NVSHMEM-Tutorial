BENCH ?= benchmark_put.py

build:
	python -m pip install -e .

clean:
	rm -rf build/

bench:
	python -m torch.distributed.run --nproc_per_node=2 benchmarks/$(BENCH)

launch:
	torchrun --nproc_per_node=8 \
		--nnodes=2 \
		--node_rank=0 \
		--master_addr=$MASTER_ADDR \
		--master_port=$MASTER_PORT \
		tests/test_internode_nvshmem.py