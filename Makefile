BENCH ?= benchmark_put.py

build:
	python -m pip install -e .

clean:
	rm -rf build/

bench:
	python -m torch.distributed.run --nproc_per_node=2 benchmarks/$(BENCH)