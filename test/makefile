# Yes I hate myself, but this is far too convenient to be able to write
# pipelines which involve tee and allow such pipelines to fail.
.PHONY: fib fibstrict playground

fib: 
	make -C ~/work/mlir/coremlir/build
	../build/bin/hask-opt ./fib.mlir
	../build/bin/hask-opt ./fib.mlir > fib-roundtrip.mlir
	../build/bin/hask-opt ./fib-roundtrip.mlir

fibstrict: 
	make -C ~/work/mlir/coremlir/build
	# test round-tripping. Is vital!
	../build/bin/hask-opt ./fib-strict.mlir -optimize 
	../build/bin/hask-opt ./fib-strict.mlir -optimize > fib-strict-roundtrip.mlir
	# ../build/bin/hask-opt ./fib-strict-roundtrip.mlir
	# check that the optimized version runs
	../build/bin/hask-opt ./fib-strict-roundtrip.mlir -optimize

playground: 
	make -C ~/work/mlir/coremlir/build
	../build/bin/hask-opt ./playground.mlir -optimize  | tee playground-roundtrip.mlir
	(../build/bin/hask-opt playground-roundtrip.mlir \
		-optimize 1>/dev/null 2>/dev/null) \
		|| (echo "===ROUNDTRIP FAILED==="; \
			../build/bin/hask-opt playground-roundtrip.mlir; \
			echo "===ROUNTRIP FAILED==="; false)
	rm playground-roundtrip.mlir

test-ops: 
	make -C ~/work/mlir/coremlir/build
	../build/bin/hask-opt ./test-ops.mlir | tee test-ops-roundtrip.mlir
	../build/bin/hask-opt ./test-ops-roundtrip.mlir

all : fib fibstrict test-ops 
