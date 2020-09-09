# Yes I hate myself, but this is far too convenient to be able to write
# pipelines which involve tee and allow such pipelines to fail.
.PHONY: fib fibstrict playground

fib: 
	make -C ~/work/mlir/coremlir/build
	../build/bin/hask-opt ./fib.mlir 
	../build/bin/hask-opt ./fib.mlir > fib-roundtrip.mlir
	../build/bin/hask-opt ./fib-roundtrip.mlir

fibstrict: 
	-rm fib-strict-roundtrip.mlir
	make -C ~/work/mlir/coremlir/build
	# test round-tripping. Is vital!
	../build/bin/hask-opt ./fib-strict.mlir
	../build/bin/hask-opt ./fib-strict.mlir > fib-strict-roundtrip.mlir
	../build/bin/hask-opt ./fib-strict-roundtrip.mlir


playground: 
	make -C ~/work/mlir/coremlir/build
	../build/bin/hask-opt ./playground.mlir 
	../build/bin/hask-opt ./playground.mlir  > playground-roundtrip.mlir
	../build/bin/hask-opt ./playground.mlir-roundtrip.mlir




test-ops: 
	make -C ~/work/mlir/coremlir/build
	../build/bin/hask-opt ./test-ops.mlir
	../build/bin/hask-opt ./test-ops.mlir > tee test-ops-roundtrip.mlir
	../build/bin/hask-opt ./test-ops-roundtrip.mlir

all : fib fibstrict test-ops 



