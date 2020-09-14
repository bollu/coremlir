# Yes I hate myself, but this is far too convenient to be able to write
# pipelines which involve tee and allow such pipelines to fail.
.PHONY: fib fibstrict playground add

add: 
	make -C ~/work/mlir/coremlir/build
	../build/bin/hask-opt ./add.mlir 
	../build/bin/hask-opt ./add.mlir > add-roundtrip.mlir
	../build/bin/hask-opt ./add-roundtrip.mlir
	../build/bin/hask-opt ./add-roundtrip.mlir -lower-std -lower-llvm
	../build/bin/hask-opt ./add-roundtrip.mlir -lower-std -lower-llvm -jit


fib: 
	make -C ~/work/mlir/coremlir/build
	../build/bin/hask-opt ./fib.mlir 
	../build/bin/hask-opt ./fib.mlir > fib-roundtrip.mlir
	../build/bin/hask-opt ./fib-roundtrip.mlir
	../build/bin/hask-opt ./fib-roundtrip.mlir -lower-std -lower-llvm
	../build/bin/hask-opt ./fib-roundtrip.mlir -lower-std -lower-llvm -jit

case-int: 
	make -C ~/work/mlir/coremlir/build
	../build/bin/hask-opt ./case-int.mlir
	../build/bin/hask-opt ./case-int.mlir > case-int-roundtrip.mlir
	../build/bin/hask-opt ./case-int-roundtrip.mlir
	../build/bin/hask-opt ./case-int.mlir -lower-std -lower-llvm
	../build/bin/hask-opt ./case-int.mlir -lower-std -lower-llvm -jit


k-lazy: 
	make -C ~/work/mlir/coremlir/build
	../build/bin/hask-opt ./k-lazy.mlir
	../build/bin/hask-opt ./k-lazy.mlir > k-lazy-roundtrip.mlir
	../build/bin/hask-opt ./k-lazy-roundtrip.mlir
	../build/bin/hask-opt ./k-lazy.mlir -lower-std -lower-llvm
	../build/bin/hask-opt ./k-lazy.mlir -lower-std -lower-llvm -jit


fibstrict: 
	-rm fib-strict-roundtrip.mlir
	make -C ~/work/mlir/coremlir/build
	# test round-tripping. Is vital!
	../build/bin/hask-opt ./fib-strict.mlir
	../build/bin/hask-opt ./fib-strict.mlir > fib-strict-roundtrip.mlir
	../build/bin/hask-opt ./fib-strict-roundtrip.mlir
	../build/bin/hask-opt ./fib-strict.mlir -lower-std -lower-llvm
	../build/bin/hask-opt ./fib-strict.mlir -lower-std -lower-llvm -jit


playground: 
	make -C ~/work/mlir/coremlir/build
	../build/bin/hask-opt ./playground.mlir 
	../build/bin/hask-opt ./playground.mlir  > playground-roundtrip.mlir
	../build/bin/hask-opt ./playground-roundtrip.mlir
	../build/bin/hask-opt ./playground.mlir -lower-std -lower-llvm



test-ops: 
	make -C ~/work/mlir/coremlir/build
	../build/bin/hask-opt ./test-ops.mlir
	../build/bin/hask-opt ./test-ops.mlir > test-ops-roundtrip.mlir
	../build/bin/hask-opt ./test-ops-roundtrip.mlir

all : fib fibstrict test-ops 



