.PHONY: fib fibstrict

fibstrict: 
	make -C ~/work/mlir/coremlir/build
	make -C ~/work/mlir/coremlir/ghc-dump
	# test round-tripping. Is vital!
	../build/bin/hask-opt ./fib-strict.mlir | tee fib-strict-roundtrip.mlir
	../build/bin/hask-opt ./fib-strict-roundtrip.mlir
fib: 
	make -C ~/work/mlir/coremlir/build
	make -C ~/work/mlir/coremlir/ghc-dump
	# test round-tripping. Is vital!
	../build/bin/hask-opt ./fib.mlir | tee fib-roundtrip.mlir
	../build/bin/hask-opt ./fib-roundtrip.mlir

	# ../build/bin/hask-opt ../ghc-dump/fib.pass-0000.mlir -allow-unregistered-dialect
	#

try-toplevel-recursive-ref:
	make -C ~/work/mlir/coremlir/build
	make -C ~/work/mlir/coremlir/ghc-dump
	# test round-tripping. Is vital!
	../build/bin/hask-opt ./try-toplevel-recursive-ref.mlir

test-ops: 
	make -C ~/work/mlir/coremlir/build
	../build/bin/hask-opt ./test-ops.mlir | tee test-ops-roundtrip.mlir
	../build/bin/hask-opt ./test-ops-roundtrip.mlir

