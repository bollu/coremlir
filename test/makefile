.PHONY: fib fibstrict

fibstrict: 
	make -C ~/work/mlir/coremlir/build
	make -C ~/work/mlir/coremlir/ghc-dump
	# test round-tripping. Is vital!
	../build/bin/standalone-opt ./fib-strict.mlir | tee fib-strict-roundtrip.mlir
	../build/bin/standalone-opt ./fib-strict-roundtrip.mlir
fib: 
	make -C ~/work/mlir/coremlir/build
	make -C ~/work/mlir/coremlir/ghc-dump
	# test round-tripping. Is vital!
	../build/bin/standalone-opt ./fib.mlir | tee fib-roundtrip.mlir
	../build/bin/standalone-opt ./fib-roundtrip.mlir

	# ../build/bin/standalone-opt ../ghc-dump/fib.pass-0000.mlir -allow-unregistered-dialect
	#
