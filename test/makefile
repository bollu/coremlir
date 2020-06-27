.PHONY: fib
fib: 
	make -C ~/work/mlir/coremlir/build
	make -C ~/work/mlir/coremlir/ghc-dump
	# test round-tripping. Is vital!
	../build/bin/standalone-opt ./fib-encoded.mlir | tee fib-encoded-roundtrip.mlir

	# ../build/bin/standalone-opt ../ghc-dump/fib.pass-0000.mlir -allow-unregistered-dialect
