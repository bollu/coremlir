.PHONY: fib
fib: 
	make -C ~/work/mlir/coremlir/build
	make -C ~/work/mlir/coremlir/ghc-dump
	# ../build/bin/standalone-opt fib.mlir -allow-unregistered-dialect
	../build/bin/standalone-opt ../ghc-dump/fib.pass-0000.mlir -allow-unregistered-dialect
