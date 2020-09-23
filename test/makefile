# Yes I hate myself, but this is far too convenient to be able to write
# pipelines which involve tee and allow such pipelines to fail.
targets = fib add case-int fib-strict k-lazy either playground test-ops case-usage

all: $(targets)

$(targets): %: %.mlir
	make -C ~/work/mlir/coremlir/build
	../build/bin/hask-opt $<
	../build/bin/hask-opt ./$< > $@-roundtrip.mlir
	../build/bin/hask-opt ./$@-roundtrip.mlir
	../build/bin/hask-opt ./$@-roundtrip.mlir -lower-std
	../build/bin/hask-opt ./$@-roundtrip.mlir -lower-std -lower-llvm
	../build/bin/hask-opt ./$@-roundtrip.mlir -lower-std -lower-llvm -jit

# $(targets): %: %.mlir
# 	make -C ~/work/mlir/coremlir/build
# 	gdb -ex run --args ../build/bin/hask-opt $<.mlir -lower-std -lower-llvm




