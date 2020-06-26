.PHONY: build
build:
	set -e
	set -o xtrace
	rm fib.pass-0000.core || true
	rm fib.pass-0000.mlir || true
	cabal build ghc-dump-util ghc-dump-mlir
	rm fib.hi rm.o fib || true
	ghc -fplugin GhcDump.Plugin -dumpdir=dump -O fib.hs -o fib.out
	# cabal install ghc-dump-util ghc-dump-mlir --overwrite-policy=always || true
	cabal exec ghc-dump show  dump/fib.pass-0018.cbor | tee fib.pass-0018.core
	cabal exec ghc-dump show  dump/fib.pass-0000.cbor | tee fib.pass-0000.core
	cabal exec ghc-dump-mlir show dump/fib.pass-0000.cbor  | tee fib.pass-0000.mlir
# 	mlir-opt fib.pass-0000.mlir -allow-unregistered-dialect
