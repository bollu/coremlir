.PHONY: build core2mlir fib fibstrict clean


build:
	set -e
	set -o xtrace
	rm fib.pass-0000.core || true
	rm fib.pass-0000.mlir || true
	cabal build ghc-dump-util ghc-dump-mlir
	rm fib.hi rm.o fib || true

fib: clean
	ghc -ddump-simpl -ddump-to-file -fplugin GhcDump.Plugin -O0 -dumpdir=dump -O fib.hs -o fib.out
	cabal exec ghc-dump show  dump/fib.pass-0018.cbor | tee dump/fib.pass-0018.core
	cabal exec ghc-dump show  dump/fib.pass-0000.cbor | tee dump/fib.pass-0000.core
	cabal exec ghc-dump-mlir show dump/fib.pass-0000.cbor  | tee dump/fib.pass-0000.mlir

fibstrict: clean
	ghc -dverbose-core2core -ddump-simpl  -ddump-to-file -fplugin GhcDump.Plugin -dumpdir=dump -O fibstrict.hs -o fibstrict.out
	cabal exec ghc-dump show  dump/fib.pass-0018.cbor | tee fib.pass-0018.core
	cabal exec ghc-dump show  dump/fib.pass-0000.cbor | tee fib.pass-0000.core
	cabal exec ghc-dump-mlir show dump/fib.pass-0000.cbor  | tee fib.pass-0000.mlir

clean:
	-rm -rf dump/
	-rm *.hi *.o *.out
	-rm *.core *.mlir

core2mlir:
	rm fib.hi rm.o fib || true
	ghc -fplugin Core2MLIR.Plugin -dumpdir=dump -O fib.hs -o fib.out
