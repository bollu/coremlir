.PHONY: build core2mlir fib fibstrict clean install

# on getting multiple package error, remove the repeated package line from
# vim ~/.ghc/x86_64-linux-8.6.5/environments/default
# the GHC version above is an example.


build:
	set -e
	set -o xtrace
	rm fib.pass-0000.core || true
	rm fib.pass-0000.mlir || true
	cabal build ghc-dump-util ghc-dump-mlir
	rm fib.hi rm.o fib || true

install:
	cabal build ghc-dump-util ghc-dump-mlir ghc-dump-core
	# HUGE FUCKING HACK.
	# 
	echo "" > ~/.ghc/x86_64-linux-8.6.5/environments/default 
	cabal install ghc-dump-core --lib 

fib: clean
	ghc -ddump-simpl -ddump-to-file -fplugin GhcDump.Plugin -O0 -dumpdir=dump -O fib.hs -o fib.out
	cabal exec ghc-dump show  dump/fib.pass-0018.cbor | tee dump/fib.pass-0018.core
	cabal exec ghc-dump show  dump/fib.pass-0000.cbor | tee dump/fib.pass-0000.core
	cabal exec ghc-dump-mlir show dump/fib.pass-0000.cbor  | tee dump/fib.pass-0000.mlir

fibstrict: clean
	ghc -O0 -dverbose-core2core -ddump-simpl  -ddump-to-file -fplugin GhcDump.Plugin -dumpdir=dump -O fibstrict.hs -o fibstrict.out
	cabal exec ghc-dump show  dump/fibstrict.pass-0018.cbor | tee dump/fibstrict.pass-0018.core
	cabal exec ghc-dump show  dump/fibstrict.pass-0000.cbor | tee dump/fibstrict.pass-0000.core
	cabal exec ghc-dump-mlir show dump/fibstrict.pass-0000.cbor  | tee dump/fibstrict.pass-0000.mlir

clean:
	-rm -rf dump/
	-rm *.hi *.o *.out
	-rm *.core *.mlir

core2mlir:
	rm fib.hi rm.o fib || true
	ghc -fplugin Core2MLIR.Plugin -dumpdir=dump -O fib.hs -o fib.out
