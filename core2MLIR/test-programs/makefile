.PHONY: build core2mlir fib fibstrict clean install

nonrecsum: clean install
	ghc -O0 -dverbose-core2core -ddump-prep -ddump-ds -ddump-simpl  -ddump-to-file -fplugin Core2MLIR.Plugin -dumpdir=dump -O nonrecsum.hs
	cp dump/tomlir-nonrecsum.pass-0001.mlir .
	cp dump/tomlir-nonrecsum.pass-0000.mlir .
	# make -C ../../build/
	# ../../build/bin/hask-opt tomlir-fib.pass-0000.mlir -optimize

fib: clean install
	ghc -O0 -dverbose-core2core -ddump-prep -ddump-ds -ddump-simpl  -ddump-to-file -fplugin Core2MLIR.Plugin -dumpdir=dump -O fib.hs
	cp dump/tomlir-fib.pass-0001.mlir .
	cp dump/tomlir-fib.pass-0000.mlir .
	make -C ../../build/
	../../build/bin/hask-opt tomlir-fib.pass-0000.mlir -optimize

fibstrict: clean install
	ghc -O0 -dverbose-core2core -ddump-prep -ddump-ds -ddump-simpl  -ddump-to-file -fplugin Core2MLIR.Plugin -dumpdir=dump -O fibstrict.hs -o fibstrict.out
	cp dump/tomlir-fibstrict.pass-0001.mlir .
	cp dump/tomlir-fibstrict.pass-0000.mlir .
	make -C ../../build/
	# cat tomlir-fibstrict.pass-0001.mlir
	# ../../build/bin/hask-opt tomlir-fibstrict.pass-0001.mlir
	cat tomlir-fibstrict.pass-0000.mlir
	../../build/bin/hask-opt tomlir-fibstrict.pass-0000.mlir -optimize
	# ../../build/bin/hask-opt tomlir-fibstrict.pass-0001.mlir


# on getting multiple package error, remove the repeated package line from
# vim ~/.ghc/x86_64-linux-8.6.5/environments/default
# the GHC version above is an example.

install:
	(cd ../ && cabal build Core2MLIR)
	# HUGE FUCKING HACK.
	-rm ~/.ghc/x86_64-linux-8.6.5/environments/default 
	(cd ../ && cabal install Core2MLIR --lib  --overwrite-policy=always --global)

clean:
	-rm -rf dump/
	-rm *.hi *.o *.out
	-rm *.core *.mlir

core2mlir:
	rm fib.hi rm.o fib || true
	ghc -fplugin Core2MLIR.Plugin -dumpdir=dump -O fib.hs -o fib.out
