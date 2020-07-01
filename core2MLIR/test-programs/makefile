.PHONY: clean fib.hs fibstrict.hs

fib:
	ghc -fplugin Core2MLIR.Plugin -dumpdir=dump -O fib.hs -o fib.out  -ddump-simpl -O0


fibstrict:
	ghc -fplugin Core2MLIR.Plugin -dumpdir=dump -O fibstrict.hs -o fib.out  -ddump-simpl -O0

clean:
	-rm -rf dump/
	-rm *.hi *.o *.out
	-rm *.core *.mlir
