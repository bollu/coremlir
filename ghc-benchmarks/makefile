.PHONY: matmul.out
targets = incr-incr repeated-incr-maybe-recursive repeated-incr-maybe

all: $(targets)

$(targets): %: %.cpp
	clang++ $< -o $@.out -std=c++17 -Wall 



# incr-incr.out: incr-incr.cpp
# 	clang++ incr-incr.cpp -std=c++17 -o incr-incr.out

matmul.out: matmul.hs
	-rm matmul.out
	ghc -ddump-to-file -ddump-simpl -O2 matmul.hs -o matmul.out -package vector
	
