targets = incr-incr repeated-incr-maybe-recursive repeated-incr-maybe

all: $(targets)

$(targets): %: %.cpp
	clang++ $< -o $@.out -std=c++17 -wall 



# incr-incr.out: incr-incr.cpp
# 	clang++ incr-incr.cpp -std=c++17 -o incr-incr.out

