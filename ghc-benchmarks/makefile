targets = incr-incr repeated-incr-maybe-recursive

all: $(targets)

$(targets): %: %.cpp
	clang++ $< -o $@.out -std=c++17 -Wall 



# incr-incr.out: incr-incr.cpp
# 	clang++ incr-incr.cpp -std=c++17 -o incr-incr.out

