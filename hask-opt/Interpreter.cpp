#pragma once
#include "Interpreter.h"

void interpretFunction(HaskFuncOp func) {
}

// interpret a module, and interpret the result as an integer. print it out.
int interpretModule(ModuleOp module) {
    standalone::HaskFuncOp main = module.lookupSymbol<standalone::HaskFuncOp>("main");
    assert(main && "unable to find main!");
    interpretFunction(main);
    return 5;
};

