#include "Hask/HaskDialect.h"
#include "Hask/HaskOps.h"
#include "Hask/Runtime.h"

using namespace mlir;

// interpret a module, and interpret the result as an integer. print it out.
int interpretModule(ModuleOp module) {
    standalone::HaskFuncOp main = module.lookupSymbol<standalone::HaskFuncOp>("main");
    assert(main && "unable to find main!");
    return 5;
};



