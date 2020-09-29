#pragma once
#include "Hask/HaskDialect.h"
#include "Hask/HaskOps.h"
#include "./Runtime.h"


using namespace mlir;
using namespace standalone;

// interpret a module, and interpret the result as an integer. print it out.
int interpretModule(ModuleOp module);

