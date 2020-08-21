//===- HaskDialect.td - Hask dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HASK_DIALECT
#define HASK_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Hask dialect definition.
//===----------------------------------------------------------------------===//

def Hask_Dialect : Dialect {
    let name = "hask";
    let cppNamespace = "standalone";
}

//===----------------------------------------------------------------------===//
// Base standalone operation definition.
//===----------------------------------------------------------------------===//

class Hask_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Hask_Dialect, mnemonic, traits>;

#endif // HASK_DIALECT
