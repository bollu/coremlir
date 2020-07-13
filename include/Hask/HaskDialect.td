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
    let summary = "A standalone out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let extraClassDeclaration = [{
      Type parseType(DialectAsmParser &parser) const override;
      void printType(Type type, DialectAsmPrinter &os) const override;
    }];


    let cppNamespace = "standalone";

}

//===----------------------------------------------------------------------===//
// Base standalone operation definition.
//===----------------------------------------------------------------------===//

class Hask_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Hask_Dialect, mnemonic, traits>;

#endif // HASK_DIALECT
