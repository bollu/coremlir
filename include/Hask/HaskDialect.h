//===- HaskDialect.h - Hask dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_STANDALONEDIALECT_H
#define STANDALONE_STANDALONEDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace standalone {

#include "Hask/HaskOpsDialect.h.inc"



/// Create a local enumeration with all of the types that are defined by Toy.
namespace HaskTypes {
  enum Types {
    // TODO: I don't really understand how this works. In that,
    //       what if someone else has another 
    Untyped = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
  };
};

class UntypedType : public mlir::Type::TypeBase<UntypedType, mlir::Type,
                                               TypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static bool kindof(unsigned kind) { return kind == HaskTypes::Untyped; }
  static UntypedType get(MLIRContext *context) { return Base::get(context, HaskTypes::Types::Untyped); } 
};



} // namespace standalone
} // namespace mlir

#endif // STANDALONE_STANDALONEDIALECT_H
