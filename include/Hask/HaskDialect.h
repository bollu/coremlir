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
#include "mlir/Pass/Pass.h"


namespace mlir {
namespace standalone {

//#include "Hask/HaskOpsDialect.h.inc"

class HaskDialect : public mlir::Dialect {
public:
  explicit HaskDialect(mlir::MLIRContext *ctx);
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;
  mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser, Type type) const override;
  void printAttribute(Attribute attr,
                      DialectAsmPrinter &printer) const override;
  static llvm::StringRef getDialectNamespace() { return "hask"; }
};

class UntypedType : public mlir::Type::TypeBase<UntypedType, mlir::Type,
                                               TypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static UntypedType get(MLIRContext *context) { return Base::get(context); }
};

class DataConstructorAttr : public mlir::Attribute::AttrBase<DataConstructorAttr, mlir::Attribute, AttributeStorage> {
public:
  // The usual story, pull stuff from AttrBase.
  using Base::Base;

  /*
  static bool classof(Attribute attr) {
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";
    const bool correct = attr.isa<DataConstructorAttr>();
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";
    return correct;
  }*/

    static DataConstructorAttr get(MLIRContext *context) { return Base::get(context); }


//  static UntypedType get(MLIRContext *context) { return Base::get(context); }
};

Attribute parseDataConstructorAttribute(DialectAsmParser &parser, Type type);

} // namespace standalone
} // namespace mlir

#endif // STANDALONE_STANDALONEDIALECT_H
