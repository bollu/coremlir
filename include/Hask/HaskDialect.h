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
#include <llvm/ADT/ArrayRef.h>

namespace mlir {
namespace standalone {

//#include "Hask/HaskOpsDialect.h.inc"

class HaskDialect : public mlir::Dialect {
public:
  explicit HaskDialect(mlir::MLIRContext *ctx);
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;
  mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
                                 Type type) const override;
  void printAttribute(Attribute attr,
                      DialectAsmPrinter &printer) const override;
  static llvm::StringRef getDialectNamespace() { return "hask"; }
};

/*
class UntypedType : public mlir::Type::TypeBase<UntypedType, mlir::Type,
                                               TypeStorage> {
public:
  using Base::Base;
  static UntypedType get(MLIRContext *context) { return Base::get(context); }
};
*/

class HaskType : public Type {
public:
  /// Inherit base constructors.
  using Type::Type;

  /// Support for PointerLikeTypeTraits.
  using Type::getAsOpaquePointer;
  static HaskType getFromOpaquePointer(const void *ptr) {
    return HaskType(static_cast<ImplType *>(const_cast<void *>(ptr)));
  }
  /// Support for isa/cast.
  static bool classof(Type type);
  HaskDialect &getDialect();
};

// Follow what ArrayAttributeStorage does:
// https://github.com/llvm/llvm-project/blob/master/mlir/lib/IR/AttributeDetail.h#L50
struct ADTTypeStorage : public TypeStorage {
  ADTTypeStorage(ArrayRef<FlatSymbolRefAttr> const name) : name(name) {}

  /// The hash key used for uniquing.
  using KeyTy = ArrayRef<FlatSymbolRefAttr>;
  bool operator==(const KeyTy &key) const { return key == name; }

  /// Construction.
  static ADTTypeStorage *construct(TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    return new (allocator.allocate<ADTTypeStorage>())
        ADTTypeStorage(allocator.copyInto(key));
  }
  ArrayRef<FlatSymbolRefAttr> name;
};

class ADTType : public mlir::Type::TypeBase<ADTType, HaskType, ADTTypeStorage> {
public:
  using Base::Base;
  static ADTType get(MLIRContext *context, FlatSymbolRefAttr name) {
    return Base::get(context, name);
  }
  FlatSymbolRefAttr getName() { return this->getImpl()->name[0]; }
};

class ValueType
    : public mlir::Type::TypeBase<ValueType, HaskType, TypeStorage> {
public:
  using Base::Base;
  static ValueType get(MLIRContext *context) { return Base::get(context); }
};

struct ThunkTypeStorage : public TypeStorage {
  ThunkTypeStorage(ArrayRef<Type> const t) : t(t) {}

  /// The hash key used for uniquing.
  using KeyTy = ArrayRef<Type>;
  bool operator==(const KeyTy &key) const { return key == t; }

  /// Construction.
  static ThunkTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<ThunkTypeStorage>())
        ThunkTypeStorage(allocator.copyInto(key));
  }

  ArrayRef<Type> getElementType() const { return t; }
  ArrayRef<Type> const t;
};

class ThunkType
    : public mlir::Type::TypeBase<ThunkType, HaskType, ThunkTypeStorage> {
public:
  using Base::Base;
  static ThunkType get(MLIRContext *context, Type elemty) {
    return Base::get(context, elemty);
  }
  Type getElementType() { return *this->getImpl()->getElementType().data(); }
};

/// Function Type Storage and Uniquing.
// https://github.com/llvm/llvm-project/blob/7a06b166b1afb457a7df6ad73a6710b4dde4db68/mlir/lib/IR/TypeDetail.h#L83
struct HaskFnTypeStorage : public TypeStorage {
  HaskFnTypeStorage(ArrayRef<Type> const inputs, ArrayRef<Type> const result)
      : inputs(inputs), result(result){};

  /// The hash key used for uniquing.
  using KeyTy = std::pair<ArrayRef<Type>, ArrayRef<Type>>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getInputs(), getResult());
  }

  /// Construction.
  static HaskFnTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<HaskFnTypeStorage>()) HaskFnTypeStorage(
        allocator.copyInto(key.first), allocator.copyInto(key.second));
  }

  ArrayRef<Type> getInputs() const { return inputs; }
  ArrayRef<Type> getResult() const { return result; }
  ArrayRef<Type> const inputs;
  ArrayRef<Type> const result;
};

// https://github.com/llvm/llvm-project/blob/7a06b166b1afb457a7df6ad73a6710b4dde4db68/mlir/include/mlir/IR/Types.h#L239
// https://github.com/llvm/llvm-project/blob/7a06b166b1afb457a7df6ad73a6710b4dde4db68/mlir/lib/IR/Types.cpp#L36
// https://github.com/llvm/llvm-project/blob/7a06b166b1afb457a7df6ad73a6710b4dde4db68/mlir/lib/IR/TypeDetail.h#L83
class HaskFnType
    : public mlir::Type::TypeBase<HaskFnType, HaskType, HaskFnTypeStorage> {
public:
  using Base::Base;
  static HaskFnType get(MLIRContext *context, ArrayRef<Type> argTy,
                        ArrayRef<Type> resultTy) {
    std::pair<ArrayRef<Type>, ArrayRef<Type>> data(argTy, resultTy);
    return Base::get(context, data);
  }

  ArrayRef<Type> getInputTypes() { return this->getImpl()->getInputs(); }
  Type getResultType() { return this->getImpl()->getResult()[0]; }
};

struct DataConstructorAttributeStorage : public AttributeStorage {
  using KeyTy = std::pair<ArrayRef<SymbolRefAttr>, ArrayRef<ArrayAttr>>;

  // Why does this not work?
  // using KeyTy = std::pair<SymbolRefAttr, ArrayAttr>;

  DataConstructorAttributeStorage(KeyTy value) : value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == value; }

  /// Construct a DataConstructorAttributeStorage storage instance.
  static DataConstructorAttributeStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    // https://github.com/llvm/llvm-project/blob/a517191a474f7d6867621d0f8e8cc454c27334bf/mlir/lib/IR/AttributeDetail.h#L281
    return new (allocator.allocate<DataConstructorAttributeStorage>())
        DataConstructorAttributeStorage(
            std::make_pair(allocator.copyInto(std::get<0>(key)),
                           allocator.copyInto(std::get<1>(key))));
  }
  KeyTy value;
};

class DataConstructorAttr
    : public mlir::Attribute::AttrBase<DataConstructorAttr, mlir::Attribute,
                                       DataConstructorAttributeStorage> {
protected:
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
  static DataConstructorAttr get(MLIRContext *context, SymbolRefAttr Name,
                                 ArrayAttr ArgTys) {
    std::pair<SymbolRefAttr, ArrayAttr> data(Name, ArgTys);
    return Base::get(context, data);
  }

  ArrayRef<SymbolRefAttr> getName() { return this->getImpl()->value.first; }
  ArrayRef<ArrayAttr> getArgTys() { return this->getImpl()->value.second; }

  //  static UntypedType get(MLIRContext *context) { return Base::get(context);
  //  }
};

Attribute parseDataConstructorAttribute(DialectAsmParser &parser, Type type);

} // namespace standalone
} // namespace mlir

#endif // STANDALONE_STANDALONEDIALECT_H
