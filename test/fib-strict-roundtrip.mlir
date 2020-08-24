

module {
  llvm.func @fibstrict(%arg0: !llvm.i64) -> !llvm.i64 {
    %0 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %1 = llvm.icmp "eq" %0, %arg0 : !llvm.i64
    llvm.cond_br %1, ^bb1(%arg0 : !llvm.i64), ^bb2(%arg0 : !llvm.i64)
  ^bb1(%2: !llvm.i64):  // pred: ^bb0
    llvm.return %arg0 : !llvm.i64
  ^bb2(%3: !llvm.i64):  // pred: ^bb0
    %4 = llvm.mlir.constant(1 : i64) : !llvm.i64
    %5 = llvm.icmp "eq" %4, %arg0 : !llvm.i64
    llvm.cond_br %5, ^bb3(%arg0 : !llvm.i64), ^bb4(%arg0 : !llvm.i64)
  ^bb3(%6: !llvm.i64):  // pred: ^bb2
    llvm.return %arg0 : !llvm.i64
  ^bb4(%7: !llvm.i64):  // pred: ^bb2
    %8 = llvm.mlir.constant(1 : i64) : !llvm.i64
    %9 = llvm.sub %arg0, %8 : !llvm.i64
    %10 = llvm.call @fibstrict(%9) : (!llvm.i64) -> !llvm.i64
    %11 = llvm.call @fibstrict(%arg0) : (!llvm.i64) -> !llvm.i64
    %12 = llvm.add %11, %10 : !llvm.i64
    llvm.return %12 : !llvm.i64
  }
}