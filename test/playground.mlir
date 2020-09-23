// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm -jit | FileCheck %s
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm -jit |  FileCheck %s
// CHECK: 42
// Test that case of int works.
module {
  hask.func @main {
      %lam = hask.lambda(%x: !hask.value) {
        %lit_42 = std.constant 42 : i64
        // hask.return(%lit_42) : i64
        %ival = hask.transmute(%lit_42 : i64): !hask.value
        hask.return(%ival) : !hask.value
      }
      hask.return(%lam): !hask.fn<(!hask.value) -> !hask.value>
      // hask.return(%lam): !hask.fn<(!hask.value) -> i64>
  }
    
}
