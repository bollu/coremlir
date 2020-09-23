// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm -jit | FileCheck %s
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm -jit |  FileCheck %s
// CHECK: 42
// Test that case of int works.
module {
  hask.func @main {
      %lam = hask.lambda() {
        %lit_42 = hask.make_i64(42)
        %ival = hask.transmute(%lit_42 : !hask.value): i64
        hask.return(%ival) : i64
      }
      hask.return(%lam): !hask.fn<() -> i64>
  }
    
}
