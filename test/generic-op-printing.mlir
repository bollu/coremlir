// test generic op printing
// RUN: ../build/bin/hask-opt %s  -interpret | FileCheck %s
// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s || true
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s || true
// CHECK: 8
// Core2MLIR: GenMLIR BeforeCorePrep
module {
  hask.func@main () -> !hask.adt<@SimpleInt> {
      %number = "hask.make_i64"  () { value = 0 : i64} : () -> !hask.value
      // %number = hask.make_i64(0 : i64)
      // Try to use the generic syntax to build a %boxed_number
      %boxed_number = "hask.construct"(%number)  { dataconstructor=@MkSimpleInt, datatype=@SimpleInt } : (!hask.value) -> (!hask.adt<@SimpleInt>)
      hask.return(%number) : !hask.value
  }

}
