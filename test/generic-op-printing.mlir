// test generic op printing
// RUN: ../build/bin/hask-opt %s  -interpret | FileCheck %s
// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s || true
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s || true
// CHECK: 8
// Core2MLIR: GenMLIR BeforeCorePrep
module {

  hask.func @two () -> !hask.adt<@SimpleInt>  {
     %v = hask.make_i64(2)
     %boxed = hask.construct(@SimpleInt, %v:!hask.value) : !hask.adt<@SimpleInt> 
     hask.return(%boxed): !hask.adt<@SimpleInt> 
  }

  hask.func @main (%v: !hask.adt<@SimpleInt>, %wt: !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt> {
      // %number = "hask.make_i64"  () { value = 0 : i64} : () -> !hask.value
      %number = hask.make_i64(0 : i64)
      // Try to use the generic syntax to build a %boxed_number
      // %boxed_number = "hask.construct"(%number)  { dataconstructor=@MkSimpleInt, datatype=@SimpleInt } : (!hask.value) -> (!hask.adt<@SimpleInt>)
      %reti = hask.case @SimpleInt %v 
           [@SimpleInt -> { ^entry(%ival: !hask.value):
              %w = hask.force(%wt):!hask.adt<@SimpleInt>
               %number43 = hask.make_i64(43 : i64)
               hask.return(%number43):!hask.value
            }]
            [@default -> { 
               ^entry:
                   %number42 = hask.make_i64(42 : i64)
                   hask.return(%number42):!hask.value

            }]
      %two = hask.ref(@two):  !hask.fn<() -> !hask.adt<@SimpleInt>>
      %twot = hask.ap(%two :  !hask.fn<() -> !hask.adt<@SimpleInt>> )
      hask.return(%reti) : !hask.value
  }
}
