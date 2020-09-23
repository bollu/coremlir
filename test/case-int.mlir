// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm -jit | FileCheck %s
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm -jit |  FileCheck %s
// CHECK: 41
// Test that case of int works.
module {
  // strict function
  hask.func @prec {
    %lam = hask.lambda(%ihash: !hask.value) {
     // do we know that %ihash is an int here?
     %ival = hask.transmute(%ihash : !hask.value): i64
     %ret = hask.caseint %ival 
     [0 -> { ^entry(%z: i64): 
                %z_val = hask.transmute(%ival: i64): !hask.value
                hask.return (%z_val): !hask.value      
     }]
     [@default -> { ^entry: // ... or here?
                     %lit_one = constant 42 : i64
                     %pred = std.subi %ival, %lit_one: i64
                     %pred_val = hask.transmute(%pred : i64): !hask.value
                     hask.return(%pred_val): !hask.value

     }]
     hask.return (%ret) : !hask.value
    }
    hask.return (%lam): !hask.fn<(!hask.value) -> !hask.value>
  }

  hask.func @main {
    %lambda = hask.lambda(%_: !hask.thunk<!hask.value>) {
      %lit_42 = hask.make_i64(42)
      %prec = hask.ref(@prec)  : !hask.fn<(!hask.value) -> !hask.value>
      %out_v = hask.ap(%prec : !hask.fn<(!hask.value) -> !hask.value>, %lit_42)
      %out_v_forced = hask.force(%out_v): !hask.value
      %x = hask.construct(@X, %out_v_forced:!hask.value): !hask.adt<@X>
      hask.return(%x) : !hask.adt<@X>
    }
    hask.return(%lambda) : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.adt<@X>>
  }
    
}
