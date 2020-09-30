// RUN: ../build/bin/hask-opt %s  -interpret | FileCheck %s
// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s || true
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s || true
// CHECK: 41
// Test that case of int works.
module {
  // strict function
  hask.func @prec {
    %lam = hask.lambda(%ihash: !hask.value) {
     // do we know that %ihash is an int here?
     %ret = hask.caseint %ihash 
     [0 -> { ^entry(%ival: !hask.value): 
                hask.return (%ival): !hask.value      
     }]
     [@default -> { ^entry: // ... or here?
                     %lit_one = hask.make_i64(1)
                     %pred = hask.primop_sub(%ihash, %lit_one)
                     hask.return(%pred): !hask.value

     }]
     hask.return (%ret) : !hask.value
    }
    hask.return (%lam): !hask.fn<(!hask.value) -> !hask.value>
  }

  hask.func @main {
    %lambda = hask.lambda() {
      %lit_42 = hask.make_i64(42)
      %prec = hask.ref(@prec)  : !hask.fn<(!hask.value) -> !hask.value>
      %out_v = hask.ap(%prec : !hask.fn<(!hask.value) -> !hask.value>, %lit_42)
      %out_v_forced = hask.force(%out_v): !hask.value
      %x = hask.construct(@X, %out_v_forced:!hask.value): !hask.adt<@X>
      hask.return(%x) : !hask.adt<@X>
    }
    hask.return(%lambda) : !hask.fn<() -> !hask.adt<@X>>
  }
    
}
