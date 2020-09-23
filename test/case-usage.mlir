// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm -jit | FileCheck %s
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm -jit |  FileCheck %s
// CHECK: 42
// Test that a non-trivial use of a case works. So we don't just have:
// %x = case {.. ret }; return(%x)
// but we have %x = case { ... ret }; %y = nontrivial(%x)
module {
  hask.func @main {
    %lambda = hask.lambda(%_: !hask.thunk<!hask.value>) {
      %lit_43 = hask.make_i64(43)
      %case_val = hask.caseint %lit_43 
      [0 -> { ^entry(%ival: !hask.value): 
                 hask.return (%ival): !hask.value      
      }]
      [@default -> { ^entry: // ... or here?
                      %lit_one = hask.make_i64(1)
                      %pred = hask.primop_sub(%lit_43, %lit_one)
                      hask.return(%pred): !hask.value

      }]

      %x = hask.construct(@X, %case_val:!hask.value): !hask.adt<@X>
      hask.return(%x) : !hask.adt<@X>
    }
    hask.return(%lambda) : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.adt<@X>>
  }
    
}