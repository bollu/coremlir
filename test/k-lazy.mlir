// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s
// CHECK: 42
module {
  // k x y = x
  hask.func @k {
    %lambda = hask.lambdaSSA(%x: !hask.thunk<!hask.untyped>, %y: !hask.thunk<!hask.untyped>) {
      hask.return(%x) : !hask.thunk<!hask.untyped>
    }
    hask.return(%lambda) :!hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
  }

  // loop a = loop a
  hask.func @loop {
    %lambda = hask.lambdaSSA(%a: !hask.thunk<!hask.untyped>) {
      %loop = hask.ref(@loop) : !hask.fn<(!hask.thunk<!hask.untyped>) ->  !hask.thunk<!hask.untyped>>
      %out_t = hask.apSSA(%loop : !hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>, %a)
      %out_v = hask.force(%out_t : !hask.thunk<!hask.untyped>) : !hask.thunk<!hask.untyped>
      hask.return(%out_v) : !hask.thunk<!hask.untyped>
    }
    hask.return(%lambda) : !hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
  }

  hask.adt @X [#hask.data_constructor<@MkX []>]

  // k (x:(X 42)) (y:(loop (X 42))) = x
  // main = 
  //     let y = loop x -- builds a closure.
  //     in (k x y)
  hask.func @main {
    %lambda = hask.lambdaSSA(%_: !hask.thunk<!hask.untyped>) {
      %lit_42 = hask.make_i64(42)
      %x = hask.construct(@X, %lit_42)
      %k = hask.ref(@k) : !hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
      %loop = hask.ref(@loop) :  !hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
      %y = hask.apSSA(%loop : !hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>, %x)
      %out_t = hask.apSSA(%k: !hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>, %x, %y)
      %out = hask.force(%out_t : !hask.thunk<!hask.untyped>) : !hask.value
      hask.return(%out) : !hask.value
    }
    hask.return(%lambda) :!hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.value>
  }
}
