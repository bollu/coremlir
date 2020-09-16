// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s
// CHECK: 3
module {
  // should it be Attr Attr, with the "list" embedded as an attribute,
  // or should it be Attr [Attr]? Who really knows :(
  // define the algebraic data type
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt [@"Int#"]>]

  // plus :: SimpleInt -> SimpleInt -> SimpleInt
  // plus i j = case i of MkSimpleInt ival -> case j of MkSimpleInt jval -> MkSimpleInt (ival +# jval)
  hask.func @plus {
    %lam = hask.lambdaSSA(%i : !hask.thunk<!hask.untyped>, %j: !hask.thunk<!hask.untyped>) {
      %icons = hask.force(%i: !hask.thunk<!hask.untyped>): !hask.value
      %reti = hask.caseSSA %icons 
           [@MkSimpleInt -> { ^entry(%ival: !hask.value):
              %jcons = hask.force(%j: !hask.thunk<!hask.untyped>):!hask.value
              %retj = hask.caseSSA %jcons 
                  [@MkSimpleInt -> { ^entry(%jval: !hask.value):
                        %sum_v = hask.primop_add(%ival, %jval)
                        %boxed = hask.construct(@MkSimpleInt, %sum_v)
                        hask.return(%boxed):!hask.thunk<!hask.untyped>
                  }]
              hask.return(%retj):!hask.thunk<!hask.untyped>
           }]
      hask.return(%reti): !hask.thunk<!hask.untyped>
    }
    hask.return(%lam): !hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) ->  !hask.thunk<!hask.untyped>>
  }

  
  // zero :: SimpleInt; zero = MkSimpleInt 0#
  hask.global @zero {
      %lit_zero = hask.make_i64(0)
      %boxed = hask.construct(@MkSimpleInt, %lit_zero)
      hask.return(%boxed): !hask.thunk<!hask.untyped>
  }

  hask.global @one {
      %v = hask.make_i64(1)
      %boxed = hask.construct(@MkSimpleInt, %v)
      hask.return(%boxed): !hask.thunk<!hask.untyped>
  }

  hask.global @two {
      %lit_two = hask.make_i64(2)
      %boxed = hask.construct(@MkSimpleInt, %lit_two)
      hask.return(%boxed): !hask.thunk<!hask.untyped>
  }

  hask.global @eight {
      %lit8 = hask.make_i64(8)
      %boxed = hask.construct(@MkSimpleInt, %lit8)
      hask.return(%boxed): !hask.thunk<!hask.untyped>
  }


  // 1 + 2 = 3
  hask.func@main {
    %lam = hask.lambdaSSA(%_: !hask.thunk<!hask.untyped>) {
      %input = hask.ref(@one) : !hask.thunk<!hask.untyped>
      %input_t = hask.apSSA(%input: !hask.thunk<!hask.untyped>)

      %input2 = hask.ref(@two) :!hask.thunk<!hask.untyped>
      %input2_t = hask.apSSA(%input2 : !hask.thunk<!hask.untyped>)

      %plus = hask.ref(@plus)  : !hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) ->  !hask.thunk<!hask.untyped>>
      %out_t = hask.apSSA(%plus : !hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) ->!hask.thunk<!hask.untyped>>, %input_t, %input2_t)
      %out_v = hask.force(%out_t : !hask.thunk<!hask.untyped>): !hask.thunk<!hask.untyped>
      hask.return(%out_v) : !hask.thunk<!hask.untyped>
    }
    hask.return (%lam) : !hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
  }
}


// module Fib where
// import GHC.Prim
// data SimpleInt = MkSimpleInt Int#
// 
// plus :: SimpleInt -> SimpleInt -> SimpleInt
// plus i j = case i of MkSimpleInt ival -> case j of MkSimpleInt jval -> MkSimpleInt (ival +# jval)
// 
// 
// minus :: SimpleInt -> SimpleInt -> SimpleInt
// minus i j = case i of MkSimpleInt ival -> case j of MkSimpleInt jval -> MkSimpleInt (ival -# jval)
//                             
// 
// one :: SimpleInt; one = MkSimpleInt 1#
// zero :: SimpleInt; zero = MkSimpleInt 0#
// 
// fib :: SimpleInt -> SimpleInt
// fib i = 
//     case i of
//        MkSimpleInt ihash -> case ihash of 
//                              0# -> zero
//                              1# -> one
//        n -> plus (fib n) (fib (minus n one))
// 
// main :: IO ();
// main = let x = fib one in return ()

