// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s
// CHECK: 8
// Core2MLIR: GenMLIR BeforeCorePrep
module {
  // should it be Attr Attr, with the "list" embedded as an attribute,
  // or should it be Attr [Attr]? Who really knows :(
  // define the algebraic data type
  hask.adt @SimpleInt [#hask.data_constructor<@SimpleInt [@"Int#"]>]

  // plus :: SimpleInt -> SimpleInt -> SimpleInt
  // plus i j = case i of SimpleInt ival -> case j of SimpleInt jval -> SimpleInt (ival +# jval)
  hask.func @plus {
    %lam = hask.lambdaSSA(%i : !hask.thunk<!hask.adt<@SimpleInt>>, %j: !hask.thunk<!hask.adt<@SimpleInt>>) {
      %icons = hask.force(%i: !hask.thunk<!hask.adt<@SimpleInt>>): !hask.adt<@SimpleInt>
      %reti = hask.caseSSA @SimpleInt %icons 
           [@SimpleInt -> { ^entry(%ival: !hask.value):
              %jcons = hask.force(%j: !hask.thunk<!hask.adt<@SimpleInt>>):!hask.adt<@SimpleInt>
              %retj = hask.caseSSA @SimpleInt %jcons 
                  [@SimpleInt -> { ^entry(%jval: !hask.value):
                        %sum_v = hask.primop_add(%ival, %jval)
                        %boxed = hask.construct(@SimpleInt, %sum_v: !hask.value)
                        hask.return(%boxed) : !hask.adt<@SimpleInt>
                  }]
              hask.return(%retj): !hask.adt<@SimpleInt>
           }]
      hask.return(%reti): !hask.adt<@SimpleInt>
    }
    hask.return(%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }

  // minus :: SimpleInt -> SimpleInt -> SimpleInt
  // minus i j = case i of SimpleInt ival -> case j of SimpleInt jval -> SimpleInt (ival -# jval)
  hask.func @minus {
    %lam = hask.lambdaSSA(%i : !hask.thunk<!hask.adt<@SimpleInt>>, %j: !hask.thunk<!hask.adt<@SimpleInt>>) {
      %icons = hask.force(%i:!hask.thunk<!hask.adt<@SimpleInt>>):!hask.adt<@SimpleInt>
      %reti = hask.caseSSA @SimpleInt %icons 
           [@SimpleInt -> { ^entry(%ival: !hask.value):
              %jcons = hask.force(%j:!hask.thunk<!hask.adt<@SimpleInt>>):!hask.adt<@SimpleInt>
              %retj = hask.caseSSA @SimpleInt %jcons 
                  [@SimpleInt -> { ^entry(%jval: !hask.value):
                        %diff_v = hask.primop_sub(%ival, %jval)
                        %boxed = hask.construct(@SimpleInt, %diff_v: !hask.value)
                        hask.return(%boxed) : !hask.adt<@SimpleInt>

                  }]
              hask.return(%retj):!hask.adt<@SimpleInt>
           }]
      hask.return(%reti): !hask.adt<@SimpleInt>
    }
    hask.return(%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }


  hask.func @zero {
     %lam = hask.lambdaSSA() {
       %v = hask.make_i64(0)
       %boxed = hask.construct(@SimpleInt, %v:!hask.value)
       hask.return(%boxed): !hask.adt<@SimpleInt>
     }
     hask.return(%lam) : !hask.fn<() -> !hask.adt<@SimpleInt>>
  }
  
  hask.func @one {
     %lam = hask.lambdaSSA() {
       %v = hask.make_i64(1)
       %boxed = hask.construct(@SimpleInt, %v:!hask.value)
       hask.return(%boxed): !hask.adt<@SimpleInt>
     }
     hask.return(%lam) : !hask.fn<() -> !hask.adt<@SimpleInt>>
  }


  hask.func @two {
     %lam = hask.lambdaSSA() {
       %v = hask.make_i64(2)
       %boxed = hask.construct(@SimpleInt, %v:!hask.value)
       hask.return(%boxed): !hask.adt<@SimpleInt>
     }
     hask.return(%lam) : !hask.fn<() -> !hask.adt<@SimpleInt>>
  }

  hask.func @eight {
     %lam = hask.lambdaSSA() {
       %v = hask.make_i64(8)
       %boxed = hask.construct(@SimpleInt, %v:!hask.value)
       hask.return(%boxed): !hask.adt<@SimpleInt>
     }
     hask.return(%lam) : !hask.fn<() -> !hask.adt<@SimpleInt>>
  }


  // fib :: SimpleInt -> SimpleInt
  // fib i = 
  //     case i of
  //        SimpleInt ihash -> 
  //            case ihash of 
  //               0# -> zero
  //               1# -> one
  //               _ -> plus (fib i) (fib (minus i one))
  hask.func @fib {
    %lam = hask.lambdaSSA(%i: !hask.thunk<!hask.adt<@SimpleInt>>) {
        %icons = hask.force(%i:!hask.thunk<!hask.adt<@SimpleInt>>):!hask.adt<@SimpleInt>
        %ret = hask.caseSSA @SimpleInt %icons
               [@SimpleInt -> { ^entry(%ihash: !hask.value):
                     %ret = hask.caseint %ihash 
                     [0 -> { ^entry(%_: !hask.value): 
                                %z = hask.ref(@zero) : !hask.fn<() -> !hask.adt<@SimpleInt>>
                                %z_t = hask.apSSA(%z: !hask.fn<() -> !hask.adt<@SimpleInt>>)
                                %z_v = hask.force(%z_t: !hask.thunk<!hask.adt<@SimpleInt>>): !hask.adt<@SimpleInt>
                                hask.return (%z_v): !hask.adt<@SimpleInt>
                     }]
                     [1 -> { ^entry(%_: !hask.value): 
                                %o = hask.ref(@one):!hask.fn<() -> !hask.adt<@SimpleInt>>
                                %o_t = hask.apSSA(%o: !hask.fn<() -> !hask.adt<@SimpleInt>>)
                                %o_v = hask.force(%o_t: !hask.thunk<!hask.adt<@SimpleInt>>): !hask.adt<@SimpleInt>
                                hask.return (%o_v): !hask.adt<@SimpleInt>
                     }]
                     [@default -> { ^entry:
                                // %o_v = hask.force(%o_t: !hask.thunk<!hask.adt<@SimpleInt>>):!hask.thunk<!hask.adt<@SimpleInt>>
                                %fib = hask.ref(@fib):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                                %minus = hask.ref(@minus): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>> 
                                %one = hask.ref(@one): !hask.fn<() -> !hask.adt<@SimpleInt>>
                                %one_t = hask.apSSA(%one: !hask.fn<() -> !hask.adt<@SimpleInt>>)

                                %i_minus_one_t = hask.apSSA(%minus: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %i, %one_t)

                                %fib_i_minus_one_t = hask.apSSA(%fib: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %i_minus_one_t)
                                %fib_i_minus_one_v = hask.force(%fib_i_minus_one_t :!hask.thunk<!hask.adt<@SimpleInt>>) : !hask.adt<@SimpleInt>


                                %two = hask.ref(@two): !hask.fn<() -> !hask.adt<@SimpleInt>>
                                %two_t = hask.apSSA(%two: !hask.fn<() -> !hask.adt<@SimpleInt>>)

                                %i_minus_two_t = hask.apSSA(%minus: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
                                   %i, %two_t)

                                %fib_i_minus_two_t = hask.apSSA(%fib: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %i_minus_two_t)
                                %fib_i_minus_two_v = hask.force(%fib_i_minus_two_t :!hask.thunk<!hask.adt<@SimpleInt>>) : !hask.adt<@SimpleInt>

                                %fib_i_minus_one_v_t = hask.thunkify(%fib_i_minus_one_v: !hask.adt<@SimpleInt>): !hask.thunk<!hask.adt<@SimpleInt>>
                                %fib_i_minus_two_v_t = hask.thunkify(%fib_i_minus_two_v: !hask.adt<@SimpleInt>): !hask.thunk<!hask.adt<@SimpleInt>>

                                 %plus = hask.ref(@plus) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                                 %sum = 
                                     hask.apSSA(%plus: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
                                         %fib_i_minus_one_v_t, %fib_i_minus_two_v_t)
                                 %sum_v = hask.force(%sum:!hask.thunk<!hask.adt<@SimpleInt>>): !hask.adt<@SimpleInt>
                                 hask.return (%sum_v): !hask.adt<@SimpleInt>
                     }]
                     hask.return(%ret): !hask.adt<@SimpleInt>
               }]
        hask.return (%ret):!hask.adt<@SimpleInt>
    }
    hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }


  // ix:  0 1 2 3 4 5 6
  // val: 0 1 1 2 3 5 8
  hask.func@main {
    %lam = hask.lambdaSSA(%_: !hask.thunk<!hask.adt<@SimpleInt>>) {
      %number = hask.make_i64(6)
      %boxed_number = hask.construct(@SimpleInt, %number: !hask.value)
      %thunk_number = hask.thunkify(%boxed_number: !hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>

      %fib = hask.ref(@fib)  : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
      %out_t = hask.apSSA(%fib : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) ->  !hask.adt<@SimpleInt>>, %thunk_number)
      %out_v = hask.force(%out_t : !hask.thunk<!hask.adt<@SimpleInt>>): !hask.adt<@SimpleInt>
      hask.return(%out_v) : !hask.adt<@SimpleInt>
    }
    hask.return (%lam) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }
    
}


// module Fib where
// import GHC.Prim
// data SimpleInt = SimpleInt Int#
// 
// plus :: SimpleInt -> SimpleInt -> SimpleInt
// plus i j = case i of SimpleInt ival -> case j of SimpleInt jval -> SimpleInt (ival +# jval)
// 
// 
// minus :: SimpleInt -> SimpleInt -> SimpleInt
// minus i j = case i of SimpleInt ival -> case j of SimpleInt jval -> SimpleInt (ival -# jval)
//                             
// 
// one :: SimpleInt; one = SimpleInt 1#
// zero :: SimpleInt; zero = SimpleInt 0#
// 
// fib :: SimpleInt -> SimpleInt
// fib i = 
//     case i of
//        SimpleInt ihash -> case ihash of 
//                              0# -> zero
//                              1# -> one
//        n -> plus (fib n) (fib (minus n one))
// 
// main :: IO ();
// main = let x = fib one in return ()

