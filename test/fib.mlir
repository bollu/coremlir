// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s
// CHECK: 8
// Core2MLIR: GenMLIR BeforeCorePrep
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
                        hask.return(%boxed) :!hask.thunk<!hask.untyped>
                  }]
              hask.return(%retj):!hask.thunk<!hask.untyped>
           }]
      hask.return(%reti): !hask.thunk<!hask.untyped>
    }
    hask.return(%lam): !hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
  }

  // minus :: SimpleInt -> SimpleInt -> SimpleInt
  // minus i j = case i of MkSimpleInt ival -> case j of MkSimpleInt jval -> MkSimpleInt (ival -# jval)
  hask.func @minus {
    %lam = hask.lambdaSSA(%i : !hask.thunk<!hask.untyped>, %j: !hask.thunk<!hask.untyped>) {
      %icons = hask.force(%i:!hask.thunk<!hask.untyped>):!hask.value
      %reti = hask.caseSSA %icons 
           [@MkSimpleInt -> { ^entry(%ival: !hask.value):
              %jcons = hask.force(%j:!hask.thunk<!hask.untyped>):!hask.value
              %retj = hask.caseSSA %jcons 
                  [@MkSimpleInt -> { ^entry(%jval: !hask.value):
                        // %minus_hash = hask.ref (@"-#")  : !hask.fn<!hask.value, !hask.fn<!hask.value, !hask.thunk<!hask.untyped>>>
                        // %diff_t = hask.apSSA(%minus_hash : !hask.fn<!hask.value, !hask.fn<!hask.value, !hask.thunk<!hask.untyped>>>, %ival, %jval)
                        // %diff_v = hask.force(%diff_t)
                        %diff_v = hask.primop_sub(%ival, %jval)
                        %boxed = hask.construct(@MkSimpleInt, %diff_v)
                        hask.return(%boxed) : !hask.thunk<!hask.untyped>

                  }]
              hask.return(%retj):!hask.thunk<!hask.untyped>
           }]
      hask.return(%reti): !hask.thunk<!hask.untyped>
    }
    hask.return(%lam): !hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
  }


  // TODO: is unit tuple value or thunk?
  // one :: SimpleInt; one = MkSimpleInt 1#
  // This maybe a hack. Perhaps we should represent this as 
  // one :: () -> SimpleInt ; one () = MkSimpleInt 1# (?)
  hask.global @one {
      %lit_one = hask.make_i64(1)
      %boxed = hask.construct(@MkSimpleInt, %lit_one)
      hask.return(%boxed): !hask.thunk<!hask.untyped>
  }
  
  // zero :: SimpleInt; zero = MkSimpleInt 0#
  hask.global @zero {
      %lit_zero = hask.make_i64(0)
      %boxed = hask.construct(@MkSimpleInt, %lit_zero)
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


  // fib :: SimpleInt -> SimpleInt
  // fib i = 
  //     case i of
  //        MkSimpleInt ihash -> 
  //            case ihash of 
  //               0# -> zero
  //               1# -> one
  //               _ -> plus (fib i) (fib (minus i one))
  hask.func @fib {
    %lam = hask.lambdaSSA(%i: !hask.thunk<!hask.untyped>) {
        %icons = hask.force(%i:!hask.thunk<!hask.untyped>):!hask.value
        %ret = hask.caseSSA %icons
               [@MkSimpleInt -> { ^entry(%ihash: !hask.value):
                     %ret = hask.caseint %ihash 
                     [0 -> { ^entry(%_: !hask.value): 
                                %z = hask.ref(@zero) : !hask.thunk<!hask.untyped>
                                %z_t = hask.apSSA(%z: !hask.thunk<!hask.untyped>)
                                %z_v = hask.force(%z_t: !hask.thunk<!hask.untyped>):!hask.thunk<!hask.untyped>
                                hask.return (%z_v): !hask.thunk<!hask.untyped>      
                     }]
                     [1 -> { ^entry(%_: !hask.value): 
                                %o = hask.ref(@one):!hask.thunk<!hask.untyped>
                                %o_t = hask.apSSA(%o: !hask.thunk<!hask.untyped>)
                                %o_v = hask.force(%o_t: !hask.thunk<!hask.untyped>):!hask.thunk<!hask.untyped>
                                hask.return (%o_v): !hask.thunk<!hask.untyped>
                     }]
                     [@default -> { ^entry:
                                // %o_v = hask.force(%o_t: !hask.thunk<!hask.untyped>):!hask.thunk<!hask.untyped>
                                %fib = hask.ref(@fib):  !hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
                                %minus = hask.ref(@minus): !hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>> 
                                %one = hask.ref(@one): !hask.thunk<!hask.untyped>
                                %one_t = hask.apSSA(%one: !hask.thunk<!hask.untyped>)

                                %i_minus_one_t = hask.apSSA(%minus: !hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>, 
                                    %i, %one_t)


                                %fib_i_minus_one_t = hask.apSSA(%fib: !hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>, %i_minus_one_t)
                                %fib_i_minus_one_v = hask.force(%fib_i_minus_one_t :!hask.thunk<!hask.untyped>) :!hask.thunk<!hask.untyped>


                                %two = hask.ref(@two): !hask.thunk<!hask.untyped>
                                %two_t = hask.apSSA(%two: !hask.thunk<!hask.untyped>)

                                %i_minus_two_t = hask.apSSA(%minus: !hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>, 
                                   %i, %two_t)

                                %fib_i_minus_two_t = hask.apSSA(%fib: !hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>, %i_minus_two_t)
                                %fib_i_minus_two_v = hask.force(%fib_i_minus_two_t :!hask.thunk<!hask.untyped>) :!hask.thunk<!hask.untyped>

                                %fib_i_minus_one_v_t = hask.thunkify(%fib_i_minus_one_v: !hask.thunk<!hask.untyped>): !hask.thunk<!hask.untyped>
                                %fib_i_minus_two_v_t = hask.thunkify(%fib_i_minus_two_v: !hask.thunk<!hask.untyped>): !hask.thunk<!hask.untyped>

                                 %plus = hask.ref(@plus) : !hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
                                 %sum = 
                                     hask.apSSA(%plus: !hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>, 
                                         %fib_i_minus_one_v_t, %fib_i_minus_two_v_t)
                                 %sum_v = hask.force(%sum:!hask.thunk<!hask.untyped>):!hask.thunk<!hask.untyped>
                                 hask.return (%sum_v):!hask.thunk<!hask.untyped>
                     }]
                     hask.return(%ret):!hask.thunk<!hask.untyped>
               }]
        hask.return (%ret):!hask.thunk<!hask.untyped>
    }
    hask.return (%lam): !hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
  }


  // ix:  0 1 2 3 4 5 6
  // val: 0 1 1 2 3 5 8
  hask.func@main {
    %lam = hask.lambdaSSA(%_: !hask.thunk<!hask.untyped>) {
      %number = hask.make_i64(6)
      %boxed_number = hask.construct(@MkSimpleInt, %number)
      %thunk_number = hask.thunkify(%boxed_number :!hask.thunk<!hask.untyped>) : !hask.thunk<!hask.untyped>

      %fib = hask.ref(@fib)  : !hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
      %out_t = hask.apSSA(%fib : !hask.fn<(!hask.thunk<!hask.untyped>) ->  !hask.thunk<!hask.untyped>>, %thunk_number)
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

