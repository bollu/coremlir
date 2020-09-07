// Fib
// Core2MLIR: GenMLIR BeforeCorePrep
module {
  // should it be Attr Attr, with the "list" embedded as an attribute,
  // or should it be Attr [Attr]? Who really knows :(
  // define the algebraic data type
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt [@"Int#"]>]

  // plus :: SimpleInt -> SimpleInt -> SimpleInt
  // plus i j = case i of MkSimpleInt ival -> case j of MkSimpleInt jval -> MkSimpleInt (ival +# jval)
  hask.func @plus {
    %lami = hask.lambdaSSA(%i : !hask.thunk) {
         %lamj = hask.lambdaSSA(%j: !hask.thunk) {
              %icons = hask.force(%i)
              %reti = hask.caseSSA %icons 
                   [@SimpleInt -> { ^entry(%ival: !hask.value):
                      %jcons = hask.force(%j)
                      %retj = hask.caseSSA %jcons 
                          [@SimpleInt -> { ^entry(%jval: !hask.value):
                                %plus_hash = hask.ref (@"+#")
                                %i_plus = hask.apSSA(%plus_hash, %ival)
                                %i_plus_j = hask.apSSA(%i_plus, %jval)
                                %mk_simple_int = hask.ref (@MkSimpleInt)
                                // make a constructor with a single value
                                %boxed = hask.construct(@MkSimpleInt, %i_plus_j)
                                hask.return(%boxed) :!hask.thunk
                          }]
                      hask.return(%retj):!hask.thunk
                   }]
              hask.return(%reti): !hask.thunk
          }
          hask.return(%lamj): !hask.thunk
    }
    hask.return(%lami): !hask.thunk
  }

  // minus :: SimpleInt -> SimpleInt -> SimpleInt
  // minus i j = case i of MkSimpleInt ival -> case j of MkSimpleInt jval -> MkSimpleInt (ival -# jval)
  hask.func @minus {
    %lami = hask.lambdaSSA(%i: !hask.thunk) {
         %lamj = hask.lambdaSSA(%j :!hask.thunk) {
              %icons = hask.force(%i)
              %reti = hask.caseSSA %icons 
                   [@SimpleInt -> { ^entry(%ival: !hask.value):
                      %jcons = hask.force(%j)
                      %retj = hask.caseSSA %jcons 
                          [@SimpleInt -> { ^entry(%jval: !hask.value):
                                %minus_hash = hask.ref (@"-#")
                                %i_sub = hask.apSSA(%minus_hash, %ival)
                                %i_sub_j = hask.apSSA(%i_sub, %jval)
                                // really we need another case here
                                %mk_simple_int = hask.ref (@MkSimpleInt)
                                // what do now?
                                %boxed = hask.apSSA(%mk_simple_int, %i_sub_j)
                                hask.return(%boxed) :!hask.thunk
                          }]
                      hask.return(%retj) :!hask.thunk
                   }]
              hask.return(%reti):!hask.thunk
          }
          hask.return(%lamj):!hask.thunk
    }
    hask.return(%lami):!hask.thunk
  }


  // one :: SimpleInt; one = MkSimpleInt 1#
  // This maybe a hack. Perhaps we should represent this as 
  // one :: () -> SimpleInt ; one () = MkSimpleInt 1# (?)
  hask.global @one {
    %mk_simple_int = hask.ref (@MkSimpleInt)
    %lit_one = hask.make_i64(1)
    %boxed = hask.apSSA(%mk_simple_int, %lit_one)
    hask.return(%boxed)
  }
  
  // zero :: SimpleInt; zero = MkSimpleInt 0#
  hask.global @zero {
    %mk_simple_int = hask.ref (@MkSimpleInt)
    %lit_zero = hask.make_i64(0)
    %boxed = hask.apSSA(%mk_simple_int, %lit_zero)
    hask.return(%boxed)
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
    %lam = hask.lambdaSSA(%i) {
        %ret = hask.caseSSA %i 
               [@MkSimpleInt -> { ^entry(%ihash: !hask.value):
                     %ret = hask.caseSSA %ihash 
                     [0 -> { ^entry(%_: !hask.value): 
                                %z = hask.ref(@zero)
                                hask.return (%z)
                     }]
                     [1 -> { ^entry(%_: !hask.value): 
                                %o = hask.ref(@one)
                                hask.return (%o)
                     }]
                     [@default -> { ^entry:
                                     %fib_ref = hask.ref(@fib)
                                     %fib_i = hask.apSSA(%fib_ref, %i)
                                     %minus_ref = hask.ref(@minus)
                                     %i_minus = hask.apSSA(%minus_ref, %i)
                                     %one_ref = hask.ref(@one)
                                     %i_minus_one = hask.apSSA(%i_minus, %one_ref)
                                     %fib_i_minus_one = hask.apSSA(%fib_ref, %i_minus_one)
                                     %plus_ref = hask.ref(@plus)
                                     %fib_i_plus = hask.apSSA(%plus_ref, %fib_i)
                                     %fib_i_plus_fib_i_minus_one = hask.apSSA(%fib_i_plus, %fib_i_minus_one)
                                     hask.return (%fib_i_plus_fib_i_minus_one)

                     }]
                     hask.return(%ret)
               }]
        hask.return (%ret)
    }
    hask.return (%lam)
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

