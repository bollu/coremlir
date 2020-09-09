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
                                %plus_hash = hask.ref (@"+#")  : !hask.fn<!hask.value, !hask.fn<!hask.value, !hask.value>>
                                %i_plus = hask.apSSA(%plus_hash: !hask.fn<!hask.value, !hask.fn<!hask.value, !hask.value>>, %ival)
                                %i_plus_j = hask.apSSA(%i_plus : !hask.fn<!hask.value, !hask.value>, %jval)
                                %mk_simple_int = hask.ref (@MkSimpleInt)  :!hask.fn<!hask.value, !hask.value>
                                // make a constructor with a single value
                                %boxed = hask.construct(@MkSimpleInt, %i_plus_j)
                                hask.return(%boxed) :!hask.thunk
                          }]
                      hask.return(%retj):!hask.thunk
                   }]
              hask.return(%reti): !hask.thunk
          }
          hask.return(%lamj): !hask.fn<!hask.thunk, !hask.thunk>
    }
    hask.return(%lami): !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
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
                                %minus_hash = hask.ref (@"-#") : !hask.fn<!hask.value, !hask.fn<!hask.value, !hask.value>>
                                %i_sub = hask.apSSA(%minus_hash : !hask.fn<!hask.value, !hask.fn<!hask.value, !hask.value>>, %ival)
                                %i_sub_j = hask.apSSA(%i_sub : !hask.fn<!hask.value, !hask.value>, %jval)
                                // really we need another case here
                                %mk_simple_int = hask.ref (@MkSimpleInt) :!hask.fn<!hask.value, !hask.thunk>
                                // what do now?
                                %boxed = hask.apSSA(%mk_simple_int:!hask.fn<!hask.value, !hask.thunk>  , %i_sub_j)
                                hask.return(%boxed) :!hask.thunk
                          }]
                      hask.return(%retj) :!hask.thunk
                   }]
              hask.return(%reti):!hask.thunk
          }
          hask.return(%lamj): !hask.fn<!hask.thunk, !hask.thunk>
    }
    hask.return(%lami): !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
  }


  // TODO: is unit tuple value or thunk?
  // one :: SimpleInt; one = MkSimpleInt 1#
  // This maybe a hack. Perhaps we should represent this as 
  // one :: () -> SimpleInt ; one () = MkSimpleInt 1# (?)
  hask.global @one {
      %mk_simple_int = hask.ref (@MkSimpleInt) :!hask.fn<!hask.value, !hask.thunk>
      %lit_one = hask.make_i64(1)
      %boxed = hask.apSSA(%mk_simple_int :!hask.fn<!hask.value, !hask.thunk>, %lit_one)
      hask.return(%boxed): !hask.thunk
  }
  
  // zero :: SimpleInt; zero = MkSimpleInt 0#
  hask.global @zero {
      %mk_simple_int = hask.ref (@MkSimpleInt) :!hask.fn<!hask.value, !hask.thunk>
      %lit_zero = hask.make_i64(0)
      %boxed = hask.apSSA(%mk_simple_int :!hask.fn<!hask.value, !hask.thunk>, %lit_zero)
      hask.return(%boxed): !hask.thunk
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
    %lam = hask.lambdaSSA(%i: !hask.thunk) {
        %icons = hask.force(%i)
        %ret = hask.caseSSA %icons
               [@MkSimpleInt -> { ^entry(%ihash: !hask.value):
                     %ret = hask.caseSSA %ihash 
                     [0 -> { ^entry(%_: !hask.value): 
                                %z = hask.ref(@zero) : !hask.thunk
                                hask.return (%z): !hask.thunk      
                     }]
                     [1 -> { ^entry(%_: !hask.value): 
                                %o = hask.ref(@one):!hask.thunk
                                hask.return (%o): !hask.thunk
                     }]
                     [@default -> { ^entry:
                                     %fib_ref = hask.ref(@fib):  !hask.fn<!hask.thunk, !hask.thunk>
                                     %fib_i = hask.apSSA(%fib_ref: !hask.fn<!hask.thunk, !hask.thunk>, %i)
                                     %minus_ref = hask.ref(@minus): !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>> 
                                     %i_minus = hask.apSSA(%minus_ref: !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>> , %i)
                                     %one_ref = hask.ref(@one): !hask.thunk
                                     %i_minus_one = hask.apSSA(%i_minus : !hask.fn<!hask.thunk, !hask.thunk> , %one_ref)
                                     %fib_i_minus_one = hask.apSSA(%fib_ref: !hask.fn<!hask.thunk, !hask.thunk>, %i_minus_one)
                                     %plus_ref = hask.ref(@plus) : !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
                                     %fib_i_plus = hask.apSSA(%plus_ref: !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>, %fib_i)
                                     %fib_i_plus_fib_i_minus_one = hask.apSSA(%fib_i_plus : !hask.fn<!hask.thunk, !hask.thunk>, %fib_i_minus_one)
                                     hask.return (%fib_i_plus_fib_i_minus_one):!hask.thunk

                     }]
                     hask.return(%ret):!hask.thunk
               }]
        hask.return (%ret):!hask.thunk
    }
    hask.return (%lam): !hask.fn<!hask.thunk, !hask.thunk>
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

