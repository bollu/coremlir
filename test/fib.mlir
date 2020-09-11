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
    %lam = hask.lambdaSSA(%i : !hask.thunk, %j: !hask.thunk) {
      %icons = hask.force(%i: !hask.thunk): !hask.value
      %reti = hask.caseSSA %icons 
           [@SimpleInt -> { ^entry(%ival: !hask.value):
              %jcons = hask.force(%j: !hask.thunk):!hask.value
              %retj = hask.caseSSA %jcons 
                  [@SimpleInt -> { ^entry(%jval: !hask.value):
                        // %plus_hash = hask.ref (@"+#")  : !hask.fn<!hask.value, !hask.fn<!hask.value, !hask.thunk>>
                        // %sum_t = hask.apSSA(%plus_hash : !hask.fn<!hask.value, !hask.fn<!hask.value, !hask.thunk>>, %ival, %jval)
                        %sum_v = hask.primop_add(%ival, %jval)
                        %boxed = hask.construct(@MkSimpleInt, %sum_v)
                        hask.return(%boxed) :!hask.thunk
                  }]
              hask.return(%retj):!hask.thunk
           }]
      hask.return(%reti): !hask.thunk
    }
    hask.return(%lam): !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
  }

  // minus :: SimpleInt -> SimpleInt -> SimpleInt
  // minus i j = case i of MkSimpleInt ival -> case j of MkSimpleInt jval -> MkSimpleInt (ival -# jval)
  hask.func @minus {
    %lam = hask.lambdaSSA(%i : !hask.thunk, %j: !hask.thunk) {
      %icons = hask.force(%i:!hask.thunk):!hask.value
      %reti = hask.caseSSA %icons 
           [@SimpleInt -> { ^entry(%ival: !hask.value):
              %jcons = hask.force(%j:!hask.thunk):!hask.value
              %retj = hask.caseSSA %jcons 
                  [@SimpleInt -> { ^entry(%jval: !hask.value):
                        // %minus_hash = hask.ref (@"-#")  : !hask.fn<!hask.value, !hask.fn<!hask.value, !hask.thunk>>
                        // %diff_t = hask.apSSA(%minus_hash : !hask.fn<!hask.value, !hask.fn<!hask.value, !hask.thunk>>, %ival, %jval)
                        // %diff_v = hask.force(%diff_t)
                        %diff_v = hask.primop_sub(%ival, %jval)
                        %boxed = hask.construct(@MkSimpleInt, %diff_v)
                        hask.return(%boxed) :!hask.thunk
                  }]
              hask.return(%retj):!hask.thunk
           }]
      hask.return(%reti): !hask.thunk
    }
    hask.return(%lam): !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
  }


  // TODO: is unit tuple value or thunk?
  // one :: SimpleInt; one = MkSimpleInt 1#
  // This maybe a hack. Perhaps we should represent this as 
  // one :: () -> SimpleInt ; one () = MkSimpleInt 1# (?)
  hask.global @one {
      %lit_one = hask.make_i64(1)
      %boxed = hask.construct(@MkSimpleInt, %lit_one)
      hask.return(%boxed): !hask.thunk
  }
  
  // zero :: SimpleInt; zero = MkSimpleInt 0#
  hask.global @zero {
      %lit_zero = hask.make_i64(0)
      %boxed = hask.construct(@MkSimpleInt, %lit_zero)
      hask.return(%boxed): !hask.thunk
  }


  hask.global @two {
      %lit_two = hask.make_i64(2)
      %boxed = hask.construct(@MkSimpleInt, %lit_two)
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
        %icons = hask.force(%i:!hask.thunk):!hask.value
        %ret = hask.caseSSA %icons
               [@MkSimpleInt -> { ^entry(%ihash: !hask.value):
                     %ret = hask.caseint %ihash 
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
                                     %one_ref = hask.ref(@one): !hask.thunk
                                     %i_minus_one = hask.apSSA(%minus_ref: !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>, 
                                        %i, %one_ref)
                                     %fib_i_minus_one = hask.apSSA(%fib_ref: !hask.fn<!hask.thunk, !hask.thunk>, %i_minus_one_val)
                                     %plus_ref = hask.ref(@plus) : !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
                                     %fib_i_plus_fib_i_minus_one = 
                                        hask.apSSA(%plus_ref: !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>, 
                                            %fib_i, %fib_i_minus_one)
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

