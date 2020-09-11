// Test that case of int works.
module {

  hask.func @prec {
    %lam = hask.lambdaSSA(%ihash: !hask.value) {
     // do we know that %ihash is an int here?
     %ret = hask.caseint %ihash 
     [0 -> { ^entry(%ival: !hask.value): 
                hask.return (%ival): !hask.value      
     }]
     [@default -> { ^entry(%ival: !hask.value): // ... or here?
                     %lit_one = hask.make_i64(1)
                     %pred = hask.primop_sub(%ival, %lit_one)
                     hask.return(%pred): !hask.value

     }]
     hask.return (%ret) : !hask.value
    }
    hask.return (%lam): !hask.fn<!hask.value, !hask.value>
  }

  hask.func @main {
    %lambda = hask.lambdaSSA(%_: !hask.thunk) {
      %lit_42 = hask.make_i64(42)
      %prec = hask.ref(@prec)  : !hask.fn<!hask.value, !hask.value>
      %out_v = hask.apSSA(%prec : !hask.fn<!hask.value, !hask.value>, %lit_42)
      %out_v_forced = hask.force(%out_v : !hask.value): !hask.value
      %x = hask.construct(@X, %out_v_forced)
      hask.return(%x) : !hask.thunk
    }
    hask.return(%lambda) :!hask.fn<!hask.thunk, !hask.thunk>
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


