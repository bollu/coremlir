module {
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt [@"Int#"]>]

  // plus :: SimpleInt -> SimpleInt -> SimpleInt
  // plus i j = case i of MkSimpleInt ival -> case j of MkSimpleInt jval -> MkSimpleInt (ival +# jval)
  hask.func @plus {
    %lami = hask.lambdaSSA(%i) {
         %lamj = hask.lambdaSSA(%j) {
              %reti = hask.caseSSA %i 
                   [@SimpleInt -> { ^entry(%ival: !hask.untyped):
                      %retj = hask.caseSSA %j 
                          [@SimpleInt -> { ^entry(%jval: !hask.untyped):
                                %plus_hash = hask.ref (@"+#")
                                %i_plus = hask.apSSA(%plus_hash, %ival)
                                %i_plus_j = hask.apSSA(%i_plus, %jval)
                                // really, we need another case here.
                                %mk_simple_int = hask.ref (@MkSimpleInt)
                                %boxed = hask.apSSA(%mk_simple_int, %i_plus_j)
                                hask.return(%boxed)
                          }]
                      hask.return(%retj)
                   }]
              hask.return(%reti)
          }
          hask.return(%lamj)
    }
    hask.return(%lami)
  }

  // minus :: SimpleInt -> SimpleInt -> SimpleInt
  // minus i j = case i of MkSimpleInt ival -> case j of MkSimpleInt jval -> MkSimpleInt (ival -# jval)
  hask.func @minus {
    %lami = hask.lambdaSSA(%i) {
         %lamj = hask.lambdaSSA(%j) {
              %reti = hask.caseSSA %i 
                   [@SimpleInt -> { ^entry(%ival: !hask.untyped):
                      %retj = hask.caseSSA %j 
                          [@SimpleInt -> { ^entry(%jval: !hask.untyped):
                                %minus_hash = hask.ref (@"-#")
                                %i_sub = hask.apSSA(%minus_hash, %ival)
                                %i_sub_j = hask.apSSA(%i_sub, %jval)
                                // really we need another case here
                                %mk_simple_int = hask.ref (@MkSimpleInt)
                                %boxed = hask.apSSA(%mk_simple_int, %i_sub_j)
                                hask.return(%boxed)
                          }]
                      hask.return(%retj)
                   }]
              hask.return(%reti)
          }
          hask.return(%lamj)
    }
    hask.return(%lami)
  }


  // one :: SimpleInt; one = MkSimpleInt 1#
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
  //        MkSimpleInt 0# -> zero ()
  //        MkSimpleInt 1# -> one ()
  //        n -> let 
  //               oneval = one ()
  //               ndec = minus n oneval
  //               fibn = fib n
  //               fib_ndec = fib ndec 
  //             in plus fibn fib_ndec
  // TODO: show capturing of bound variables.
  hask.func @fib {
    %lam = hask.lambdaSSA(%i) {
        %ret = hask.caseSSA %i 
               [@MkSimpleInt -> { ^entry(%ihash: !hask.untyped):
                     %ret = hask.caseSSA %ihash 
                     [0 -> { ^entry(%_: !hask.untyped): 
                                %z = hask.ref(@zero)
                                hask.return (%z)
                     }]
                     [1 -> { ^entry(%_: !hask.untyped): 
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

// {-# LANGUAGE MagicHash #-}
// {-# LANGUAGE UnboxedTuples #-}
// module Fib where
// import GHC.Prim
// data SimpleInt = MkSimpleInt Int#
// 
// plus :: SimpleInt -> SimpleInt -> SimpleInt
// plus i j = case i of MkSimpleInt ival -> case j of MkSimpleInt jval -> case ival +# jval of sum -> MkSimpleInt sum
// 
// minus :: SimpleInt -> SimpleInt -> SimpleInt
// minus i j = case i of MkSimpleInt ival -> case j of MkSimpleInt jval -> case ival -# jval of diff -> MkSimpleInt diff
//                             
// 
// one :: () -> SimpleInt; one () = MkSimpleInt 1#
// zero :: () -> SimpleInt; zero () = MkSimpleInt 0#
// 
// fib :: SimpleInt -> SimpleInt
// fib i = 
//     case i of
//        MkSimpleInt 0# -> zero ()
//        MkSimpleInt 1# -> one ()
//        n -> let 
//               oneval = one ()
//               ndec = minus n oneval
//               fibn = fib n
//               fib_ndec = fib ndec 
//             in plus fibn fib_ndec

