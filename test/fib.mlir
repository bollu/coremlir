// Fib
// Core2MLIR: GenMLIR BeforeCorePrep
module {
  // should it be Attr Attr, with the "list" embedded as an attribute,
  // or should it be Attr [Attr]? Who really knows :(
  // define the algebraic data type
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt, [@"Int#"]>]
  // how do we represent this?

  hask.func @plus {
  %lambda_0 = hask.lambdaSSA(%i_a12Q) {
    %lambda_1 = hask.lambdaSSA(%j_a12R) {
      %case_2 = hask.caseSSA  %i_a12Q
      [@"MkSimpleInt" -> { ^entry(%wild_00: !hask.untyped, %ival_a12S: !hask.untyped):
        %case_3 = hask.caseSSA  %j_a12R
        [@"MkSimpleInt" -> { ^entry(%wild_X5: !hask.untyped, %jval_a12T: !hask.untyped):
          %app_4 = hask.apSSA(@"+#", %ival_a12S)
          %app_5 = hask.apSSA(%app_4, %jval_a12T)
          %app_6 = hask.apSSA(%MkSimpleInt, %app_5)
          hask.return(%app_6)
        }
        ]
        hask.return(%case_3)
      }
      ]
      hask.return(%case_2)
    }
    hask.return(%lambda_1)
  }
  hask.return(%lambda_0)
  }
  hask.func @minus {
  %lambda_7 = hask.lambdaSSA(%i_a12U) {
    %lambda_8 = hask.lambdaSSA(%j_a12V) {
      %case_9 = hask.caseSSA  %i_a12U
      [@"MkSimpleInt" ->
      {
      ^entry(%wild_00: !hask.untyped, %ival_a12W: !hask.untyped):
        %case_10 = hask.caseSSA  %j_a12V
        [@"MkSimpleInt" ->
        {
        ^entry(%wild_X6: !hask.untyped, %jval_a12X: !hask.untyped):
          %app_11 = hask.apSSA(@"-#", %ival_a12W)
          %app_12 = hask.apSSA(%app_11, %jval_a12X)
          %app_13 = hask.apSSA(%MkSimpleInt, %app_12)
        hask.return(%app_13)
        }
        ]
      hask.return(%case_10)
      }
      ]
      hask.return(%case_9)
    }
    hask.return(%lambda_8)
  }
  hask.return(%lambda_7)
  }
  hask.func @one {
  %lit_14 = hask.make_i64(1)
  %app_15 = hask.apSSA(%MkSimpleInt, %lit_14)
  hask.return(%app_15)
  }
  hask.func @zero {
  %lit_16 = hask.make_i64(0)
  %app_17 = hask.apSSA(%MkSimpleInt, %lit_16)
  hask.return(%app_17)
  }
  hask.func @fib {
  %lambda_18 = hask.lambdaSSA(%i_a12Y) {
    %case_19 = hask.caseSSA  %i_a12Y
    [@"MkSimpleInt" ->
    {
    ^entry(%wild_00: !hask.untyped, %ds_d1ky: !hask.untyped):
      %case_20 = hask.caseSSA  %ds_d1ky
      ["default" ->
      {
      ^entry(%ds_X1kH: !hask.untyped):
        %app_21 = hask.apSSA(@fib, %i_a12Y)
        %app_22 = hask.apSSA(@plus, %app_21)
        %app_23 = hask.apSSA(@minus, %i_a12Y)
        %app_24 = hask.apSSA(%app_23, @one)
        %app_25 = hask.apSSA(@fib, %app_24)
        %app_26 = hask.apSSA(%app_22, %app_25)
      hask.return(%app_26)
      }
      ]
      [0 ->
      {
      ^entry(%ds_X1kH: !hask.untyped):
      hask.return(@zero)
      }
      ]
      [1 ->
      {
      ^entry(%ds_X1kH: !hask.untyped):
      hask.return(@one)
      }
      ]
    hask.return(%case_20)
    }
    ]
    hask.return(%case_19)
  }
  hask.return(%lambda_18)
  }
}

// ==== Haskell source ===
// {-# LANGUAGE MagicHash #-}
// {-# LANGUAGE UnboxedTuples #-}
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
//        MkSimpleInt 0# -> zero
//        MkSimpleInt 1# -> one
//        n -> plus (fib n) (fib (minus n one))
// 
// main :: IO ();
// main = let x = fib one in return ()
