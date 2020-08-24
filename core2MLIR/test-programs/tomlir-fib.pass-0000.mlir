// Fib
// Core2MLIR: GenMLIR BeforeCorePrep
module {
    hask.make_data_constructor @"+#"
    hask.make_data_constructor @"-#"
    hask.make_data_constructor @"()"
  hask.func @plus {
  %lambda_0 = hask.lambdaSSA(%i_a12Q) {
    %lambda_1 = hask.lambdaSSA(%j_a12R) {
      %case_2 = hask.caseSSA  %i_a12Q
      [@"MkSimpleInt" ->
      {
      ^entry(%wild_00: !hask.untyped, %ival_a12S: !hask.untyped):
        %case_3 = hask.caseSSA  %j_a12R
        [@"MkSimpleInt" ->
        {
        ^entry(%wild_X5: !hask.untyped, %jval_a12T: !hask.untyped):
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
  hask.func @$trModule {
  %lit_27 = hask.make_string("main")
  %app_28 = hask.apSSA(%TrNameS, %lit_27)
  %app_29 = hask.apSSA(%Module, %app_28)
  %lit_30 = hask.make_string("Fib")
  %app_31 = hask.apSSA(%TrNameS, %lit_30)
  %app_32 = hask.apSSA(%app_29, %app_31)
  hask.return(%app_32)
  }
  hask.func @$krep_a1k5 {
  %app_33 = hask.apSSA(%KindRepTyConApp, %$tcInt#)
  %type_34 = hask.make_string("TYPEINFO_ERASED")
  %app_35 = hask.apSSA(%[], %type_34)
  %app_36 = hask.apSSA(%app_33, %app_35)
  hask.return(%app_36)
  }
  hask.func @$tcSimpleInt {
  %lit_37 = 7748924056139856202
  %app_38 = hask.apSSA(%TyCon, %lit_37)
  %lit_39 = 3495965821938844824
  %app_40 = hask.apSSA(%app_38, %lit_39)
  %app_41 = hask.apSSA(%app_40, @$trModule)
  %lit_42 = hask.make_string("SimpleInt")
  %app_43 = hask.apSSA(%TrNameS, %lit_42)
  %app_44 = hask.apSSA(%app_41, %app_43)
  %lit_45 = hask.make_i64(0)
  %app_46 = hask.apSSA(%app_44, %lit_45)
  %app_47 = hask.apSSA(%app_46, %krep$*)
  hask.return(%app_47)
  }
  hask.func @$krep_a1k6 {
  %app_48 = hask.apSSA(%KindRepTyConApp, @$tcSimpleInt)
  %type_49 = hask.make_string("TYPEINFO_ERASED")
  %app_50 = hask.apSSA(%[], %type_49)
  %app_51 = hask.apSSA(%app_48, %app_50)
  hask.return(%app_51)
  }
  hask.func @$krep_a1k4 {
  %app_52 = hask.apSSA(%KindRepFun, @$krep_a1k5)
  %app_53 = hask.apSSA(%app_52, @$krep_a1k6)
  hask.return(%app_53)
  }
  hask.func @$tc'MkSimpleInt {
  %lit_54 = 4702169295613205611
  %app_55 = hask.apSSA(%TyCon, %lit_54)
  %lit_56 = 5219586552818290360
  %app_57 = hask.apSSA(%app_55, %lit_56)
  %app_58 = hask.apSSA(%app_57, @$trModule)
  %lit_59 = hask.make_string("'MkSimpleInt")
  %app_60 = hask.apSSA(%TrNameS, %lit_59)
  %app_61 = hask.apSSA(%app_58, %app_60)
  %lit_62 = hask.make_i64(0)
  %app_63 = hask.apSSA(%app_61, %lit_62)
  %app_64 = hask.apSSA(%app_63, @$krep_a1k4)
  hask.return(%app_64)
  }
  hask.func @main {
  %type_65 = hask.make_string("TYPEINFO_ERASED")
  %app_66 = hask.apSSA(%return, %type_65)
  %app_67 = hask.apSSA(%app_66, %$fMonadIO)
  %type_68 = hask.make_string("TYPEINFO_ERASED")
  %app_69 = hask.apSSA(%app_67, %type_68)
  %app_70 = hask.apSSA(%app_69, @"()")
  hask.return(%app_70)
  }
}
// ============ Haskell Core ========================
//-- RHS size: {terms: 12, types: 6, coercions: 0, joins: 0/0}
//main:Fib.plus
//  :: main:Fib.SimpleInt -> main:Fib.SimpleInt -> main:Fib.SimpleInt
//[LclIdX]
//main:Fib.plus
//  = \ (i_a12Q :: main:Fib.SimpleInt)
//      (j_a12R :: main:Fib.SimpleInt) ->
//      case i_a12Q of { main:Fib.MkSimpleInt ival_a12S ->
//      case j_a12R of { main:Fib.MkSimpleInt jval_a12T ->
//      main:Fib.MkSimpleInt
//        (ghc-prim-0.5.3:GHC.Prim.+# ival_a12S jval_a12T)
//      }
//      }
//
//-- RHS size: {terms: 12, types: 6, coercions: 0, joins: 0/0}
//main:Fib.minus
//  :: main:Fib.SimpleInt -> main:Fib.SimpleInt -> main:Fib.SimpleInt
//[LclIdX]
//main:Fib.minus
//  = \ (i_a12U :: main:Fib.SimpleInt)
//      (j_a12V :: main:Fib.SimpleInt) ->
//      case i_a12U of { main:Fib.MkSimpleInt ival_a12W ->
//      case j_a12V of { main:Fib.MkSimpleInt jval_a12X ->
//      main:Fib.MkSimpleInt
//        (ghc-prim-0.5.3:GHC.Prim.-# ival_a12W jval_a12X)
//      }
//      }
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//main:Fib.one :: main:Fib.SimpleInt
//[LclIdX]
//main:Fib.one = main:Fib.MkSimpleInt 1#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//main:Fib.zero :: main:Fib.SimpleInt
//[LclIdX]
//main:Fib.zero = main:Fib.MkSimpleInt 0#
//
//Rec {
//-- RHS size: {terms: 18, types: 4, coercions: 0, joins: 0/0}
//main:Fib.fib [Occ=LoopBreaker]
//  :: main:Fib.SimpleInt -> main:Fib.SimpleInt
//[LclIdX]
//main:Fib.fib
//  = \ (i_a12Y :: main:Fib.SimpleInt) ->
//      case i_a12Y of { main:Fib.MkSimpleInt ds_d1ky ->
//      case ds_d1ky of {
//        __DEFAULT ->
//          main:Fib.plus
//            (main:Fib.fib i_a12Y)
//            (main:Fib.fib (main:Fib.minus i_a12Y main:Fib.one));
//        0# -> main:Fib.zero;
//        1# -> main:Fib.one
//      }
//      }
//end Rec }
//
//-- RHS size: {terms: 5, types: 0, coercions: 0, joins: 0/0}
//main:Fib.$trModule :: ghc-prim-0.5.3:GHC.Types.Module
//[LclIdX]
//main:Fib.$trModule
//  = ghc-prim-0.5.3:GHC.Types.Module
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "main"#)
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "Fib"#)
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_a1k5 [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1k5
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      ghc-prim-0.5.3:GHC.Types.$tcInt#
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:Fib.$tcSimpleInt :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:Fib.$tcSimpleInt
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      7748924056139856202##
//      3495965821938844824##
//      main:Fib.$trModule
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "SimpleInt"#)
//      0#
//      ghc-prim-0.5.3:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_a1k6 [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1k6
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      main:Fib.$tcSimpleInt
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_a1k4 [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1k4
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_a1k5 $krep_a1k6
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:Fib.$tc'MkSimpleInt :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:Fib.$tc'MkSimpleInt
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      4702169295613205611##
//      5219586552818290360##
//      main:Fib.$trModule
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "'MkSimpleInt"#)
//      0#
//      $krep_a1k4
//
//-- RHS size: {terms: 3, types: 2, coercions: 0, joins: 0/0}
//main:Fib.main :: ghc-prim-0.5.3:GHC.Types.IO ()
//[LclIdX]
//main:Fib.main
//  = base-4.12.0.0:GHC.Base.return
//      @ ghc-prim-0.5.3:GHC.Types.IO
//      base-4.12.0.0:GHC.Base.$fMonadIO
//      @ ()
//      ghc-prim-0.5.3:GHC.Tuple.()
//