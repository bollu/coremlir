// Fib
// Core2MLIR: GenMLIR BeforeCorePrep
module {
    hask.make_data_constructor @"+#"
    hask.make_data_constructor @"-#"
    hask.make_data_constructor @"()"
  hask.func @fib {
  %lambda_0 = hask.lambdaSSA(%i_a12Y) {
    %case_1 = hask.caseSSA  %i_a12Y
    [@"MkSimpleInt" ->
    {
    ^entry(%wild_00: !hask.untyped, %ds_d1ky: !hask.untyped):
      %case_2 = hask.caseSSA  %ds_d1ky
      ["default" ->
      {
      ^entry(%ds_X1kH: !hask.untyped):
        %app_3 = hask.apSSA(@fib, %i_a12Y)
        %app_4 = hask.apSSA(@plus, %app_3)
        %app_5 = hask.apSSA(@minus, %i_a12Y)
        %app_6 = hask.apSSA(%app_5, @one)
        %app_7 = hask.apSSA(@fib, %app_6)
        %app_8 = hask.apSSA(%app_4, %app_7)
      hask.return(%app_8)
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
    hask.return(%case_2)
    }
    ]
    hask.return(%case_1)
  }
  hask.return(%lambda_0)
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