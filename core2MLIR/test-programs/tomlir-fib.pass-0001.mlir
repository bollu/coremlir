// Fib
// Core2MLIR: GenMLIR AfterCorePrep
module {
    hask.make_data_constructor @"+#"
    hask.make_data_constructor @"-#"
    hask.make_data_constructor @"()"
  hask.func @sat_s1xB {
  %lambda_0 = hask.lambdaSSA(%i_s1xu) {
    %case_1 = hask.caseSSA  %i_s1xu
    [@"MkSimpleInt" ->
    {
    ^entry(%wild_s1xv: !hask.untyped, %ds_s1xw: !hask.untyped):
      %case_2 = hask.caseSSA  %ds_s1xw
      ["default" ->
      {
      ^entry(%ds_s1xx: !hask.untyped):
        %unimpl_3  =  hask.make_i32(42)
      hask.return(%unimpl_3)
      }
      ]
      [0 ->
      {
      ^entry(%ds_s1xx: !hask.untyped):
      hask.return(@zero)
      }
      ]
      [1 ->
      {
      ^entry(%ds_s1xx: !hask.untyped):
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
  hask.func @fib {
  hask.return(@sat_s1xB)
  }
}
// ============ Haskell Core ========================
//-- RHS size: {terms: 15, types: 7, coercions: 0, joins: 0/0}
//sat_s1xh
//  :: main:Fib.SimpleInt -> main:Fib.SimpleInt -> main:Fib.SimpleInt
//[LclId]
//sat_s1xh
//  = \ (i_s1xa [Occ=Once!] :: main:Fib.SimpleInt)
//      (j_s1xb [Occ=Once!] :: main:Fib.SimpleInt) ->
//      case i_s1xa of { main:Fib.MkSimpleInt ival_s1xd [Occ=Once] ->
//      case j_s1xb of { main:Fib.MkSimpleInt jval_s1xf [Occ=Once] ->
//      case ghc-prim-0.5.3:GHC.Prim.+# ival_s1xd jval_s1xf
//      of sat_s1xg [Occ=Once]
//      { __DEFAULT ->
//      main:Fib.MkSimpleInt sat_s1xg
//      }
//      }
//      }
//
//-- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
//main:Fib.plus
//  :: main:Fib.SimpleInt -> main:Fib.SimpleInt -> main:Fib.SimpleInt
//[LclIdX]
//main:Fib.plus = sat_s1xh
//
//-- RHS size: {terms: 15, types: 7, coercions: 0, joins: 0/0}
//sat_s1xq
//  :: main:Fib.SimpleInt -> main:Fib.SimpleInt -> main:Fib.SimpleInt
//[LclId]
//sat_s1xq
//  = \ (i_s1xj [Occ=Once!] :: main:Fib.SimpleInt)
//      (j_s1xk [Occ=Once!] :: main:Fib.SimpleInt) ->
//      case i_s1xj of { main:Fib.MkSimpleInt ival_s1xm [Occ=Once] ->
//      case j_s1xk of { main:Fib.MkSimpleInt jval_s1xo [Occ=Once] ->
//      case ghc-prim-0.5.3:GHC.Prim.-# ival_s1xm jval_s1xo
//      of sat_s1xp [Occ=Once]
//      { __DEFAULT ->
//      main:Fib.MkSimpleInt sat_s1xp
//      }
//      }
//      }
//
//-- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
//main:Fib.minus
//  :: main:Fib.SimpleInt -> main:Fib.SimpleInt -> main:Fib.SimpleInt
//[LclIdX]
//main:Fib.minus = sat_s1xq
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
//-- RHS size: {terms: 24, types: 7, coercions: 0, joins: 0/3}
//sat_s1xB :: main:Fib.SimpleInt -> main:Fib.SimpleInt
//[LclId]
//sat_s1xB
//  = \ (i_s1xu :: main:Fib.SimpleInt) ->
//      case i_s1xu of { main:Fib.MkSimpleInt ds_s1xw [Occ=Once!] ->
//      case ds_s1xw of {
//        __DEFAULT ->
//          let {
//            sat_s1xA [Occ=Once] :: main:Fib.SimpleInt
//            [LclId]
//            sat_s1xA
//              = let {
//                  sat_s1xz [Occ=Once] :: main:Fib.SimpleInt
//                  [LclId]
//                  sat_s1xz = main:Fib.minus i_s1xu main:Fib.one } in
//                main:Fib.fib sat_s1xz } in
//          let {
//            sat_s1xy [Occ=Once] :: main:Fib.SimpleInt
//            [LclId]
//            sat_s1xy = main:Fib.fib i_s1xu } in
//          main:Fib.plus sat_s1xy sat_s1xA;
//        0# -> main:Fib.zero;
//        1# -> main:Fib.one
//      }
//      }
//
//-- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
//main:Fib.fib [Occ=LoopBreaker]
//  :: main:Fib.SimpleInt -> main:Fib.SimpleInt
//[LclIdX]
//main:Fib.fib = sat_s1xB
//end Rec }
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1xE :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1xE = ghc-prim-0.5.3:GHC.Types.TrNameS "Fib"#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1xD :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1xD = ghc-prim-0.5.3:GHC.Types.TrNameS "main"#
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//main:Fib.$trModule :: ghc-prim-0.5.3:GHC.Types.Module
//[LclIdX]
//main:Fib.$trModule
//  = ghc-prim-0.5.3:GHC.Types.Module sat_s1xD sat_s1xE
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_s1xF [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1xF
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      ghc-prim-0.5.3:GHC.Types.$tcInt#
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1xH :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1xH = ghc-prim-0.5.3:GHC.Types.TrNameS "SimpleInt"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:Fib.$tcSimpleInt :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:Fib.$tcSimpleInt
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      7748924056139856202##
//      3495965821938844824##
//      main:Fib.$trModule
//      sat_s1xH
//      0#
//      ghc-prim-0.5.3:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_s1xI [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1xI
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      main:Fib.$tcSimpleInt
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_s1xJ [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1xJ
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_s1xF $krep_s1xI
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1xL :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1xL = ghc-prim-0.5.3:GHC.Types.TrNameS "'MkSimpleInt"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:Fib.$tc'MkSimpleInt :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:Fib.$tc'MkSimpleInt
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      4702169295613205611##
//      5219586552818290360##
//      main:Fib.$trModule
//      sat_s1xL
//      0#
//      $krep_s1xJ
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
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//main:Fib.MkSimpleInt
//  :: ghc-prim-0.5.3:GHC.Prim.Int# -> main:Fib.SimpleInt
//[GblId[DataCon],
// Arity=1,
// Caf=NoCafRefs,
// Str=<L,U>m,
// Unf=OtherCon []]
//main:Fib.MkSimpleInt
//  = \ (eta_B1 [Occ=Once] :: ghc-prim-0.5.3:GHC.Prim.Int#) ->
//      main:Fib.MkSimpleInt eta_B1
//