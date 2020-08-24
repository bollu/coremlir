// Fib
// Core2MLIR: GenMLIR AfterCorePrep
module {
    hask.make_data_constructor @"+#"
    hask.make_data_constructor @"-#"
    hask.make_data_constructor @"()"
  hask.func @sat_s1xh {
  %lambda_0 = hask.lambdaSSA(%i_s1xa) {
    %lambda_1 = hask.lambdaSSA(%j_s1xb) {
      %case_2 = hask.caseSSA  %i_s1xa
      [@"MkSimpleInt" ->
      {
      ^entry(%wild_s1xc: !hask.untyped, %ival_s1xd: !hask.untyped):
        %case_3 = hask.caseSSA  %j_s1xb
        [@"MkSimpleInt" ->
        {
        ^entry(%wild_s1xe: !hask.untyped, %jval_s1xf: !hask.untyped):
          %app_4 = hask.apSSA(@"+#", %ival_s1xd)
          %app_5 = hask.apSSA(%app_4, %jval_s1xf)
          %sat_s1xg = hask.force (%app_5)
          %app_7 = hask.apSSA(@MkSimpleInt, %sat_s1xg)
        hask.return(%app_7)
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
  hask.func @plus {
  hask.return(@sat_s1xh)
  }
  hask.func @sat_s1xq {
  %lambda_8 = hask.lambdaSSA(%i_s1xj) {
    %lambda_9 = hask.lambdaSSA(%j_s1xk) {
      %case_10 = hask.caseSSA  %i_s1xj
      [@"MkSimpleInt" ->
      {
      ^entry(%wild_s1xl: !hask.untyped, %ival_s1xm: !hask.untyped):
        %case_11 = hask.caseSSA  %j_s1xk
        [@"MkSimpleInt" ->
        {
        ^entry(%wild_s1xn: !hask.untyped, %jval_s1xo: !hask.untyped):
          %app_12 = hask.apSSA(@"-#", %ival_s1xm)
          %app_13 = hask.apSSA(%app_12, %jval_s1xo)
          %sat_s1xp = hask.force (%app_13)
          %app_15 = hask.apSSA(@MkSimpleInt, %sat_s1xp)
        hask.return(%app_15)
        }
        ]
      hask.return(%case_11)
      }
      ]
      hask.return(%case_10)
    }
    hask.return(%lambda_9)
  }
  hask.return(%lambda_8)
  }
  hask.func @minus {
  hask.return(@sat_s1xq)
  }
  hask.func @one {
  %lit_16 = hask.make_i64(1)
  %app_17 = hask.apSSA(@MkSimpleInt, %lit_16)
  hask.return(%app_17)
  }
  hask.func @zero {
  %lit_18 = hask.make_i64(0)
  %app_19 = hask.apSSA(@MkSimpleInt, %lit_18)
  hask.return(%app_19)
  }
  hask.func @sat_s1xB {
  %lambda_20 = hask.lambdaSSA(%i_s1xu) {
    %case_21 = hask.caseSSA  %i_s1xu
    [@"MkSimpleInt" ->
    {
    ^entry(%wild_s1xv: !hask.untyped, %ds_s1xw: !hask.untyped):
      %case_22 = hask.caseSSA  %ds_s1xw
      ["default" ->
      {
      ^entry(%ds_s1xx: !hask.untyped):
        %unimpl_23  =  hask.make_i32(42)
      hask.return(%unimpl_23)
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
    hask.return(%case_22)
    }
    ]
    hask.return(%case_21)
  }
  hask.return(%lambda_20)
  }
  hask.func @fib {
  hask.return(@sat_s1xB)
  }
  hask.func @sat_s1xE {
  %lit_24 = hask.make_string("Fib")
  %app_25 = hask.apSSA(%TrNameS, %lit_24)
  hask.return(%app_25)
  }
  hask.func @sat_s1xD {
  %lit_26 = hask.make_string("main")
  %app_27 = hask.apSSA(%TrNameS, %lit_26)
  hask.return(%app_27)
  }
  hask.func @$trModule {
  %app_28 = hask.apSSA(%Module, @sat_s1xD)
  %app_29 = hask.apSSA(%app_28, @sat_s1xE)
  hask.return(%app_29)
  }
  hask.func @$krep_s1xF {
  %app_30 = hask.apSSA(%KindRepTyConApp, %$tcInt#)
  %type_31 = hask.make_string("TYPEINFO_ERASED")
  %app_32 = hask.apSSA(%[], %type_31)
  %app_33 = hask.apSSA(%app_30, %app_32)
  hask.return(%app_33)
  }
  hask.func @sat_s1xH {
  %lit_34 = hask.make_string("SimpleInt")
  %app_35 = hask.apSSA(%TrNameS, %lit_34)
  hask.return(%app_35)
  }
  hask.func @$tcSimpleInt {
  %lit_36 = 7748924056139856202
  %app_37 = hask.apSSA(%TyCon, %lit_36)
  %lit_38 = 3495965821938844824
  %app_39 = hask.apSSA(%app_37, %lit_38)
  %app_40 = hask.apSSA(%app_39, @$trModule)
  %app_41 = hask.apSSA(%app_40, @sat_s1xH)
  %lit_42 = hask.make_i64(0)
  %app_43 = hask.apSSA(%app_41, %lit_42)
  %app_44 = hask.apSSA(%app_43, %krep$*)
  hask.return(%app_44)
  }
  hask.func @$krep_s1xI {
  %app_45 = hask.apSSA(%KindRepTyConApp, @$tcSimpleInt)
  %type_46 = hask.make_string("TYPEINFO_ERASED")
  %app_47 = hask.apSSA(%[], %type_46)
  %app_48 = hask.apSSA(%app_45, %app_47)
  hask.return(%app_48)
  }
  hask.func @$krep_s1xJ {
  %app_49 = hask.apSSA(%KindRepFun, @$krep_s1xF)
  %app_50 = hask.apSSA(%app_49, @$krep_s1xI)
  hask.return(%app_50)
  }
  hask.func @sat_s1xL {
  %lit_51 = hask.make_string("'MkSimpleInt")
  %app_52 = hask.apSSA(%TrNameS, %lit_51)
  hask.return(%app_52)
  }
  hask.func @$tc'MkSimpleInt {
  %lit_53 = 4702169295613205611
  %app_54 = hask.apSSA(%TyCon, %lit_53)
  %lit_55 = 5219586552818290360
  %app_56 = hask.apSSA(%app_54, %lit_55)
  %app_57 = hask.apSSA(%app_56, @$trModule)
  %app_58 = hask.apSSA(%app_57, @sat_s1xL)
  %lit_59 = hask.make_i64(0)
  %app_60 = hask.apSSA(%app_58, %lit_59)
  %app_61 = hask.apSSA(%app_60, @$krep_s1xJ)
  hask.return(%app_61)
  }
  hask.func @main {
  %type_62 = hask.make_string("TYPEINFO_ERASED")
  %app_63 = hask.apSSA(%return, %type_62)
  %app_64 = hask.apSSA(%app_63, %$fMonadIO)
  %type_65 = hask.make_string("TYPEINFO_ERASED")
  %app_66 = hask.apSSA(%app_64, %type_65)
  %app_67 = hask.apSSA(%app_66, @"()")
  hask.return(%app_67)
  }
  hask.func @MkSimpleInt {
  %lambda_68 = hask.lambdaSSA(%eta_B1) {
    %app_69 = hask.apSSA(@MkSimpleInt, %eta_B1)
    hask.return(%app_69)
  }
  hask.return(%lambda_68)
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