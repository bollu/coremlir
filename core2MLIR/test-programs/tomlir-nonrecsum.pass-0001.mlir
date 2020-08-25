// NonrecSum
// Core2MLIR: GenMLIR AfterCorePrep
module {
//unique:rwp
//name: SimpleSum
//datacons: [SimpleLeft, SimpleRight]
//ctype: Nothing
//arity: 0
//binders: []
  hask.func @sat_s1wj {
  %lambda_0 = hask.lambdaSSA(%x_s1wf) {
    %case_1 = hask.caseSSA  %x_s1wf
    [@"SimpleLeft" ->
    {
    ^entry(%wild_s1wg: !hask.untyped, %i_s1wh: !hask.untyped):
      %app_2 = hask.apSSA(@SimpleRight, %i_s1wh)
    hask.return(%app_2)
    }
    ]
    [@"SimpleRight" ->
    {
    ^entry(%wild_s1wg: !hask.untyped, %i_s1wi: !hask.untyped):
      %app_3 = hask.apSSA(@SimpleLeft, %i_s1wi)
    hask.return(%app_3)
    }
    ]
    hask.return(%case_1)
  }
  hask.return(%lambda_0)
  }
  hask.func @f {
  hask.return(@sat_s1wj)
  }
  hask.func @sslone {
  %lit_4 = hask.make_i64(1)
  %app_5 = hask.apSSA(@SimpleLeft, %lit_4)
  hask.return(%app_5)
  }
  hask.func @sat_s1wn {
  %lit_6 = hask.make_string("NonrecSum")
  %app_7 = hask.apSSA(%TrNameS, %lit_6)
  hask.return(%app_7)
  }
  hask.func @sat_s1wm {
  %lit_8 = hask.make_string("main")
  %app_9 = hask.apSSA(%TrNameS, %lit_8)
  hask.return(%app_9)
  }
  hask.func @$trModule {
  %app_10 = hask.apSSA(%Module, @sat_s1wm)
  %app_11 = hask.apSSA(%app_10, @sat_s1wn)
  hask.return(%app_11)
  }
  hask.func @$krep_s1wo {
  %app_12 = hask.apSSA(%KindRepTyConApp, %$tcInt#)
  %type_13 = hask.make_string("TYPEINFO_ERASED")
  %app_14 = hask.apSSA(%[], %type_13)
  %app_15 = hask.apSSA(%app_12, %app_14)
  hask.return(%app_15)
  }
  hask.func @sat_s1wq {
  %lit_16 = hask.make_string("SimpleSum")
  %app_17 = hask.apSSA(%TrNameS, %lit_16)
  hask.return(%app_17)
  }
  hask.func @$tcSimpleSum {
  %lit_18 = 10539666157017918392
  %app_19 = hask.apSSA(%TyCon, %lit_18)
  %lit_20 = 238426795703873600
  %app_21 = hask.apSSA(%app_19, %lit_20)
  %app_22 = hask.apSSA(%app_21, @$trModule)
  %app_23 = hask.apSSA(%app_22, @sat_s1wq)
  %lit_24 = hask.make_i64(0)
  %app_25 = hask.apSSA(%app_23, %lit_24)
  %app_26 = hask.apSSA(%app_25, %krep$*)
  hask.return(%app_26)
  }
  hask.func @$krep_s1wr {
  %app_27 = hask.apSSA(%KindRepTyConApp, @$tcSimpleSum)
  %type_28 = hask.make_string("TYPEINFO_ERASED")
  %app_29 = hask.apSSA(%[], %type_28)
  %app_30 = hask.apSSA(%app_27, %app_29)
  hask.return(%app_30)
  }
  hask.func @$krep_s1ws {
  %app_31 = hask.apSSA(%KindRepFun, @$krep_s1wo)
  %app_32 = hask.apSSA(%app_31, @$krep_s1wr)
  hask.return(%app_32)
  }
  hask.func @sat_s1wu {
  %lit_33 = hask.make_string("'SimpleLeft")
  %app_34 = hask.apSSA(%TrNameS, %lit_33)
  hask.return(%app_34)
  }
  hask.func @$tc'SimpleLeft {
  %lit_35 = 13771812374155265349
  %app_36 = hask.apSSA(%TyCon, %lit_35)
  %lit_37 = 8649954118969440788
  %app_38 = hask.apSSA(%app_36, %lit_37)
  %app_39 = hask.apSSA(%app_38, @$trModule)
  %app_40 = hask.apSSA(%app_39, @sat_s1wu)
  %lit_41 = hask.make_i64(0)
  %app_42 = hask.apSSA(%app_40, %lit_41)
  %app_43 = hask.apSSA(%app_42, @$krep_s1ws)
  hask.return(%app_43)
  }
  hask.func @sat_s1ww {
  %lit_44 = hask.make_string("'SimpleRight")
  %app_45 = hask.apSSA(%TrNameS, %lit_44)
  hask.return(%app_45)
  }
  hask.func @$tc'SimpleRight {
  %lit_46 = 12955657238496603268
  %app_47 = hask.apSSA(%TyCon, %lit_46)
  %lit_48 = 1764516989908937679
  %app_49 = hask.apSSA(%app_47, %lit_48)
  %app_50 = hask.apSSA(%app_49, @$trModule)
  %app_51 = hask.apSSA(%app_50, @sat_s1ww)
  %lit_52 = hask.make_i64(0)
  %app_53 = hask.apSSA(%app_51, %lit_52)
  %app_54 = hask.apSSA(%app_53, @$krep_s1ws)
  hask.return(%app_54)
  }
  hask.func @main {
  %type_55 = hask.make_string("TYPEINFO_ERASED")
  %app_56 = hask.apSSA(%return, %type_55)
  %app_57 = hask.apSSA(%app_56, %$fMonadIO)
  %type_58 = hask.make_string("TYPEINFO_ERASED")
  %app_59 = hask.apSSA(%app_57, %type_58)
  %app_60 = hask.apSSA(%app_59, @"()")
  hask.return(%app_60)
  }
  hask.func @SimpleLeft {
  %lambda_61 = hask.lambdaSSA(%eta_B1) {
    %app_62 = hask.apSSA(@SimpleLeft, %eta_B1)
    hask.return(%app_62)
  }
  hask.return(%lambda_61)
  }
  hask.func @SimpleRight {
  %lambda_63 = hask.lambdaSSA(%eta_B1) {
    %app_64 = hask.apSSA(@SimpleRight, %eta_B1)
    hask.return(%app_64)
  }
  hask.return(%lambda_63)
  }
}
// ============ Haskell Core ========================
//-- RHS size: {terms: 9, types: 4, coercions: 0, joins: 0/0}
//sat_s1wj :: main:NonrecSum.SimpleSum -> main:NonrecSum.SimpleSum
//[LclId]
//sat_s1wj
//  = \ (x_s1wf [Occ=Once!] :: main:NonrecSum.SimpleSum) ->
//      case x_s1wf of {
//        main:NonrecSum.SimpleLeft i_s1wh [Occ=Once] ->
//          main:NonrecSum.SimpleRight i_s1wh;
//        main:NonrecSum.SimpleRight i_s1wi [Occ=Once] ->
//          main:NonrecSum.SimpleLeft i_s1wi
//      }
//
//-- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.f
//  :: main:NonrecSum.SimpleSum -> main:NonrecSum.SimpleSum
//[LclIdX]
//main:NonrecSum.f = sat_s1wj
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.sslone :: main:NonrecSum.SimpleSum
//[LclIdX]
//main:NonrecSum.sslone = main:NonrecSum.SimpleLeft 1#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1wn :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1wn = ghc-prim-0.5.3:GHC.Types.TrNameS "NonrecSum"#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1wm :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1wm = ghc-prim-0.5.3:GHC.Types.TrNameS "main"#
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$trModule :: ghc-prim-0.5.3:GHC.Types.Module
//[LclIdX]
//main:NonrecSum.$trModule
//  = ghc-prim-0.5.3:GHC.Types.Module sat_s1wm sat_s1wn
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_s1wo [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1wo
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      ghc-prim-0.5.3:GHC.Types.$tcInt#
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1wq :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1wq = ghc-prim-0.5.3:GHC.Types.TrNameS "SimpleSum"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcSimpleSum :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tcSimpleSum
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      10539666157017918392##
//      238426795703873600##
//      main:NonrecSum.$trModule
//      sat_s1wq
//      0#
//      ghc-prim-0.5.3:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_s1wr [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1wr
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcSimpleSum
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_s1ws [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1ws
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_s1wo $krep_s1wr
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1wu :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1wu = ghc-prim-0.5.3:GHC.Types.TrNameS "'SimpleLeft"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'SimpleLeft :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tc'SimpleLeft
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      13771812374155265349##
//      8649954118969440788##
//      main:NonrecSum.$trModule
//      sat_s1wu
//      0#
//      $krep_s1ws
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1ww :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1ww = ghc-prim-0.5.3:GHC.Types.TrNameS "'SimpleRight"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'SimpleRight :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tc'SimpleRight
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      12955657238496603268##
//      1764516989908937679##
//      main:NonrecSum.$trModule
//      sat_s1ww
//      0#
//      $krep_s1ws
//
//-- RHS size: {terms: 3, types: 2, coercions: 0, joins: 0/0}
//main:NonrecSum.main :: ghc-prim-0.5.3:GHC.Types.IO ()
//[LclIdX]
//main:NonrecSum.main
//  = base-4.12.0.0:GHC.Base.return
//      @ ghc-prim-0.5.3:GHC.Types.IO
//      base-4.12.0.0:GHC.Base.$fMonadIO
//      @ ()
//      ghc-prim-0.5.3:GHC.Tuple.()
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//main:NonrecSum.SimpleLeft
//  :: ghc-prim-0.5.3:GHC.Prim.Int# -> main:NonrecSum.SimpleSum
//[GblId[DataCon],
// Arity=1,
// Caf=NoCafRefs,
// Str=<L,U>m1,
// Unf=OtherCon []]
//main:NonrecSum.SimpleLeft
//  = \ (eta_B1 [Occ=Once] :: ghc-prim-0.5.3:GHC.Prim.Int#) ->
//      main:NonrecSum.SimpleLeft eta_B1
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//main:NonrecSum.SimpleRight
//  :: ghc-prim-0.5.3:GHC.Prim.Int# -> main:NonrecSum.SimpleSum
//[GblId[DataCon],
// Arity=1,
// Caf=NoCafRefs,
// Str=<L,U>m2,
// Unf=OtherCon []]
//main:NonrecSum.SimpleRight
//  = \ (eta_B1 [Occ=Once] :: ghc-prim-0.5.3:GHC.Prim.Int#) ->
//      main:NonrecSum.SimpleRight eta_B1
//