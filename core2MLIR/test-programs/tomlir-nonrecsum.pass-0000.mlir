// NonrecSum
// Core2MLIR: GenMLIR BeforeCorePrep
module {
//unique:rwp
//name: SimpleSum
//datacons: [SimpleLeft, SimpleRight]
//ctype: Nothing
//arity: 0
//binders: []
  hask.func @f {
  %lambda_0 = hask.lambdaSSA(%x_a12O) {
    %case_1 = hask.caseSSA  %x_a12O
    [@"SimpleLeft" ->
    {
    ^entry(%wild_00: !hask.untyped, %i_a12P: !hask.untyped):
      %app_2 = hask.apSSA(%SimpleRight, %i_a12P)
    hask.return(%app_2)
    }
    ]
    [@"SimpleRight" ->
    {
    ^entry(%wild_00: !hask.untyped, %i_a12Q: !hask.untyped):
      %app_3 = hask.apSSA(%SimpleLeft, %i_a12Q)
    hask.return(%app_3)
    }
    ]
    hask.return(%case_1)
  }
  hask.return(%lambda_0)
  }
  hask.func @sslone {
  %lit_4 = hask.make_i64(1)
  %app_5 = hask.apSSA(%SimpleLeft, %lit_4)
  hask.return(%app_5)
  }
  hask.func @$trModule {
  %lit_6 = hask.make_string("main")
  %app_7 = hask.apSSA(%TrNameS, %lit_6)
  %app_8 = hask.apSSA(%Module, %app_7)
  %lit_9 = hask.make_string("NonrecSum")
  %app_10 = hask.apSSA(%TrNameS, %lit_9)
  %app_11 = hask.apSSA(%app_8, %app_10)
  hask.return(%app_11)
  }
  hask.func @$krep_a1ju {
  %app_12 = hask.apSSA(%KindRepTyConApp, %$tcInt#)
  %type_13 = hask.make_string("TYPEINFO_ERASED")
  %app_14 = hask.apSSA(%[], %type_13)
  %app_15 = hask.apSSA(%app_12, %app_14)
  hask.return(%app_15)
  }
  hask.func @$tcSimpleSum {
  %lit_16 = 10539666157017918392
  %app_17 = hask.apSSA(%TyCon, %lit_16)
  %lit_18 = 238426795703873600
  %app_19 = hask.apSSA(%app_17, %lit_18)
  %app_20 = hask.apSSA(%app_19, @$trModule)
  %lit_21 = hask.make_string("SimpleSum")
  %app_22 = hask.apSSA(%TrNameS, %lit_21)
  %app_23 = hask.apSSA(%app_20, %app_22)
  %lit_24 = hask.make_i64(0)
  %app_25 = hask.apSSA(%app_23, %lit_24)
  %app_26 = hask.apSSA(%app_25, %krep$*)
  hask.return(%app_26)
  }
  hask.func @$krep_a1jv {
  %app_27 = hask.apSSA(%KindRepTyConApp, @$tcSimpleSum)
  %type_28 = hask.make_string("TYPEINFO_ERASED")
  %app_29 = hask.apSSA(%[], %type_28)
  %app_30 = hask.apSSA(%app_27, %app_29)
  hask.return(%app_30)
  }
  hask.func @$krep_a1jt {
  %app_31 = hask.apSSA(%KindRepFun, @$krep_a1ju)
  %app_32 = hask.apSSA(%app_31, @$krep_a1jv)
  hask.return(%app_32)
  }
  hask.func @$tc'SimpleLeft {
  %lit_33 = 13771812374155265349
  %app_34 = hask.apSSA(%TyCon, %lit_33)
  %lit_35 = 8649954118969440788
  %app_36 = hask.apSSA(%app_34, %lit_35)
  %app_37 = hask.apSSA(%app_36, @$trModule)
  %lit_38 = hask.make_string("'SimpleLeft")
  %app_39 = hask.apSSA(%TrNameS, %lit_38)
  %app_40 = hask.apSSA(%app_37, %app_39)
  %lit_41 = hask.make_i64(0)
  %app_42 = hask.apSSA(%app_40, %lit_41)
  %app_43 = hask.apSSA(%app_42, @$krep_a1jt)
  hask.return(%app_43)
  }
  hask.func @$tc'SimpleRight {
  %lit_44 = 12955657238496603268
  %app_45 = hask.apSSA(%TyCon, %lit_44)
  %lit_46 = 1764516989908937679
  %app_47 = hask.apSSA(%app_45, %lit_46)
  %app_48 = hask.apSSA(%app_47, @$trModule)
  %lit_49 = hask.make_string("'SimpleRight")
  %app_50 = hask.apSSA(%TrNameS, %lit_49)
  %app_51 = hask.apSSA(%app_48, %app_50)
  %lit_52 = hask.make_i64(0)
  %app_53 = hask.apSSA(%app_51, %lit_52)
  %app_54 = hask.apSSA(%app_53, @$krep_a1jt)
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
}
// ============ Haskell Core ========================
//-- RHS size: {terms: 9, types: 4, coercions: 0, joins: 0/0}
//main:NonrecSum.f
//  :: main:NonrecSum.SimpleSum -> main:NonrecSum.SimpleSum
//[LclIdX]
//main:NonrecSum.f
//  = \ (x_a12O :: main:NonrecSum.SimpleSum) ->
//      case x_a12O of {
//        main:NonrecSum.SimpleLeft i_a12P ->
//          main:NonrecSum.SimpleRight i_a12P;
//        main:NonrecSum.SimpleRight i_a12Q ->
//          main:NonrecSum.SimpleLeft i_a12Q
//      }
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.sslone :: main:NonrecSum.SimpleSum
//[LclIdX]
//main:NonrecSum.sslone = main:NonrecSum.SimpleLeft 1#
//
//-- RHS size: {terms: 5, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$trModule :: ghc-prim-0.5.3:GHC.Types.Module
//[LclIdX]
//main:NonrecSum.$trModule
//  = ghc-prim-0.5.3:GHC.Types.Module
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "main"#)
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "NonrecSum"#)
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_a1ju [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1ju
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      ghc-prim-0.5.3:GHC.Types.$tcInt#
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcSimpleSum :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tcSimpleSum
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      10539666157017918392##
//      238426795703873600##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "SimpleSum"#)
//      0#
//      ghc-prim-0.5.3:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_a1jv [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1jv
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcSimpleSum
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_a1jt [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1jt
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_a1ju $krep_a1jv
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'SimpleLeft :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tc'SimpleLeft
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      13771812374155265349##
//      8649954118969440788##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "'SimpleLeft"#)
//      0#
//      $krep_a1jt
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'SimpleRight :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tc'SimpleRight
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      12955657238496603268##
//      1764516989908937679##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "'SimpleRight"#)
//      0#
//      $krep_a1jt
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