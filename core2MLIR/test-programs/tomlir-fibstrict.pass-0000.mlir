// Main
// Core2MLIR: GenMLIR BeforeCorePrep
hask.module {
  hask.recursive_ref {
    %fib = hask.toplevel_binding {
             %lambda_10 =
               hask.lambdaSSA(%i_a12E) {
                 %case_0 =
                   hask.caseSSA %i_a12E
                   ["default" ->
                     { ^entry(%ds_d1jZ: none):
                         %app_0  =  hask.apSSA(%minus_hash, %i_a12E)
                         %lit_1  =  hask.make_i32(1)
                         %app_2  =  hask.apSSA(%app_0, %lit_1)
                         %app_3  =  hask.apSSA(%fib, %app_2)
                         %case_4 =
                           hask.caseSSA %app_3
                           ["default" ->
                             { ^entry(%wild_00: none):
                                 %app_4  =  hask.apSSA(%fib, %i_a12E)
                                 %case_5 =
                                   hask.caseSSA %app_4
                                   ["default" ->
                                     { ^entry(%wild_X5: none):
                                         %app_5  =  hask.apSSA(%plus_hash, %wild_X5)
                                         hask.return(%app_5)
                                     }]
                                 %app_7  =  hask.apSSA(%case_5, %wild_00)
                                 hask.return(%app_7)
                             }]
                         hask.return(%case_4)
                     }]
                   [0 ->
                     { ^entry(%ds_d1jZ: none):
                         hask.return(%i_a12E)
                     }]
                   [1 ->
                     { ^entry(%ds_d1jZ: none):
                         hask.return(%i_a12E)
                     }]
                 hask.return(%case_0)
               }
             hask.return(%lambda_10)
           }
  }
  %$trModule =
    hask.toplevel_binding {
      %lit_0  =  hask.make_string("main")
      %app_1  =  hask.apSSA(%TrNameS, %lit_0)
      %app_2  =  hask.apSSA(%Module, %app_1)
      %lit_3  =  hask.make_string("Main")
      %app_4  =  hask.apSSA(%TrNameS, %lit_3)
      %app_5  =  hask.apSSA(%app_2, %app_4)
      hask.return(%app_5)
    }
  hask.dummy_finish
}
// ============ Haskell Core ========================
//Rec {
//-- RHS size: {terms: 21, types: 4, coercions: 0, joins: 0/0}
//(main:Main.fib{v rwj} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                               -> ghc-prim:GHC.Prim.Int#{(w) tc 3s}) [Occ=LoopBreaker]
//  :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//     -> ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//[LclId]
//(main:Main.fib{v rwj} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                               -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//  = \ ((i{v a12E} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//         :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}) ->
//      case (i{v a12E} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}) return ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//      of ((ds_d1jZ{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}) [Occ=Dead]
//            :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}) {
//        __DEFAULT ->
//          case (main:Main.fib{v rwj} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                              -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//                 ((ghc-prim:GHC.Prim.-#{(w) v 99} [gid[PrimOp]] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                                                   -> ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                                                      -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//                    (i{v a12E} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}) 1#)
//          return ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//          of ((wild_00{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//                :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//          { __DEFAULT ->
//          (case (main:Main.fib{v rwj} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                               -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//                  (i{v a12E} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//           return ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                  -> ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//           of ((wild_X5{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//                 :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//           { __DEFAULT ->
//           (ghc-prim:GHC.Prim.+#{(w) v 98} [gid[PrimOp]] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                                            -> ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                                               -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//             (wild_X5{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//           })
//            (wild_00{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//          };
//        0# -> (i{v a12E} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s});
//        1# -> (i{v a12E} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//      }
//end Rec }
//
//-- RHS size: {terms: 5, types: 0, coercions: 0, joins: 0/0}
//(main:Main.$trModule{v r1iw} [lidx] :: ghc-prim:GHC.Types.Module{tc 622})
//  :: ghc-prim:GHC.Types.Module{tc 622}
//[LclIdX]
//(main:Main.$trModule{v r1iw} [lidx] :: ghc-prim:GHC.Types.Module{tc 622})
//  = (ghc-prim:GHC.Types.Module{v r7} [gid[DataCon]] :: ghc-prim:GHC.Types.TrName{tc 628}
//                                                       -> ghc-prim:GHC.Types.TrName{tc 628}
//                                                          -> ghc-prim:GHC.Types.Module{tc 622})
//      ((ghc-prim:GHC.Types.TrNameS{v ra} [gid[DataCon]] :: ghc-prim:GHC.Prim.Addr#{(w) tc 32}
//                                                           -> ghc-prim:GHC.Types.TrName{tc 628})
//         "main"#)
//      ((ghc-prim:GHC.Types.TrNameS{v ra} [gid[DataCon]] :: ghc-prim:GHC.Prim.Addr#{(w) tc 32}
//                                                           -> ghc-prim:GHC.Types.TrName{tc 628})
//         "Main"#)
//
//-- RHS size: {terms: 7, types: 3, coercions: 0, joins: 0/0}
//(main:Main.main{v ryV} [lidx] :: ghc-prim:GHC.Types.IO{tc 31Q}
//                                   ghc-prim:GHC.Tuple.(){(w) tc 40})
//  :: ghc-prim:GHC.Types.IO{tc 31Q} ghc-prim:GHC.Tuple.(){(w) tc 40}
//[LclIdX]
//(main:Main.main{v ryV} [lidx] :: ghc-prim:GHC.Types.IO{tc 31Q}
//                                   ghc-prim:GHC.Tuple.(){(w) tc 40})
//  = case (main:Main.fib{v rwj} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                        -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//           10#
//    return ghc-prim:GHC.Types.IO{tc 31Q}
//             ghc-prim:GHC.Tuple.(){(w) tc 40}
//    of ((x{v a1hu} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}) [Occ=Dead]
//          :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//    { __DEFAULT ->
//    (base:GHC.Base.return{v 02O} [gid[ClassOp]] :: forall @(m{tv a1hx} [tv] :: ghc-prim:GHC.Prim.TYPE{(w) tc 32Q}
//                                                                                 'ghc-prim:GHC.Types.LiftedRep{(w) d 63A}
//                                                                               -> ghc-prim:GHC.Prim.TYPE{(w) tc 32Q}
//                                                                                    'ghc-prim:GHC.Types.LiftedRep{(w) d 63A}).
//                                                     base:GHC.Base.Monad{tc 28}
//                                                       (m{tv a1hx} [tv] :: ghc-prim:GHC.Prim.TYPE{(w) tc 32Q}
//                                                                             'ghc-prim:GHC.Types.LiftedRep{(w) d 63A}
//                                                                           -> ghc-prim:GHC.Prim.TYPE{(w) tc 32Q}
//                                                                                'ghc-prim:GHC.Types.LiftedRep{(w) d 63A})
//                                                     -> forall @(a{tv a1hG} [tv] :: ghc-prim:GHC.Prim.TYPE{(w) tc 32Q}
//                                                                                      'ghc-prim:GHC.Types.LiftedRep{(w) d 63A}).
//                                                          (a{tv a1hG} [tv] :: ghc-prim:GHC.Prim.TYPE{(w) tc 32Q}
//                                                                                'ghc-prim:GHC.Types.LiftedRep{(w) d 63A})
//                                                          -> (m{tv a1hx} [tv] :: ghc-prim:GHC.Prim.TYPE{(w) tc 32Q}
//                                                                                   'ghc-prim:GHC.Types.LiftedRep{(w) d 63A}
//                                                                                 -> ghc-prim:GHC.Prim.TYPE{(w) tc 32Q}
//                                                                                      'ghc-prim:GHC.Types.LiftedRep{(w) d 63A})
//                                                               (a{tv a1hG} [tv] :: ghc-prim:GHC.Prim.TYPE{(w) tc 32Q}
//                                                                                     'ghc-prim:GHC.Types.LiftedRep{(w) d 63A}))
//      @ ghc-prim:GHC.Types.IO{tc 31Q}
//      (base:GHC.Base.$fMonadIO{v rob} [gid[DFunId]] :: base:GHC.Base.Monad{tc 28}
//                                                         ghc-prim:GHC.Types.IO{tc 31Q})
//      @ ghc-prim:GHC.Tuple.(){(w) tc 40}
//      (ghc-prim:GHC.Tuple.(){(w) v 71} [gid[DataCon]] :: ghc-prim:GHC.Tuple.(){(w) tc 40})
//    }
//
//-- RHS size: {terms: 2, types: 1, coercions: 0, joins: 0/0}
//(main::Main.main{v 01D} [lidx] :: ghc-prim:GHC.Types.IO{tc 31Q}
//                                    ghc-prim:GHC.Tuple.(){(w) tc 40})
//  :: ghc-prim:GHC.Types.IO{tc 31Q} ghc-prim:GHC.Tuple.(){(w) tc 40}
//[LclIdX]
//(main::Main.main{v 01D} [lidx] :: ghc-prim:GHC.Types.IO{tc 31Q}
//                                    ghc-prim:GHC.Tuple.(){(w) tc 40})
//  = (base:GHC.TopHandler.runMainIO{v 01E} [gid] :: forall @(a{tv a1iY} [tv] :: ghc-prim:GHC.Prim.TYPE{(w) tc 32Q}
//                                                                                 'ghc-prim:GHC.Types.LiftedRep{(w) d 63A}).
//                                                     ghc-prim:GHC.Types.IO{tc 31Q}
//                                                       (a{tv a1iY} [tv] :: ghc-prim:GHC.Prim.TYPE{(w) tc 32Q}
//                                                                             'ghc-prim:GHC.Types.LiftedRep{(w) d 63A})
//                                                     -> ghc-prim:GHC.Types.IO{tc 31Q}
//                                                          (a{tv a1iY} [tv] :: ghc-prim:GHC.Prim.TYPE{(w) tc 32Q}
//                                                                                'ghc-prim:GHC.Types.LiftedRep{(w) d 63A}))
//      @ ghc-prim:GHC.Tuple.(){(w) tc 40}
//      (main:Main.main{v ryV} [lidx] :: ghc-prim:GHC.Types.IO{tc 31Q}
//                                         ghc-prim:GHC.Tuple.(){(w) tc 40})
//