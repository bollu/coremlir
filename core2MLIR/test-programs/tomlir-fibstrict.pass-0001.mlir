// Main
// Core2MLIR: GenMLIR AfterCorePrep
hask.module {
  hask.recursive_ref {
    %sat_s1wO = hask.toplevel_binding {
                  %lambda_12 =
                    hask.lambdaSSA(%i_s1wH) {
                      %case_0 =
                        hask.caseSSA %i_s1wH
                        ["default" ->
                          { ^entry(%ds_s1wI: none):
                              %app_0  =  hask.apSSA(%minus_hash, %i_s1wH)
                              %lit_1  =  hask.make_i32(1)
                              %app_2  =  hask.apSSA(%app_0, %lit_1)
                              %case_3 =
                                hask.caseSSA %app_2
                                ["default" ->
                                  { ^entry(%sat_s1wJ: none):
                                      %app_3  =  hask.apSSA(%fib, %sat_s1wJ)
                                      %case_4 =
                                        hask.caseSSA %app_3
                                        ["default" ->
                                          { ^entry(%wild_s1wK: none):
                                              %app_4  =  hask.apSSA(%fib, %i_s1wH)
                                              %case_5 =
                                                hask.caseSSA %app_4
                                                ["default" ->
                                                  { ^entry(%wild_s1wL: none):
                                                      %unimpl_5  =  hask.make_i32(42)
                                                      hask.return(%unimpl_5)
                                                  }]
                                              %case_7 =
                                                hask.caseSSA %case_5
                                                ["default" ->
                                                  { ^entry(%sat_s1wN: none):
                                                      %app_7  =  hask.apSSA(%sat_s1wN, %wild_s1wK)
                                                      hask.return(%app_7)
                                                  }]
                                              hask.return(%case_7)
                                          }]
                                      hask.return(%case_4)
                                  }]
                              hask.return(%case_3)
                          }]
                        [0 ->
                          { ^entry(%ds_s1wI: none):
                              hask.return(%i_s1wH)
                          }]
                        [1 ->
                          { ^entry(%ds_s1wI: none):
                              hask.return(%i_s1wH)
                          }]
                      hask.return(%case_0)
                    }
                  hask.return(%lambda_12)
                }
    %fib = hask.toplevel_binding {
             hask.return(%sat_s1wO)
           }
  }
  %sat_s1wR =
    hask.toplevel_binding {
      %lit_0  =  hask.make_string("Main")
      %app_1  =  hask.apSSA(%TrNameS, %lit_0)
      hask.return(%app_1)
    }
  %sat_s1wQ =
    hask.toplevel_binding {
      %lit_0  =  hask.make_string("main")
      %app_1  =  hask.apSSA(%TrNameS, %lit_0)
      hask.return(%app_1)
    }
  %$trModule =
    hask.toplevel_binding {
      %app_0  =  hask.apSSA(%Module, %sat_s1wQ)
      %app_1  =  hask.apSSA(%app_0, %sat_s1wR)
      hask.return(%app_1)
    }
  hask.dummy_finish
}
// ============ Haskell Core ========================
//Rec {
//-- RHS size: {terms: 31, types: 10, coercions: 0, joins: 0/1}
//(sat_s1wO{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                      -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//  :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//     -> ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//[LclId]
//(sat_s1wO{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                      -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//  = \ ((i{v s1wH} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//         :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}) ->
//      case (i{v s1wH} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}) return ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//      of ((ds_s1wI{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}) [Occ=Dead]
//            :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}) {
//        __DEFAULT ->
//          case (ghc-prim:GHC.Prim.-#{(w) v 99} [gid[PrimOp]] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                                                -> ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                                                   -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//                 (i{v s1wH} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}) 1#
//          return ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//          of ((sat_s1wJ{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}) [Occ=Once]
//                :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//          { __DEFAULT ->
//          case (main:Main.fib{v s1wG} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                               -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//                 (sat_s1wJ{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//          return ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//          of ((wild_s1wK{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}) [Occ=Once]
//                :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//          { __DEFAULT ->
//          case case (main:Main.fib{v s1wG} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                                    -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//                      (i{v s1wH} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//               return ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                      -> ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//               of ((wild_s1wL{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}) [Occ=OnceL]
//                     :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//               { __DEFAULT ->
//               let {
//                 (sat_s1wM{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                       -> ghc-prim:GHC.Prim.Int#{(w) tc 3s}) [Occ=OnceT[0]]
//                   :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                      -> ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                 [LclId]
//                 (sat_s1wM{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                       -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//                   = \ ((eta_B1{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}) [Occ=Once]
//                          :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}) ->
//                       (ghc-prim:GHC.Prim.+#{(w) v 98} [gid[PrimOp]] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                                                        -> ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                                                           -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//                         (wild_s1wL{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//                         (eta_B1{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}) } in
//               (sat_s1wM{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                     -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//               }
//          return ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//          of ((sat_s1wN{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                    -> ghc-prim:GHC.Prim.Int#{(w) tc 3s}) [Occ=Once!]
//                :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                   -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//          { __DEFAULT ->
//          (sat_s1wN{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//            (wild_s1wK{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//          }
//          }
//          };
//        0# -> (i{v s1wH} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s});
//        1# -> (i{v s1wH} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//      }
//
//-- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
//(main:Main.fib{v s1wG} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                -> ghc-prim:GHC.Prim.Int#{(w) tc 3s}) [Occ=LoopBreaker]
//  :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//     -> ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//[LclId]
//(main:Main.fib{v s1wG} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//  = (sat_s1wO{v} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                          -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//end Rec }
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//(sat_s1wR{v} [lid] :: ghc-prim:GHC.Types.TrName{tc 628})
//  :: ghc-prim:GHC.Types.TrName{tc 628}
//[LclId]
//(sat_s1wR{v} [lid] :: ghc-prim:GHC.Types.TrName{tc 628})
//  = (ghc-prim:GHC.Types.TrNameS{v ra} [gid[DataCon]] :: ghc-prim:GHC.Prim.Addr#{(w) tc 32}
//                                                        -> ghc-prim:GHC.Types.TrName{tc 628})
//      "Main"#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//(sat_s1wQ{v} [lid] :: ghc-prim:GHC.Types.TrName{tc 628})
//  :: ghc-prim:GHC.Types.TrName{tc 628}
//[LclId]
//(sat_s1wQ{v} [lid] :: ghc-prim:GHC.Types.TrName{tc 628})
//  = (ghc-prim:GHC.Types.TrNameS{v ra} [gid[DataCon]] :: ghc-prim:GHC.Prim.Addr#{(w) tc 32}
//                                                        -> ghc-prim:GHC.Types.TrName{tc 628})
//      "main"#
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//(main:Main.$trModule{v s1wP} [lidx] :: ghc-prim:GHC.Types.Module{tc 622})
//  :: ghc-prim:GHC.Types.Module{tc 622}
//[LclIdX]
//(main:Main.$trModule{v s1wP} [lidx] :: ghc-prim:GHC.Types.Module{tc 622})
//  = (ghc-prim:GHC.Types.Module{v r7} [gid[DataCon]] :: ghc-prim:GHC.Types.TrName{tc 628}
//                                                       -> ghc-prim:GHC.Types.TrName{tc 628}
//                                                          -> ghc-prim:GHC.Types.Module{tc 622})
//      (sat_s1wQ{v} [lid] :: ghc-prim:GHC.Types.TrName{tc 628})
//      (sat_s1wR{v} [lid] :: ghc-prim:GHC.Types.TrName{tc 628})
//
//-- RHS size: {terms: 7, types: 3, coercions: 0, joins: 0/0}
//(main:Main.main{v s1wS} [lidx] :: ghc-prim:GHC.Types.IO{tc 31Q}
//                                    ghc-prim:GHC.Tuple.(){(w) tc 40})
//  :: ghc-prim:GHC.Types.IO{tc 31Q} ghc-prim:GHC.Tuple.(){(w) tc 40}
//[LclIdX]
//(main:Main.main{v s1wS} [lidx] :: ghc-prim:GHC.Types.IO{tc 31Q}
//                                    ghc-prim:GHC.Tuple.(){(w) tc 40})
//  = case (main:Main.fib{v s1wG} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}
//                                         -> ghc-prim:GHC.Prim.Int#{(w) tc 3s})
//           10#
//    return ghc-prim:GHC.Types.IO{tc 31Q}
//             ghc-prim:GHC.Tuple.(){(w) tc 40}
//    of ((x{v s1wT} [lid] :: ghc-prim:GHC.Prim.Int#{(w) tc 3s}) [Occ=Dead]
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
//(main::Main.main{v s1xa} [lidx] :: ghc-prim:GHC.Types.IO{tc 31Q}
//                                     ghc-prim:GHC.Tuple.(){(w) tc 40})
//  :: ghc-prim:GHC.Types.IO{tc 31Q} ghc-prim:GHC.Tuple.(){(w) tc 40}
//[LclIdX]
//(main::Main.main{v s1xa} [lidx] :: ghc-prim:GHC.Types.IO{tc 31Q}
//                                     ghc-prim:GHC.Tuple.(){(w) tc 40})
//  = (base:GHC.TopHandler.runMainIO{v 01E} [gid] :: forall @(a{tv a1iY} [tv] :: ghc-prim:GHC.Prim.TYPE{(w) tc 32Q}
//                                                                                 'ghc-prim:GHC.Types.LiftedRep{(w) d 63A}).
//                                                     ghc-prim:GHC.Types.IO{tc 31Q}
//                                                       (a{tv a1iY} [tv] :: ghc-prim:GHC.Prim.TYPE{(w) tc 32Q}
//                                                                             'ghc-prim:GHC.Types.LiftedRep{(w) d 63A})
//                                                     -> ghc-prim:GHC.Types.IO{tc 31Q}
//                                                          (a{tv a1iY} [tv] :: ghc-prim:GHC.Prim.TYPE{(w) tc 32Q}
//                                                                                'ghc-prim:GHC.Types.LiftedRep{(w) d 63A}))
//      @ ghc-prim:GHC.Tuple.(){(w) tc 40}
//      (main:Main.main{v s1wS} [lidx] :: ghc-prim:GHC.Types.IO{tc 31Q}
//                                          ghc-prim:GHC.Tuple.(){(w) tc 40})
//