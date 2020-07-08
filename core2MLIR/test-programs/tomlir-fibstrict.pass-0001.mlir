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
  %main =
    hask.toplevel_binding {
      %lit_0  =  hask.make_i32(10)
      %app_1  =  hask.apSSA(%fib, %lit_0)
      %case_2 =
        hask.caseSSA %app_1
        ["default" ->
          { ^entry(%x_s1wT: none):
              %type_2  =  hask.make_string("TYPEINFO_ERASED")
              %app_3  =  hask.apSSA(%return, %type_2)
              %app_4  =  hask.apSSA(%app_3, %$fMonadIO)
              %type_5  =  hask.make_string("TYPEINFO_ERASED")
              %app_6  =  hask.apSSA(%app_4, %type_5)
              %app_7  =  hask.apSSA(%app_6, %unit_tuple)
              hask.return(%app_7)
          }]
      hask.return(%case_2)
    }
  %main =
    hask.toplevel_binding {
      %type_0  =  hask.make_string("TYPEINFO_ERASED")
      %app_1  =  hask.apSSA(%runMainIO, %type_0)
      %app_2  =  hask.apSSA(%app_1, %main)
      hask.return(%app_2)
    }
  hask.dummy_finish
}
// ============ Haskell Core ========================
//Rec {
//-- RHS size: {terms: 31, types: 10, coercions: 0, joins: 0/1}
//sat_s1wO :: Int# -> Int#
//[LclId]
//sat_s1wO
//  = \ (i_s1wH :: Int#) ->
//      case i_s1wH of {
//        __DEFAULT ->
//          case -# i_s1wH 1# of sat_s1wJ [Occ=Once] { __DEFAULT ->
//          case fib sat_s1wJ of wild_s1wK [Occ=Once] { __DEFAULT ->
//          case case fib i_s1wH of wild_s1wL [Occ=OnceL] { __DEFAULT ->
//               let {
//                 sat_s1wM [Occ=OnceT[0]] :: Int# -> Int#
//                 [LclId]
//                 sat_s1wM
//                   = \ (eta_B1 [Occ=Once] :: Int#) -> +# wild_s1wL eta_B1 } in
//               sat_s1wM
//               }
//          of sat_s1wN [Occ=Once!]
//          { __DEFAULT ->
//          sat_s1wN wild_s1wK
//          }
//          }
//          };
//        0# -> i_s1wH;
//        1# -> i_s1wH
//      }
//
//-- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
//fib [Occ=LoopBreaker] :: Int# -> Int#
//[LclId]
//fib = sat_s1wO
//end Rec }
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1wR :: TrName
//[LclId]
//sat_s1wR = TrNameS "Main"#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1wQ :: TrName
//[LclId]
//sat_s1wQ = TrNameS "main"#
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$trModule :: Module
//[LclIdX]
//$trModule = Module sat_s1wQ sat_s1wR
//
//-- RHS size: {terms: 7, types: 3, coercions: 0, joins: 0/0}
//main :: IO ()
//[LclIdX]
//main
//  = case fib 10# of { __DEFAULT -> return @ IO $fMonadIO @ () () }
//
//-- RHS size: {terms: 2, types: 1, coercions: 0, joins: 0/0}
//main :: IO ()
//[LclIdX]
//main = runMainIO @ () main
//