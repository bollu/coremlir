// Main
// Core2MLIR: GenMLIR BeforeCorePrep
hask.module {
  hask.recursive_ref {
    %var_fib_rwj = hask.toplevel_binding {
                     %lambda_10 =
                       hask.lambdaSSA(%i_a12E) {
                         %case_0 =
                           hask.caseSSA %var_i_a12E
                           ["default" ->
                             { ^entry(%var_ds_d1jZ: none):
                                 %app_0  =  hask.apSSA(%var_minus_hash_99, %var_i_a12E)
                                 %lit_1  =  hask.make_i32(1)
                                 %app_2  =  hask.apSSA(%app_0, %lit_1)
                                 %app_3  =  hask.apSSA(%var_fib_rwj, %app_2)
                                 %case_4 =
                                   hask.caseSSA %app_3
                                   ["default" ->
                                     { ^entry(%var_wild_00: none):
                                         %app_4  =  hask.apSSA(%var_fib_rwj, %var_i_a12E)
                                         %case_5 =
                                           hask.caseSSA %app_4
                                           ["default" ->
                                             { ^entry(%var_wild_X5: none):
                                                 %app_5  =  hask.apSSA(%var_plus_hash_98, %var_wild_X5)
                                                 hask.return(%app_5)
                                             }]
                                         %app_7  =  hask.apSSA(%case_5, %var_wild_00)
                                         hask.return(%app_7)
                                     }]
                                 hask.return(%case_4)
                             }]
                           [0 ->
                             { ^entry(%var_ds_d1jZ: none):
                                 hask.return(%var_i_a12E)
                             }]
                           [1 ->
                             { ^entry(%var_ds_d1jZ: none):
                                 hask.return(%var_i_a12E)
                             }]
                         hask.return(%case_0)
                       }
                     hask.return(%lambda_10)
                   }
  }
  %var_$trModule_r1iw =
    hask.toplevel_binding {
      %lit_0  =  hask.make_string("main")
      %app_1  =  hask.apSSA(%var_TrNameS_ra, %lit_0)
      %app_2  =  hask.apSSA(%var_Module_r7, %app_1)
      %lit_3  =  hask.make_string("Main")
      %app_4  =  hask.apSSA(%var_TrNameS_ra, %lit_3)
      %app_5  =  hask.apSSA(%app_2, %app_4)
      hask.return(%app_5)
    }
  %var_main_ryV =
    hask.toplevel_binding {
      %lit_0  =  hask.make_i32(10)
      %app_1  =  hask.apSSA(%var_fib_rwj, %lit_0)
      %case_2 =
        hask.caseSSA %app_1
        ["default" ->
          { ^entry(%var_x_a1hu: none):
              %type_2  =  hask.make_string("TYPEINFO_ERASED")
              %app_3  =  hask.apSSA(%var_return_02O, %type_2)
              %app_4  =  hask.apSSA(%app_3, %var_$fMonadIO_rob)
              %type_5  =  hask.make_string("TYPEINFO_ERASED")
              %app_6  =  hask.apSSA(%app_4, %type_5)
              %app_7  =  hask.apSSA(%app_6, %var_unit_tuple_71)
              hask.return(%app_7)
          }]
      hask.return(%case_2)
    }
  %var_main_01D =
    hask.toplevel_binding {
      %type_0  =  hask.make_string("TYPEINFO_ERASED")
      %app_1  =  hask.apSSA(%var_runMainIO_01E, %type_0)
      %app_2  =  hask.apSSA(%app_1, %var_main_ryV)
      hask.return(%app_2)
    }
  hask.dummy_finish
}
// ============ Haskell Core ========================
//Rec {
//-- RHS size: {terms: 21, types: 4, coercions: 0, joins: 0/0}
//fib [Occ=LoopBreaker] :: Int# -> Int#
//[LclId]
//fib
//  = \ (i_a12E :: Int#) ->
//      case i_a12E of {
//        __DEFAULT ->
//          case fib (-# i_a12E 1#) of wild_00 { __DEFAULT ->
//          (case fib i_a12E of wild_X5 { __DEFAULT -> +# wild_X5 }) wild_00
//          };
//        0# -> i_a12E;
//        1# -> i_a12E
//      }
//end Rec }
//
//-- RHS size: {terms: 5, types: 0, coercions: 0, joins: 0/0}
//$trModule :: Module
//[LclIdX]
//$trModule = Module (TrNameS "main"#) (TrNameS "Main"#)
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