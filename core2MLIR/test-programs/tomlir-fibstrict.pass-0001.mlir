// Main
// Core2MLIR: GenMLIR AfterCorePrep
hask.module {
  hask.recursive_ref {
    %var_sat_s1wO = hask.toplevel_binding {
                      %lambda_12 =
                        hask.lambdaSSA(%i_s1wH) {
                          %case_0 =
                            hask.caseSSA %var_i_s1wH
                            ["default" ->
                              { ^entry(%var_ds_s1wI: none):
                                  %app_0  =  hask.apSSA(%var_minus_hash_99, %var_i_s1wH)
                                  %lit_1  =  hask.make_i32(1)
                                  %app_2  =  hask.apSSA(%app_0, %lit_1)
                                  %case_3 =
                                    hask.caseSSA %app_2
                                    ["default" ->
                                      { ^entry(%var_sat_s1wJ: none):
                                          %app_3  =  hask.apSSA(%var_fib_s1wG, %var_sat_s1wJ)
                                          %case_4 =
                                            hask.caseSSA %app_3
                                            ["default" ->
                                              { ^entry(%var_wild_s1wK: none):
                                                  %app_4  =  hask.apSSA(%var_fib_s1wG, %var_i_s1wH)
                                                  %case_5 =
                                                    hask.caseSSA %app_4
                                                    ["default" ->
                                                      { ^entry(%var_wild_s1wL: none):
                                                          %unimpl_5  =  hask.make_i32(42)
                                                          hask.return(%unimpl_5)
                                                      }]
                                                  %case_7 =
                                                    hask.caseSSA %case_5
                                                    ["default" ->
                                                      { ^entry(%var_sat_s1wN: none):
                                                          %app_7  =  hask.apSSA(%var_sat_s1wN, %var_wild_s1wK)
                                                          hask.return(%app_7)
                                                      }]
                                                  hask.return(%case_7)
                                              }]
                                          hask.return(%case_4)
                                      }]
                                  hask.return(%case_3)
                              }]
                            [0 ->
                              { ^entry(%var_ds_s1wI: none):
                                  hask.return(%var_i_s1wH)
                              }]
                            [1 ->
                              { ^entry(%var_ds_s1wI: none):
                                  hask.return(%var_i_s1wH)
                              }]
                          hask.return(%case_0)
                        }
                      hask.return(%lambda_12)
                    }
    %var_fib_s1wG = hask.toplevel_binding {
                      hask.return(%var_sat_s1wO)
                    }
  }
  %var_sat_s1wR =
    hask.toplevel_binding {
      %lit_0  =  hask.make_string("Main")
      %app_1  =  hask.apSSA(%var_TrNameS_ra, %lit_0)
      hask.return(%app_1)
    }
  %var_sat_s1wQ =
    hask.toplevel_binding {
      %lit_0  =  hask.make_string("main")
      %app_1  =  hask.apSSA(%var_TrNameS_ra, %lit_0)
      hask.return(%app_1)
    }
  %var_$trModule_s1wP =
    hask.toplevel_binding {
      %app_0  =  hask.apSSA(%var_Module_r7, %var_sat_s1wQ)
      %app_1  =  hask.apSSA(%app_0, %var_sat_s1wR)
      hask.return(%app_1)
    }
  %var_main_s1wS =
    hask.toplevel_binding {
      %lit_0  =  hask.make_i32(10)
      %app_1  =  hask.apSSA(%var_fib_s1wG, %lit_0)
      %case_2 =
        hask.caseSSA %app_1
        ["default" ->
          { ^entry(%var_x_s1wT: none):
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
  %var_main_s1xa =
    hask.toplevel_binding {
      %type_0  =  hask.make_string("TYPEINFO_ERASED")
      %app_1  =  hask.apSSA(%var_runMainIO_01E, %type_0)
      %app_2  =  hask.apSSA(%app_1, %var_main_s1wS)
      hask.return(%app_2)
    }
  hask.dummy_finish
}