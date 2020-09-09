// Main
// Core2MLIR: GenMLIR BeforeCorePrep
module {
  // hask.make_data_constructor @"+#"
  // hask.make_data_constructor @"-#"
  // hask.make_data_constructor @"()"

  hask.func @fibstrict {
    %lambda = hask.lambdaSSA(%i: !hask.value) {
      %retval = hask.caseSSA  %i
      ["default" -> { ^entry: // todo: remove this defult
        %fib_rec = hask.ref (@fibstrict):!hask.fn<!hask.value, !hask.thunk>
        %minus_hash = hask.ref (@"-#"): !hask.fn<!hask.value, !hask.fn<!hask.value, !hask.thunk>>
        %lit_one = hask.make_i64(1)
        %i_minus_one_t = hask.apSSA(%minus_hash: !hask.fn<!hask.value, !hask.fn<!hask.value, !hask.thunk>>, %i, %lit_one)
        %i_minus_one = hask.force(%i_minus_one_t)

        %fib_i_minus_one_t = hask.apSSA(%fib_rec: !hask.fn<!hask.value, !hask.thunk>, %i_minus_one)
        %fib_i_minus_one = hask.force(%fib_i_minus_one_t)

        %fib_i_t = hask.apSSA(%fib_rec: !hask.fn<!hask.value, !hask.thunk>, %i)
        %fib_i = hask.force(%fib_i_t)

        %plus_hash = hask.ref(@"+#"):!hask.fn<!hask.value, !hask.fn<!hask.value, !hask.thunk>>
        %fib_sum_t = hask.apSSA(%plus_hash: !hask.fn<!hask.value, !hask.fn<!hask.value, !hask.thunk>>, 
            %fib_i, %fib_i_minus_one)
        %fib_sum  = hask.force(%fib_sum_t)

        hask.return(%fib_sum):!hask.value }]
      [0 -> { ^entry(%default_random_name: !hask.value):
        hask.return(%i):!hask.value }]
      [1 -> { ^entry(%default_random_name: !hask.value):
        hask.return(%i):!hask.value }]
      hask.return(%retval) : !hask.value
    }
    hask.return(%lambda) : !hask.fn<!hask.value, !hask.value>
  }
}

// ============ Haskell Core ========================
//Rec {
//-- RHS size: {terms: 21, types: 4, coercions: 0, joins: 0/0}
//main:Main.fibstrict [Occ=LoopBreaker]
//  :: ghc-prim-0.5.3:GHC.Prim.Int# -> ghc-prim-0.5.3:GHC.Prim.Int#
//[LclId]
//main:Main.fibstrict
//  = \ (i_a12E :: ghc-prim-0.5.3:GHC.Prim.Int#) ->
//      case i_a12E of {
//        __DEFAULT ->
//          case main:Main.fibstrict (ghc-prim-0.5.3:GHC.Prim.-# i_a12E 1#)
//          of wild_00
//          { __DEFAULT ->
//          (case main:Main.fibstrict i_a12E of wild_X5 { __DEFAULT ->
//           ghc-prim-0.5.3:GHC.Prim.+# wild_X5
//           })
//            wild_00
//          };
//        0# -> i_a12E;
//        1# -> i_a12E
//      }
//end Rec }
//
//-- RHS size: {terms: 5, types: 0, coercions: 0, joins: 0/0}
//main:Main.$trModule :: ghc-prim-0.5.3:GHC.Types.Module
//[LclIdX]
//main:Main.$trModule
//  = ghc-prim-0.5.3:GHC.Types.Module
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "main"#)
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "Main"#)
//
//-- RHS size: {terms: 7, types: 3, coercions: 0, joins: 0/0}
//main:Main.main :: ghc-prim-0.5.3:GHC.Types.IO ()
//[LclIdX]
//main:Main.main
//  = case main:Main.fibstrict 10# of { __DEFAULT ->
//    base-4.12.0.0:GHC.Base.return
//      @ ghc-prim-0.5.3:GHC.Types.IO
//      base-4.12.0.0:GHC.Base.$fMonadIO
//      @ ()
//      ghc-prim-0.5.3:GHC.Tuple.()
//    }
//
//-- RHS size: {terms: 2, types: 1, coercions: 0, joins: 0/0}
//main::Main.main :: ghc-prim-0.5.3:GHC.Types.IO ()
//[LclIdX]
//main::Main.main
//  = base-4.12.0.0:GHC.TopHandler.runMainIO @ () main:Main.main
//
