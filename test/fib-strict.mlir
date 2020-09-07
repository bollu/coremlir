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
        %fib_rec = hask.ref (@fibstrict):!hask.func<!hask.value, !hask.value>
        %minus_hash = hask.ref (@"-#"): !hask.func<!hask.value, !hask.func<!hask.value, !hask.value>>
        %i_minus = hask.apSSA(%minus_hash: !hask.func<!hask.value, !hask.func<!hask.value, !hask.value>>, %i)
        %lit_one = hask.make_i64(1)
        %i_minus_one = hask.apSSA(%i_minus: !hask.func<!hask.value, !hask.value>, %lit_one)
        %fib_i_minus_one = hask.apSSA(%fib_rec: !hask.func<!hask.value, !hask.value>, %i_minus_one)
        %fib_i = hask.apSSA(%fib_rec: !hask.func<!hask.value, !hask.value>, %i) // what is the type?
        %plus_hash = hask.ref(@"+#"):!hask.func<!hask.value, !hask.func<!hask.value, !hask.value>>
        %plus_fib_i = hask.apSSA(%plus_hash: !hask.func<!hask.value, !hask.func<!hask.value, !hask.value>>, %fib_i)
        %fib_i_plus_fib_i_minus_one = hask.apSSA(%plus_fib_i : !hask.func<!hask.value, !hask.value> , %fib_i_minus_one)
        hask.return(%fib_i_plus_fib_i_minus_one):!hask.value }]
      [0 -> { ^entry(%default_random_name: !hask.value):
        hask.return(%i):!hask.value }]
      [1 -> { ^entry(%default_random_name: !hask.value):
        hask.return(%i):!hask.value }]
      hask.return(%retval) : !hask.value
    }
    hask.return(%lambda) : !hask.func<!hask.value, !hask.value>
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
