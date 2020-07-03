// Haskell program
// ===============
// fib :: Int -> Int
// fib i = case i of
//           0 -> 0
//           1 -> 1
//           n -> fib n + fib (n - 1)
// main :: IO (); main = print (fib 10)
//
// Pretty printed core from `ghc-dump-core` [output directly from `desugar`]:
// =========================================================================
// {- Core2MLIR: AfterCorePrep -}
// module Main where
// 
// sat :: TrName
// 
// {- Core Size{terms=2 types=0 cos=0 vbinds=0 jbinds=0} -}
// sat =
//   APP(GHC.Types.TrNameS "Main"#)
// 
// sat :: TrName
// 
// {- Core Size{terms=2 types=0 cos=0 vbinds=0 jbinds=0} -}
// sat =
//   APP(GHC.Types.TrNameS "main"#)
// 
// $trModule :: Module
// 
// {- Core Size{terms=3 types=0 cos=0 vbinds=0 jbinds=0} -}
// $trModule =
//   APP(GHC.Types.Module sat sat)
// 
// rec {
// sat :: Int -> Int
// 
// {- Core Size{terms=31 types=10 cos=0 vbinds=4 jbinds=0} -}
// sat =
//   λ i →
//     case i of wild {
//       I# ds →
//         case ds of ds {
//           DEFAULT →
//             let sat =
//                   let sat =
//                         let  sat = APP(GHC.Types.I# 1#)
//                         in APP(GHC.Num.-
//                              @I nt GHC.Num.$fNumInt i sat)
//                   in APP(Main.fib sat)
//             in let 
//                    
//                    sat = APP(Main.fib i)
//                in APP(GHC.Num.+
//                     @Int GHC.Num.$fNumInt sat sat)
//           0# → APP(GHC.Types.I# 0#)
//           1# → APP(GHC.Types.I# 1#)
//         }
//     }
// 
// fib :: Int -> Int
// 
// {- Core Size{terms=1 types=0 cos=0 vbinds=0 jbinds=0} -}
// fib = sat
// }
// sat :: Int
// 
// {- Core Size{terms=5 types=1 cos=0 vbinds=1 jbinds=0} -}
// sat =
//   let 
//       
//       sat = APP(GHC.Types.I# 10#)
//   in APP(Main.fib sat)
// 
// main :: IO ()
// 
// {- Core Size{terms=3 types=1 cos=0 vbinds=0 jbinds=0} -}
// main =
//   APP(System.IO.print
//     @Int GHC.Show.$fShowInt sat)
// 
// main :: IO ()
// 
// {- Core Size{terms=2 types=1 cos=0 vbinds=0 jbinds=0} -}
// main =
//   APP(GHC.TopHandler.runMainIO
//     @() Main.main)


fib.module {
    %constructor_ihash  = hask.make_data_constructor<"I#"> 
    // This is kind of a lie, we should call it as inbuilt fn or whatever.
    // %constructor_plus = hask.make_data_constructor<"GHC.Num.+"> 
    // %constructor_minus = hask.make_data_constructor<"GHC.Num.-"> 
    // %value_dict_num_int = hask.make_data_constructor<"GHC.Num.$fNumInt">

    // The syntax that I encoded into the parser
    // fib =
    //   λ i →
     //     case i of ds {
    // %fib :: Int -> Int
    %fib = hask.toplevel_binding  {  
      hask.lambda (%i) {
        %resulttop = hask.caseSSA %i 
            ["I#" -> { ^entry(%ds: none):

              //default
            }]


}


// 2020-07-02 17:31:27.678342239 UTC
// 
// Result size of CorePrep
//   = {terms: 59, types: 27, coercions: 5, joins: 0/4}
// 
// -- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
// sat_r35u :: GHC.Types.TrName
// [GblId, Caf=NoCafRefs]
// sat_r35u = GHC.Types.TrNameS "Main"#
// 
// -- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
// sat1_r36y :: GHC.Types.TrName
// [GblId, Caf=NoCafRefs]
// sat1_r36y = GHC.Types.TrNameS "main"#
// 
// -- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
// Main.$trModule :: GHC.Types.Module
// [GblId, Caf=NoCafRefs]
// Main.$trModule = GHC.Types.Module sat1_r36y sat_r35u
// 
// Rec {
// -- RHS size: {terms: 31, types: 10, coercions: 0, joins: 0/4}
// sat2_r36z :: GHC.Types.Int -> GHC.Types.Int
// [GblId, Arity=1]
// sat2_r36z
//   = \ (i_s36C :: GHC.Types.Int) ->
//       case i_s36C of { GHC.Types.I# ds_s36E [Occ=Once!] ->
//       case ds_s36E of {
//         __DEFAULT ->
//           let {
//             sat4_s36G [Occ=Once] :: GHC.Types.Int
//             [LclId]
//             sat4_s36G
//               = let {
//                   sat5_s36H [Occ=Once] :: GHC.Types.Int
//                   [LclId]
//                   sat5_s36H
//                     = let {
//                         sat6_s36I [Occ=Once] :: GHC.Types.Int
//                         [LclId]
//                         sat6_s36I = GHC.Types.I# 1# } in
//                       GHC.Num.- @ GHC.Types.Int GHC.Num.$fNumInt i_s36C sat6_s36I } in
//                 fib_s36a sat5_s36H } in
//           let {
//             sat5_s36J [Occ=Once] :: GHC.Types.Int
//             [LclId]
//             sat5_s36J = fib_s36a i_s36C } in
//           GHC.Num.+ @ GHC.Types.Int GHC.Num.$fNumInt sat5_s36J sat4_s36G;
//         0# -> GHC.Types.I# 0#;
//         1# -> GHC.Types.I# 1#
//       }
//       }
// 
// -- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
// fib_s36a :: GHC.Types.Int -> GHC.Types.Int
// [GblId]
// fib_s36a = sat2_r36z
// end Rec }
// 
// -- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
// sat4_s36K [Occ=Once] :: GHC.Types.Int
// [LclId]
// sat4_s36K = GHC.Types.I# 10#
// 
// -- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
// sat3_r36A :: GHC.Types.Int
// [GblId]
// sat3_r36A = fib_s36a sat4_s36K
// 
// -- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
// Main.main :: GHC.Types.IO ()
// [GblId]
// Main.main
//   = System.IO.print @ GHC.Types.Int GHC.Show.$fShowInt sat3_r36A
// 
// -- RHS size: {terms: 4, types: 3, coercions: 5, joins: 0/0}
// :Main.main :: GHC.Types.IO ()
// [GblId, Arity=1]
// :Main.main
//   = (\ (eta_B1 [Occ=Once] :: GHC.Prim.State# GHC.Prim.RealWorld) ->
//        ((GHC.TopHandler.runMainIO @ () Main.main)
//         `cast` (GHC.Types.N:IO[0] <()>_R
//                 :: GHC.Types.IO ()
//                    ~R# (GHC.Prim.State# GHC.Prim.RealWorld
//                         -> (# GHC.Prim.State# GHC.Prim.RealWorld, () #))))
//          eta_B1)
//     `cast` (Sym (GHC.Types.N:IO[0] <()>_R)
//             :: (GHC.Prim.State# GHC.Prim.RealWorld
//                 -> (# GHC.Prim.State# GHC.Prim.RealWorld, () #))
//                ~R# GHC.Types.IO ())
// 



