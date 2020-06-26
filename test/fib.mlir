// module Main where
// 
// $trModule :: Module
// 
// {- Core Size{terms=5 types=0 cos=0 vbinds=0 jbinds=0} -}
// $trModule =
//   APP(GHC.Types.Module
//     (APP(GHC.Types.TrNameS "main"#))
//     (APP(GHC.Types.TrNameS
//        "Main"#)))
// 
// rec {
// fib :: Int -> Int
// 
// {- Core Size{terms=23 types=6 cos=0 vbinds=0 jbinds=0} -}
// fib =
//   λ i →
//     case i of wild {
//       I# ds →
//         case ds of ds {
//           DEFAULT →
//             APP(GHC.Num.+
//               @Int
//               GHC.Num.$fNumInt
//               (APP(Main.fib i))
//               (APP(Main.fib
//                  (APP(GHC.Num.-
//                     @Int
//                     GHC.Num.$fNumInt
//                     i
//                     (APP(GHC.Types.I# 1#)))))))
//           0# → APP(GHC.Types.I# 0#)
//           1# → APP(GHC.Types.I# 1#)
//         }
//     }
// }
// main :: IO ()
// 
// {- Core Size{terms=6 types=1 cos=0 vbinds=0 jbinds=0} -}
// main =
//   APP(System.IO.putStrLn
//     (APP(GHC.Show.show
//        @Int
//        GHC.Show.$fShowInt
//        (APP(Main.fib
//           (APP(GHC.Types.I# 10#)))))))
// 
// main :: IO ()
// 
// {- Core Size{terms=2 types=1 cos=0 vbinds=0 jbinds=0} -}
// main =
//   APP(GHC.TopHandler.runMainIO
//     @() Main.main)




"core.module" () ({
  %altdefault = "core.case_default" () : () -> (none)
  // encode all alts
  %alt_int_0 = "core.case_alt" () : () -> (none)
  %alt_int_1 = "core.case_alt" () : () -> (none)
  %coreIhash = "core.case_alt" () : () -> (none)
  %fact = "core.lambda" () ({ // λ i ...
        ^entry(%i: none):
	          "core.case" () ({ //case i of ...
	          	"core.return" (%i) : (none) -> ()
	          },
	          { // I# ds -> 
	           	  	^entry(%ds: none): 
	           	  			"core.case" (%ds) ({
	           	  				^entry(%ds_arg: i64):	
	           	  				"core.finish" () : () -> ()
	           	  			},
	           	  			{
	           	  				"core.app" () ({ //lhs

	           	  				}, { //rhs

	           	  				}) : () -> () 
           	  					"core.finish" () : () -> ()
	           	  			},
	           	  			{
	           	  				"core.finish" () : () -> ()
	           	  			}) {alt0="default", alt1=1, alt2=2}: (none) -> (none) 
	          		
	          		"core.finish" () : () -> () 			
	           }): ()-> (none) 
	          "core.finish" () : () -> ()
  }): () -> (none)
}): () -> ()