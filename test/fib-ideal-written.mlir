// The ideal syntax that I dreamed up.
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
  // %ghc_types_ihash

  %constructor_ihash  = "core.make_none" () : () ->  none
  %c0 = constant 0 : i32
  %c1 = constant 1 : i32

  %fact = "core.lambda" () ({ // λ i ...
        ^entry(%i: none):
	          "core.case" () ({ //[[case]]  ...
	          	"core.return" (%i) : (none) -> () // case [[i]] of
	          },
	          { // I# ds -> 
	           	  	^entry(%ds: none): 
	           	  			"core.case" (%ds) ({ // [[case]] ds of
	           	  				^entry(%ds_arg: none):	 // [[ds]]
	           	  				"core.return" (%ds_arg) : (none) -> (none)
	           	  			},
	           	  			{ // default -> 
	           	  				"core.app" () ({ //lhs

	           	  				}, { //rhs
	           	  					"core.app" (%fact) : (none) -> (none)

	           	  				}) : () -> () 
           	  					"core.finish" () : () -> ()
	           	  			},
	           	  			{ // 0# -> // APP(GHC.Types.I# 0#)
           	  					"core.app" () ({
           	  						"core.return" (%constructor_ihash) : (none) -> ()
           	  					}, 
           	  					{
           	  						"core.make_i32" (%c0) : (i32) -> (none)
           	  					}) : () -> (none)
	           	  			},
	           	  			{ // 1# ->
	           	  					// APP(GHC.Types.I# 1#)
	           	  					"core.app" () ({
	           	  						"core.return" (%constructor_ihash) : (none) -> ()
	           	  					}, 
	           	  					{
	           	  						"core.make_i32" (%c1) : (i32) -> (none)
	           	  					}) : () -> (none)

	           	  			}) {alt0="default", alt1=1, alt2=2}: (none) -> (none) 
	          				
	           }): ()-> (none) 
  }): () -> (none)
}): () -> (none)
