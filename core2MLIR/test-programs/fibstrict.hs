{-# LANGUAGE MagicHash #-}
-- https://downloads.haskell.org/~ghc/8.2.1/docs/html/libraries/base-4.10.0.0/src/GHC-Base.html
-- https://hackage.haskell.org/package/base-4.3.1.0/docs/src/GHC-Int.html
--  https://hackage.haskell.org/package/ghc-prim-0.6.1/docs/GHC-Prim.html
import GHC.Prim
fibstrict :: Int# -> Int#
fibstrict i = case i of 0# ->  i; 1# ->  i; _ ->  (fibstrict i) +# (fibstrict (i -# 1#))
main :: IO ();
main = let x = fibstrict 10# in return ()
