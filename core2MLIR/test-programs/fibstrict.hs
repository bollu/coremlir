{-# LANGUAGE MagicHash #-}
-- https://downloads.haskell.org/~ghc/8.2.1/docs/html/libraries/base-4.10.0.0/src/GHC-Base.html
-- https://hackage.haskell.org/package/base-4.3.1.0/docs/src/GHC-Int.html
--  https://hackage.haskell.org/package/ghc-prim-0.6.1/docs/GHC-Prim.html
import GHC.Prim

-- | wired in 
-- addInt :: Int# -> Int# -> Int
-- addInt (IntConstructor i) (IntConstructor j) = IntConstructor (i +# j)
-- 
-- 
-- subInt :: Int -> Int -> Int
-- subInt (IntConstructor i) (IntConstructor j) = IntConstructor (i -# j)
-- 
-- oneInt :: Int
-- oneInt = IntConstructor 1#
-- 
-- 
-- -- | loop
-- undefined :: a
-- undefined = undefined
-- 
-- -- | wired in
-- printRawInt :: Int# -> IO ()
-- 
-- -- | wired in 
-- printInt :: Int -> IO ()                                           
-- printInt i = case i of 
--                 IntConstructor ihash -> printRawInt ihash
-- 
-- printRawInt = undefined
fib :: Int# -> Int#
fib i = case i of 0# ->  i; 1# ->  i; _ ->  (fib i) +# (fib (i -# 1#))


main :: IO ();
main = let x = fib 10# in return ()
