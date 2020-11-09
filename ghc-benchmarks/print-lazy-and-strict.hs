{-# LANGUAGE BangPatterns #-}
-- | show how both lazy and strict look the same in the compiler
module Foo where
data Lazy = MkLazy Int Int
data Strict = MkStrict !Int !Int

printLazyAndStrict :: Lazy -> Strict -> IO ()
printLazyAndStrict lz str = 
  case lz of 
   MkLazy la lb -> 
       case str of 
         MkStrict sa sb ->
          do print sa; print sb; print la; print lb

