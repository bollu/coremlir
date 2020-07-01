{-# LANGUAGE MagicHash #-}
import GHC.Prim
import GHC.Int

fib :: Int# -> Int#
fib i = case i of
        0# -> 0#
        1# -> 1#
        n -> fib (n -# 1#) +# fib (n -# 2#)

main :: IO ()
main =  putStrLn (show (I# (fib 10#)))
