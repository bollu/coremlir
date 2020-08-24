{-# LANGUAGE MagicHash #-}
{-# LANGUAGE UnboxedTuples #-}
module Fib where
import GHC.Prim
data SimpleInt = MkSimpleInt Int#

plus :: SimpleInt -> SimpleInt -> SimpleInt
plus i j = case i of MkSimpleInt ival -> case j of MkSimpleInt jval -> MkSimpleInt (ival +# jval)


minus :: SimpleInt -> SimpleInt -> SimpleInt
minus i j = case i of MkSimpleInt ival -> case j of MkSimpleInt jval -> MkSimpleInt (ival -# jval)
                            

one :: SimpleInt; one = MkSimpleInt 1#
zero :: SimpleInt; zero = MkSimpleInt 0#

fib :: SimpleInt -> SimpleInt
fib i = 
    case i of
       MkSimpleInt 0# -> zero
       MkSimpleInt 1# -> one
       n -> plus (fib n) (fib (minus n one))

main :: IO ();
main = let x = fib one in return ()
