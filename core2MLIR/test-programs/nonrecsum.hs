{-# LANGUAGE MagicHash #-}
{-# LANGUAGE UnboxedTuples #-}
module NonrecSum where
import GHC.Prim
data SimpleSum = SimpleLeft Int# | SimpleRight Int#

f :: SimpleSum -> SimpleSum
f x = case x of
        SimpleLeft i -> SimpleRight i
        SimpleRight i -> SimpleLeft i

sslone :: SimpleSum; sslone = SimpleLeft 1#


main :: IO ();
main = let y = f sslone in return ()
