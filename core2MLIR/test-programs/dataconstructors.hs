{-# LANGUAGE MagicHash #-}
{-# LANGUAGE UnboxedTuples #-}
module NonrecSum where
import GHC.Prim
data ConcreteProd = MkConcreteProd Int# Int#
data ConcreteSum = ConcreteLeft Int# | ConcreteRight Int#
data ConcreteRec = MkConcreteRec Int# ConcreteRec
data ConcreteRecSum = ConcreteRecSumCons Int# ConcreteRecSum | ConcreteRecSumNone

data AbstractProd a b = MkAbstractProd a b

f :: ConcreteSum -> ConcreteSum
f x = case x of
        ConcreteLeft i -> ConcreteRight i
        ConcreteRight i -> ConcreteLeft i

sslone :: ConcreteSum; sslone = ConcreteLeft 1#


main :: IO ();
main = let y = f sslone in return ()
